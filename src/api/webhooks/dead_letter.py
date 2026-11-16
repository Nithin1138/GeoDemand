"""
Dead-Letter Queue (DLQ) & Idempotency Store for Payment & Messaging Webhooks.

Provides:
- Deduplication and idempotency protection against duplicate webhook delivery
- In-memory and persistent buffer for unrecoverable or malformed webhook events
- Credential-sanitized event payload storage
- Failure recovery & manual / automated DLQ retry execution
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from api.webhooks.security import mask_sensitive_payload
try:
    from src.ingestion.schema import TransactionRecord
    from src.ingestion.validator import IngestionResult, TransactionIngestionPipeline
except ImportError:
    from ingestion.schema import TransactionRecord
    from ingestion.validator import IngestionResult, TransactionIngestionPipeline

logger = logging.getLogger(__name__)


@dataclass
class DeadLetterRecord:
    dlq_id: str
    provider: str
    error_reason: str
    sanitized_payload: Dict[str, Any]
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    status: str = "FAILED"  # FAILED, RECOVERED, DROPPED
    last_error: Optional[str] = None
    recovered_transaction_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dlq_id": self.dlq_id,
            "provider": self.provider,
            "error_reason": self.error_reason,
            "sanitized_payload": self.sanitized_payload,
            "received_at": self.received_at.isoformat() if isinstance(self.received_at, datetime) else str(self.received_at),
            "retry_count": self.retry_count,
            "status": self.status,
            "last_error": self.last_error,
            "recovered_transaction_id": self.recovered_transaction_id,
        }


class DeadLetterQueue:
    """
    Thread-safe in-memory Dead-Letter Queue with idempotent deduplication tracking.
    """

    _instance: Optional[DeadLetterQueue] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._records: Dict[str, DeadLetterRecord] = {}
                cls._instance._seen_events: Set[str] = set()
                cls._instance._op_lock = threading.RLock()
            return cls._instance

    def reset_state_for_testing(self):
        with self._op_lock:
            self._records.clear()
            self._seen_events.clear()

    # -----------------------------------------------------------------------
    # Idempotency Tracking
    # -----------------------------------------------------------------------

    def make_event_key(self, provider: str, tx_id: str) -> str:
        return f"{provider.lower().strip()}:{str(tx_id).strip()}"

    def is_duplicate(self, provider: str, tx_id: str) -> bool:
        """Returns True if the event has already been successfully ingested."""
        with self._op_lock:
            return self.make_event_key(provider, tx_id) in self._seen_events

    def mark_processed(self, provider: str, tx_id: str):
        """Records event key in idempotency store."""
        with self._op_lock:
            self._seen_events.add(self.make_event_key(provider, tx_id))

    # -----------------------------------------------------------------------
    # Dead-Letter Operations
    # -----------------------------------------------------------------------

    def enqueue(
        self,
        provider: str,
        raw_payload: Dict[str, Any],
        error_reason: str,
    ) -> DeadLetterRecord:
        """
        Enqueues a failed or unparseable webhook event into the Dead-Letter Queue.
        Payload is strictly sanitized so credentials are never preserved in DLQ.
        """
        with self._op_lock:
            dlq_id = f"DLQ-{uuid.uuid4().hex[:10].upper()}"
            sanitized = mask_sensitive_payload(raw_payload)
            record = DeadLetterRecord(
                dlq_id=dlq_id,
                provider=provider.lower().strip(),
                error_reason=error_reason,
                sanitized_payload=sanitized,
                status="FAILED",
            )
            self._records[dlq_id] = record
            logger.warning(f"Enqueued dead-letter event {dlq_id} for provider '{provider}': {error_reason}")
            return record

    def list_records(
        self,
        provider: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[DeadLetterRecord]:
        with self._op_lock:
            res = list(self._records.values())
            if provider:
                res = [r for r in res if r.provider == provider.lower().strip()]
            if status:
                res = [r for r in res if r.status == status.upper().strip()]
            return res[:limit]

    def get_record(self, dlq_id: str) -> Optional[DeadLetterRecord]:
        with self._op_lock:
            return self._records.get(dlq_id)

    def mark_recovered(self, dlq_id: str, tx_id: str):
        with self._op_lock:
            rec = self._records.get(dlq_id)
            if rec:
                rec.status = "RECOVERED"
                rec.recovered_transaction_id = tx_id
                rec.retry_count += 1

    def mark_retry_failed(self, dlq_id: str, error_msg: str):
        with self._op_lock:
            rec = self._records.get(dlq_id)
            if rec:
                rec.retry_count += 1
                rec.last_error = error_msg


dead_letter_queue = DeadLetterQueue()
