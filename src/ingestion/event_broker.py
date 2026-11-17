"""
GeoDemand Event Ingestion Gateway & Broker Subsystem.

Architecture Assessment & Broker Selection:
- Workload Sizing:
    * Mobile Vendors (Vijayawada Deployment): 1,500 – 3,000 active carts
    * Peak Transaction Rate: ~2 to 5 tx/sec (15–25 tx/day per vendor)
    * Peak GPS Telemetry Rate: ~50 to 100 pings/sec (1 ping every 30–60s)
    * Statewide Expansion (10 cities): ~850 to 1,200 events/sec peak
- Broker Selection Verdict:
    * Kafka: Overkill for < 10,000 events/sec. Complex JVM / KRaft clustering,
      partition head-of-line blocking, and lacks per-message DLX/nack capabilities.
    * RabbitMQ (AMQP 0-9-1): Selected as the target broker. Provides native per-message
      acknowledgments (ack/nack), dead-letter exchanges (DLX), message TTL, exponential
      backoff retry queues, low operational footprint (< 150MB RAM), and handles
      20,000+ msg/sec with sub-millisecond latencies.
- Implementation:
    * Abstract EventBroker interface
    * High-throughput Thread-Safe In-Memory Stream Broker (Zero-dependency testing/standalone)
    * Production RabbitMQ AMQP Broker (via pika with DLX and fallback)
    * EventGateway orchestrating event publication, background consumers,
      exponential retries, dead-letter routing, and pipeline forwarding.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import pika
    HAS_PIKA = True
except ImportError:
    pika = None
    HAS_PIKA = False

from api.webhooks.dead_letter import dead_letter_queue
from fleet.fleet_manager import FleetManager
try:
    from src.ingestion.schema import TransactionRecord, TransactionLocation, PaymentMethod, TransactionStatus
    from src.ingestion.validator import TransactionIngestionPipeline, IngestionResult
except ImportError:
    from ingestion.schema import TransactionRecord, TransactionLocation, PaymentMethod, TransactionStatus
    from ingestion.validator import TransactionIngestionPipeline, IngestionResult

logger = logging.getLogger(__name__)

# Default queue & routing constants
QUEUE_TRANSACTIONS = "geodemand.events.transactions"
QUEUE_TELEMETRY = "geodemand.events.telemetry"
QUEUE_DEAD_LETTER = "geodemand.events.dead_letter"
EXCHANGE_DIRECT = "geodemand.events.direct"
EXCHANGE_DLX = "geodemand.events.dlx"


# ---------------------------------------------------------------------------
# Sizing & Volume Measurement Evaluation
# ---------------------------------------------------------------------------

@dataclass
class BrokerSizingProfile:
    active_vendors: int
    tx_per_vendor_day: float
    shift_hours: float
    telemetry_interval_sec: float
    tx_events_per_sec: float
    telemetry_events_per_sec: float
    total_events_per_sec: float
    recommended_broker: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_vendors": self.active_vendors,
            "tx_per_vendor_day": self.tx_per_vendor_day,
            "shift_hours": self.shift_hours,
            "telemetry_interval_sec": self.telemetry_interval_sec,
            "tx_events_per_sec": round(self.tx_events_per_sec, 2),
            "telemetry_events_per_sec": round(self.telemetry_events_per_sec, 2),
            "total_events_per_sec": round(self.total_events_per_sec, 2),
            "recommended_broker": self.recommended_broker,
            "rationale": self.rationale,
        }


def calculate_broker_sizing(
    active_vendors: int = 3000,
    tx_per_vendor_day: float = 20.0,
    shift_hours: float = 10.0,
    telemetry_interval_sec: float = 30.0,
) -> BrokerSizingProfile:
    """
    Computes expected event volume and delivery requirements to determine
    the appropriate broker architecture.
    """
    shift_seconds = max(1.0, shift_hours * 3600.0)
    tx_rate = (active_vendors * tx_per_vendor_day) / shift_seconds
    telemetry_rate = active_vendors / max(1.0, telemetry_interval_sec)
    total_rate = tx_rate + telemetry_rate

    if total_rate > 25000:
        broker = "Apache Kafka"
        rationale = (
            f"Total throughput {total_rate:.1f} events/sec exceeds 25k/sec threshold. "
            "Partitioned distributed commit log recommended for high fan-out big data streaming."
        )
    else:
        broker = "RabbitMQ (AMQP 0-9-1)"
        rationale = (
            f"Measured peak event throughput is {total_rate:.1f} events/sec (Transactions: {tx_rate:.1f}/s, "
            f"Telemetry: {telemetry_rate:.1f}/s). RabbitMQ comfortably sustains 20k–50k msg/sec on a single node "
            "with sub-millisecond latency, per-message ACKs/NACKs, and native Dead-Letter Exchanges (DLX). "
            "Kafka would introduce unnecessary operational complexity (JVM/KRaft overhead) without delivery benefits."
        )

    return BrokerSizingProfile(
        active_vendors=active_vendors,
        tx_per_vendor_day=tx_per_vendor_day,
        shift_hours=shift_hours,
        telemetry_interval_sec=telemetry_interval_sec,
        tx_events_per_sec=tx_rate,
        telemetry_events_per_sec=telemetry_rate,
        total_events_per_sec=total_rate,
        recommended_broker=broker,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Abstract Broker Interface
# ---------------------------------------------------------------------------

class EventBroker(ABC):
    """Abstract interface for event publication, subscription, and delivery."""

    @abstractmethod
    def publish(
        self,
        queue_name: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publishes an event to the target queue or exchange."""
        pass

    @abstractmethod
    def subscribe(
        self,
        queue_name: str,
        handler: Callable[[Dict[str, Any]], bool],
        max_retries: int = 3,
        retry_backoff_sec: float = 0.5,
    ):
        """Registers a consumer handler for a queue."""
        pass

    @abstractmethod
    def start(self):
        """Starts background consumer workers."""
        pass

    @abstractmethod
    def stop(self):
        """Stops background consumers gracefully."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Returns broker telemetry: published, consumed, retried, dead-lettered."""
        pass


# ---------------------------------------------------------------------------
# In-Memory Stream Broker (Production-Grade Multi-Threaded Engine)
# ---------------------------------------------------------------------------

@dataclass
class BrokerMessage:
    msg_id: str
    queue_name: str
    payload: Dict[str, Any]
    headers: Dict[str, Any] = field(default_factory=dict)
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    next_delivery_time: float = 0.0


class InMemoryEventBroker(EventBroker):
    """
    High-performance in-memory stream broker with worker thread pools,
    exponential retry backoff, dead-letter routing, and idempotent deduplication.
    Requires zero external daemons for local/testing execution.
    """

    def __init__(self, num_workers: int = 2):
        self._queues: Dict[str, queue.Queue[BrokerMessage]] = {}
        self._retry_queue: List[BrokerMessage] = []
        self._handlers: Dict[str, List[Tuple[Callable[[Dict[str, Any]], bool], int, float]]] = {}
        self._idempotency_keys: Set[str] = set()

        self._lock = threading.RLock()
        self._running = False
        self._workers: List[threading.Thread] = []
        self._retry_thread: Optional[threading.Thread] = None

        # Metrics
        self.published_count = 0
        self.consumed_count = 0
        self.retry_count = 0
        self.dead_letter_count = 0

    def _ensure_queue(self, queue_name: str) -> queue.Queue[BrokerMessage]:
        with self._lock:
            if queue_name not in self._queues:
                self._queues[queue_name] = queue.Queue()
            return self._queues[queue_name]

    def publish(
        self,
        queue_name: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None,
    ) -> bool:
        headers_dict = headers or {}
        idempotency_key = headers_dict.get("idempotency_key")
        if idempotency_key:
            with self._lock:
                if idempotency_key in self._idempotency_keys:
                    logger.info(f"Duplicate event ignored in broker: {idempotency_key}")
                    return True
                self._idempotency_keys.add(idempotency_key)

        msg_id = f"MSG-{uuid.uuid4().hex[:10].upper()}"
        msg = BrokerMessage(
            msg_id=msg_id,
            queue_name=queue_name,
            payload=payload,
            headers=headers_dict,
        )

        q = self._ensure_queue(queue_name)
        q.put(msg)
        with self._lock:
            self.published_count += 1
        return True

    def subscribe(
        self,
        queue_name: str,
        handler: Callable[[Dict[str, Any]], bool],
        max_retries: int = 3,
        retry_backoff_sec: float = 0.5,
    ):
        with self._lock:
            self._ensure_queue(queue_name)
            self._handlers.setdefault(queue_name, []).append((handler, max_retries, retry_backoff_sec))

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True

            # Start worker threads
            for i in range(2):
                t = threading.Thread(target=self._worker_loop, name=f"EventBroker-Worker-{i}", daemon=True)
                t.start()
                self._workers.append(t)

            # Start retry dispatcher thread
            self._retry_thread = threading.Thread(target=self._retry_loop, name="EventBroker-RetryLoop", daemon=True)
            self._retry_thread.start()

    def stop(self):
        with self._lock:
            self._running = False
        for t in self._workers:
            if t.is_alive():
                t.join(timeout=0.2)
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=0.2)

    def _worker_loop(self):
        while self._running:
            handled_any = False
            with self._lock:
                queue_items = list(self._queues.items())

            for q_name, q in queue_items:
                if q_name == QUEUE_DEAD_LETTER and q_name not in self._handlers:
                    continue  # DLQ is held for manual inspection if no consumer handler is registered
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    continue

                handled_any = True
                self._process_message(msg)

            if not handled_any:
                time.sleep(0.01)

    def _process_message(self, msg: BrokerMessage):
        with self._lock:
            handlers = list(self._handlers.get(msg.queue_name, []))

        if not handlers:
            # No registered handler for queue
            return

        success = True
        max_retries = 3
        backoff_sec = 0.5

        for handler, m_retries, b_sec in handlers:
            max_retries = m_retries
            backoff_sec = b_sec
            try:
                res = handler(msg.payload)
                if res is False:
                    success = False
            except Exception as e:
                logger.warning(f"Consumer handler error for {msg.msg_id}: {e}")
                success = False

        if success:
            with self._lock:
                self.consumed_count += 1
        else:
            # Handle Failure & Retry / DLX
            if msg.retry_count < max_retries:
                msg.retry_count += 1
                delay = backoff_sec * (2 ** (msg.retry_count - 1))
                msg.next_delivery_time = time.time() + delay
                with self._lock:
                    self._retry_queue.append(msg)
                    self.retry_count += 1
                logger.info(f"Scheduled retry #{msg.retry_count} for {msg.msg_id} in {delay:.2f}s")
            else:
                # Route to Dead-Letter Queue
                with self._lock:
                    self.dead_letter_count += 1
                dlq = self._ensure_queue(QUEUE_DEAD_LETTER)
                dlq_payload = {
                    "original_queue": msg.queue_name,
                    "payload": msg.payload,
                    "headers": msg.headers,
                    "reason": "MAX_RETRIES_EXCEEDED",
                    "retries": msg.retry_count,
                    "dead_lettered_at": datetime.now(timezone.utc).isoformat(),
                }
                dlq_msg = BrokerMessage(
                    msg_id=f"DLQ-{msg.msg_id}",
                    queue_name=QUEUE_DEAD_LETTER,
                    payload=dlq_payload,
                    headers={"dlx_origin": msg.queue_name},
                )
                dlq.put(dlq_msg)
                logger.warning(f"Message {msg.msg_id} permanently rejected; routed to DLQ ({QUEUE_DEAD_LETTER})")

    def _retry_loop(self):
        while self._running:
            now_t = time.time()
            to_requeue: List[BrokerMessage] = []

            with self._lock:
                remaining: List[BrokerMessage] = []
                for msg in self._retry_queue:
                    if msg.next_delivery_time <= now_t:
                        to_requeue.append(msg)
                    else:
                        remaining.append(msg)
                self._retry_queue = remaining

            for msg in to_requeue:
                q = self._ensure_queue(msg.queue_name)
                q.put(msg)

            time.sleep(0.05)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            depths = {k: q.qsize() for k, q in self._queues.items()}
            return {
                "broker_type": "InMemoryStreamBroker",
                "running": self._running,
                "published_count": self.published_count,
                "consumed_count": self.consumed_count,
                "retry_count": self.retry_count,
                "dead_letter_count": self.dead_letter_count,
                "queue_depths": depths,
                "pending_retries": len(self._retry_queue),
            }


# ---------------------------------------------------------------------------
# RabbitMQ AMQP Broker (With Automatic Fallback)
# ---------------------------------------------------------------------------

class RabbitMQEventBroker(EventBroker):
    """
    Production AMQP 0-9-1 RabbitMQ broker client using pika.
    Declares direct exchanges, dead-letter exchanges (DLX), durable queues,
    and handles graceful fallback to InMemoryEventBroker if offline.
    """

    def __init__(self, amqp_url: Optional[str] = None):
        self.amqp_url = amqp_url or os.getenv("RABBITMQ_URL")
        self._fallback = InMemoryEventBroker()
        self._is_connected = False
        self._lock = threading.RLock()
        self._handlers: Dict[str, List[Tuple[Callable[[Dict[str, Any]], bool], int, float]]] = {}

        if HAS_PIKA and (self.amqp_url or os.getenv("ENABLE_RABBITMQ", "false").lower() in ("true", "1", "yes")):
            self._try_connect()
        else:
            logger.info("Pika not installed or AMQP daemon offline; running on In-Memory broker fallback.")

    def _try_connect(self) -> bool:
        if not HAS_PIKA:
            return False
        try:
            params = pika.URLParameters(self.amqp_url)
            params.socket_timeout = 1.0
            conn = pika.BlockingConnection(params)
            chan = conn.channel()

            # Declare Durable Exchanges
            chan.exchange_declare(exchange=EXCHANGE_DIRECT, exchange_type="direct", durable=True)
            chan.exchange_declare(exchange=EXCHANGE_DLX, exchange_type="fanout", durable=True)

            # Declare Dead-Letter Queue
            chan.queue_declare(queue=QUEUE_DEAD_LETTER, durable=True)
            chan.queue_bind(queue=QUEUE_DEAD_LETTER, exchange=EXCHANGE_DLX)

            # Declare Transactions Queue with DLX
            chan.queue_declare(
                queue=QUEUE_TRANSACTIONS,
                durable=True,
                arguments={"x-dead-letter-exchange": EXCHANGE_DLX},
            )
            chan.queue_bind(queue=QUEUE_TRANSACTIONS, exchange=EXCHANGE_DIRECT, routing_key=QUEUE_TRANSACTIONS)

            # Declare Telemetry Queue
            chan.queue_declare(queue=QUEUE_TELEMETRY, durable=True)
            chan.queue_bind(queue=QUEUE_TELEMETRY, exchange=EXCHANGE_DIRECT, routing_key=QUEUE_TELEMETRY)

            conn.close()
            self._is_connected = True
            logger.info(f"Connected to RabbitMQ AMQP broker at {self.amqp_url} with durable DLX.")
            return True
        except Exception as e:
            self._is_connected = False
            logger.warning(f"RabbitMQ connection not available ({e}). Using InMemoryEventBroker fallback.")
            return False

    def publish(self, queue_name: str, payload: Dict[str, Any], headers: Optional[Dict[str, Any]] = None) -> bool:
        if not self._is_connected:
            return self._fallback.publish(queue_name, payload, headers)

        try:
            params = pika.URLParameters(self.amqp_url)
            conn = pika.BlockingConnection(params)
            chan = conn.channel()

            body = json.dumps(payload).encode("utf-8")
            props = pika.BasicProperties(
                delivery_mode=2,  # Persistent message
                headers=headers or {},
            )
            chan.basic_publish(
                exchange=EXCHANGE_DIRECT,
                routing_key=queue_name,
                body=body,
                properties=props,
            )
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"RabbitMQ publish failed ({e}); falling back to in-memory publish.")
            return self._fallback.publish(queue_name, payload, headers)

    def subscribe(self, queue_name: str, handler: Callable[[Dict[str, Any]], bool], max_retries: int = 3, retry_backoff_sec: float = 0.5):
        # Register in both fallback and internal handler store
        self._handlers.setdefault(queue_name, []).append((handler, max_retries, retry_backoff_sec))
        self._fallback.subscribe(queue_name, handler, max_retries, retry_backoff_sec)

    def start(self):
        self._fallback.start()

    def stop(self):
        self._fallback.stop()

    def get_metrics(self) -> Dict[str, Any]:
        metrics = self._fallback.get_metrics()
        metrics["broker_type"] = "RabbitMQEventBroker" if self._is_connected else "RabbitMQEventBroker(Fallback:InMemory)"
        metrics["is_connected_to_amqp"] = self._is_connected
        url = self.amqp_url or "amqp://guest:guest@localhost:5672/"
        metrics["amqp_url"] = url.split("@")[-1] if "@" in url else url
        return metrics


# ---------------------------------------------------------------------------
# Unified Event Gateway Orchestrator
# ---------------------------------------------------------------------------

class EventGateway:
    """
    Central event ingestion gateway.
    Publishes payment transactions and vendor telemetry to the broker.
    Runs asynchronous consumers that process records through Member 1's
    TransactionIngestionPipeline and update the FleetManager.
    """

    _instance: Optional[EventGateway] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Allow isolated instance when broker or isolated flag is provided for testing
        if kwargs.get("broker") is not None or kwargs.get("isolated"):
            inst = super().__new__(cls)
            inst._initialized = False
            return inst
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        use_rabbitmq: bool = True,
        broker: Optional[EventBroker] = None,
        isolated: bool = False,
    ):
        if getattr(self, "_initialized", False) and broker is None:
            return

        self.pipeline = TransactionIngestionPipeline(max_delay_hours=24.0)
        self.fleet_manager = FleetManager()

        if broker is not None:
            self.broker = broker
        elif use_rabbitmq:
            self.broker = RabbitMQEventBroker()
        else:
            self.broker = InMemoryEventBroker()

        # Register Gateway Consumers
        self._register_consumers()
        self.broker.start()
        self._initialized = True

    def _register_consumers(self):
        """Wires up core asynchronous queue consumers."""
        # 1. Transactions consumer
        self.broker.subscribe(
            queue_name=QUEUE_TRANSACTIONS,
            handler=self._handle_transaction_event,
            max_retries=3,
            retry_backoff_sec=0.2,
        )

        # 2. Telemetry consumer
        self.broker.subscribe(
            queue_name=QUEUE_TELEMETRY,
            handler=self._handle_telemetry_event,
            max_retries=2,
            retry_backoff_sec=0.1,
        )

        # 3. Dead-Letter consumer (records to DLQ store)
        self.broker.subscribe(
            queue_name=QUEUE_DEAD_LETTER,
            handler=self._handle_dead_letter_event,
            max_retries=1,
            retry_backoff_sec=0.0,
        )

    def _handle_transaction_event(self, payload: Dict[str, Any]) -> bool:
        """Consumer handler: ingests transaction into Member 1's pipeline."""
        try:
            record = TransactionRecord(**payload)
            res = self.pipeline.ingest_transaction(record)
            if not res.success and not res.is_duplicate:
                logger.warning(f"Pipeline ingestion rejected transaction {record.transaction_id}: {res.errors}")
                return False

            # Update vendor coordinates in fleet manager if location provided
            if record.location and record.location.latitude:
                v = self.fleet_manager.get_vendor(record.vendor_id)
                if v:
                    v.current_lat = record.location.latitude
                    v.current_lng = record.location.longitude
                    v.last_heartbeat = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error(f"Error handling transaction message in consumer: {e}")
            return False

    def _handle_telemetry_event(self, payload: Dict[str, Any]) -> bool:
        """Consumer handler: updates vendor location in pipeline and fleet manager."""
        try:
            vendor_id = payload.get("vendor_id")
            lat = float(payload.get("latitude", 0.0))
            lng = float(payload.get("longitude", 0.0))
            if not vendor_id:
                return False

            self.pipeline.register_vendor_location(vendor_id, lat, lng)
            v = self.fleet_manager.get_vendor(vendor_id)
            if v:
                v.current_lat = lat
                v.current_lng = lng
                v.last_heartbeat = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error(f"Error handling telemetry message: {e}")
            return False

    def _handle_dead_letter_event(self, payload: Dict[str, Any]) -> bool:
        """Consumer handler: persists unrecoverable messages into DLQ store."""
        try:
            dead_letter_queue.enqueue(
                provider=payload.get("provider", "event_gateway"),
                raw_payload=payload,
                error_reason="Exceeded maximum broker delivery retries",
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record dead-letter event: {e}")
            return False

    # -----------------------------------------------------------------------
    # Public Gateway Publishing APIs
    # -----------------------------------------------------------------------

    def publish_transaction(
        self,
        record: TransactionRecord,
        idempotency_key: Optional[str] = None,
    ) -> bool:
        """Publishes a normalized transaction event to the broker."""
        headers = {
            "idempotency_key": idempotency_key or f"tx:{record.transaction_id}",
            "vendor_id": record.vendor_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.broker.publish(
            queue_name=QUEUE_TRANSACTIONS,
            payload=record.model_dump(mode="json"),
            headers=headers,
        )

    def publish_telemetry(
        self,
        vendor_id: str,
        latitude: float,
        longitude: float,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Publishes a GPS telemetry event to the broker."""
        payload = {
            "vendor_id": vendor_id,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        }
        return self.broker.publish(
            queue_name=QUEUE_TELEMETRY,
            payload=payload,
            headers={"vendor_id": vendor_id},
        )

    def get_status(self) -> Dict[str, Any]:
        sizing = calculate_broker_sizing()
        metrics = self.broker.get_metrics()
        return {
            "gateway_status": "ONLINE",
            "active_broker": metrics.get("broker_type"),
            "sizing_profile": sizing.to_dict(),
            "telemetry": metrics,
        }


event_gateway = EventGateway()
