"""
Transaction Ingestion, Validation and Spatial Enrichment Pipeline.

Implements Phase 11.1 requirements:
- Ingestion validation & data sanitization
- Duplicate transaction ID detection
- Delayed event detection
- Corrected and cancelled transaction processing
- Missing location handling and fallback logic
- Spatial mapping to Uber H3 Resolution 8
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, Tuple
from pydantic import BaseModel, Field

try:
    import h3
except ImportError:
    h3 = None

from src.ingestion.schema import (
    TransactionRecord,
    TransactionLocation,
    TransactionStatus,
)

logger = logging.getLogger(__name__)

# Default Vijayawada city center centroid & H3 cell for spatial fallback
DEFAULT_VIJAYAWADA_LAT = 16.5062
DEFAULT_VIJAYAWADA_LNG = 80.6480
DEFAULT_FALLBACK_H3_CELL = "88619aa6c7fffff"


class IngestionResult(BaseModel):
    """Result of transaction validation and spatial enrichment."""
    success: bool
    status_code: str
    record: Optional[TransactionRecord] = None
    is_duplicate: bool = False
    is_delayed: bool = False
    is_cancelled: bool = False
    is_correction: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TransactionIngestionPipeline:
    """
    Stateful ingestion pipeline providing deduplication, spatial resolution,
    delay detection, and lifecycle status accounting.
    """

    def __init__(
        self,
        max_delay_hours: float = 24.0,
        vendor_locations: Optional[Dict[str, Tuple[float, float]]] = None,
        h3_resolution: int = 8,
    ) -> None:
        """
        Args:
            max_delay_hours: Maximum latency before an incoming event is flagged delayed.
            vendor_locations: Optional map of vendor_id -> (lat, lng) default location.
            h3_resolution: Target H3 spatial resolution level (strictly 8 in V1/V2).
        """
        self.max_delay_hours = max_delay_hours
        self.vendor_locations = vendor_locations or {}
        self.h3_resolution = h3_resolution
        
        # State tracking stores
        self._seen_transaction_ids: Set[str] = set()
        self._cancelled_transaction_ids: Set[str] = set()
        self._corrections_map: Dict[str, str] = {}  # corrected_id -> original_id

    def register_vendor_location(self, vendor_id: str, lat: float, lng: float) -> None:
        """Register a default base location for a vendor."""
        self.vendor_locations[vendor_id] = (lat, lng)

    def ingest_transaction(
        self,
        transaction: TransactionRecord,
        as_of_time: Optional[datetime] = None,
    ) -> IngestionResult:
        """
        Validate, enrich, and process an incoming transaction record.
        """
        now = as_of_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
            
        record = transaction.model_copy(deep=True)
        record.ingestion_timestamp = now

        errors: list[str] = []
        warnings: list[str] = []

        # 1. Duplicate Detection
        if record.transaction_id in self._seen_transaction_ids:
            return IngestionResult(
                success=False,
                status_code="DUPLICATE_TRANSACTION_REJECTED",
                record=record,
                is_duplicate=True,
                errors=[f"Transaction ID {record.transaction_id} already exists in ingestion buffer."],
            )

        # 2. Delayed Event Detection
        time_diff = now - record.event_timestamp
        is_delayed = time_diff > timedelta(hours=self.max_delay_hours)
        if is_delayed:
            warnings.append(
                f"Transaction event is delayed by {time_diff.total_seconds() / 3600:.1f} hours "
                f"(threshold: {self.max_delay_hours}h)."
            )

        # 3. Lifecycle Status (Cancelled / Refunded / Corrected)
        is_cancelled = record.status in (TransactionStatus.CANCELLED, TransactionStatus.REFUNDED)
        is_correction = record.status == TransactionStatus.CORRECTED

        if is_cancelled:
            self._cancelled_transaction_ids.add(record.transaction_id)
            if record.correction_of_transaction_id:
                self._cancelled_transaction_ids.add(record.correction_of_transaction_id)
                warnings.append(
                    f"Cancelled transaction {record.transaction_id} marks original "
                    f"{record.correction_of_transaction_id} as cancelled."
                )

        if is_correction and record.correction_of_transaction_id:
            self._corrections_map[record.transaction_id] = record.correction_of_transaction_id
            warnings.append(
                f"Transaction {record.transaction_id} updates previous transaction "
                f"{record.correction_of_transaction_id}."
            )

        # 4. Spatial Enrichment & H3 Resolution 8 Mapping
        record.location = self._enrich_location(record.vendor_id, record.location, warnings)

        # Register transaction as seen
        self._seen_transaction_ids.add(record.transaction_id)

        return IngestionResult(
            success=True,
            status_code="ACCEPTED",
            record=record,
            is_delayed=is_delayed,
            is_cancelled=is_cancelled,
            is_correction=is_correction,
            errors=errors,
            warnings=warnings,
        )

    def _enrich_location(
        self,
        vendor_id: str,
        loc: Optional[TransactionLocation],
        warnings: list[str],
    ) -> TransactionLocation:
        """Resolve GPS coordinates into an H3 Resolution 8 cell, handling missing fallbacks."""
        lat: Optional[float] = None
        lng: Optional[float] = None
        is_fallback = False

        if loc and loc.latitude is not None and loc.longitude is not None:
            lat = loc.latitude
            lng = loc.longitude
            acc = loc.accuracy_meters
        elif vendor_id in self.vendor_locations:
            lat, lng = self.vendor_locations[vendor_id]
            acc = 50.0  # nominal vendor stall radius
            is_fallback = True
            warnings.append(f"Missing transaction GPS: fell back to registered location for vendor '{vendor_id}'.")
        else:
            lat, lng = DEFAULT_VIJAYAWADA_LAT, DEFAULT_VIJAYAWADA_LNG
            acc = 500.0
            is_fallback = True
            warnings.append(f"Missing transaction GPS and no registered vendor base: fell back to city default.")

        # Compute H3 Resolution 8 cell ID
        h3_cell = DEFAULT_FALLBACK_H3_CELL
        if h3 is not None and lat is not None and lng is not None:
            try:
                h3_cell = h3.latlng_to_cell(lat, lng, self.h3_resolution)
            except Exception as e:
                logger.warning(f"Failed H3 indexing for lat={lat}, lng={lng}: {e}")
                h3_cell = DEFAULT_FALLBACK_H3_CELL

        return TransactionLocation(
            latitude=lat,
            longitude=lng,
            h3_cell_id=h3_cell,
            accuracy_meters=acc if loc is None else (loc.accuracy_meters or acc),
            is_fallback_location=is_fallback,
        )

    def is_cancelled(self, transaction_id: str) -> bool:
        """Check if a transaction ID has been cancelled or refunded."""
        return transaction_id in self._cancelled_transaction_ids

    def clear(self) -> None:
        """Clear the ingestion memory buffers."""
        self._seen_transaction_ids.clear()
        self._cancelled_transaction_ids.clear()
        self._corrections_map.clear()
