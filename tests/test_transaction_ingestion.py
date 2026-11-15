"""
Unit and Integration Tests for Real Transaction Ingestion Pipeline (Phase 11.1).
"""

from datetime import datetime, timezone, timedelta
import pytest
from src.ingestion.schema import (
    TransactionRecord,
    TransactionItem,
    TransactionLocation,
    PaymentMethod,
    TransactionStatus,
)
from src.ingestion.validator import (
    TransactionIngestionPipeline,
    DEFAULT_FALLBACK_H3_CELL,
)


@pytest.fixture
def pipeline():
    p = TransactionIngestionPipeline(max_delay_hours=12.0)
    p.register_vendor_location("vendor_001", 16.5176, 80.5278)
    return p


def test_standard_valid_transaction_schema(pipeline):
    """Test standard valid transaction parsing and fields (Phase 11.1 Tasks 2–9)."""
    now = datetime.now(timezone.utc)
    record = TransactionRecord(
        transaction_id="tx_1001",
        vendor_id="vendor_001",
        customer_id="cust_abc123",
        event_timestamp=now,
        amount_inr=240.0,
        payment_method=PaymentMethod.UPI,
        status=TransactionStatus.COMPLETED,
        items=[
            TransactionItem(item_name="Masala Dosa", quantity=2, unit_price_inr=60.0),
            TransactionItem(item_name="Filter Coffee", quantity=3, unit_price_inr=40.0),
        ],
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )

    result = pipeline.ingest_transaction(record)
    assert result.success is True
    assert result.status_code == "ACCEPTED"
    assert result.record is not None
    assert result.record.amount_inr == 240.0
    assert len(result.record.items) == 2
    assert result.record.items[0].total_price_inr == 120.0
    assert result.record.items[1].total_price_inr == 120.0


def test_h3_resolution_8_spatial_mapping(pipeline):
    """Test GPS coordinates are automatically mapped to H3 Resolution 8 (Task 15)."""
    now = datetime.now(timezone.utc)
    record = TransactionRecord(
        transaction_id="tx_1002",
        vendor_id="vendor_002",
        event_timestamp=now,
        amount_inr=150.0,
        payment_method=PaymentMethod.CASH,
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )

    result = pipeline.ingest_transaction(record)
    assert result.success is True
    assert result.record.location.h3_cell_id is not None
    # Resolution 8 cells are 15-character hex strings
    assert len(result.record.location.h3_cell_id) == 15
    assert result.record.location.is_fallback_location is False


def test_duplicate_transaction_detection(pipeline):
    """Test duplicate transaction IDs are rejected (Task 11)."""
    now = datetime.now(timezone.utc)
    record = TransactionRecord(
        transaction_id="tx_dup_999",
        vendor_id="vendor_001",
        event_timestamp=now,
        amount_inr=80.0,
        payment_method=PaymentMethod.UPI,
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )

    r1 = pipeline.ingest_transaction(record)
    assert r1.success is True

    # Duplicate call with identical transaction_id
    r2 = pipeline.ingest_transaction(record)
    assert r2.success is False
    assert r2.is_duplicate is True
    assert r2.status_code == "DUPLICATE_TRANSACTION_REJECTED"
    assert "already exists" in r2.errors[0]


def test_delayed_transaction_handling(pipeline):
    """Test delayed transaction events are flagged with appropriate warnings (Task 12)."""
    # Event happened 20 hours ago, pipeline max delay is 12 hours
    old_time = datetime.now(timezone.utc) - timedelta(hours=20)
    record = TransactionRecord(
        transaction_id="tx_delayed_555",
        vendor_id="vendor_001",
        event_timestamp=old_time,
        amount_inr=120.0,
        payment_method=PaymentMethod.CARD,
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )

    result = pipeline.ingest_transaction(record)
    assert result.success is True
    assert result.is_delayed is True
    assert any("delayed" in w.lower() for w in result.warnings)


def test_cancelled_and_corrected_transactions(pipeline):
    """Test handling of cancelled, refunded, and corrected transactions (Task 13)."""
    now = datetime.now(timezone.utc)
    
    # 1. Original order
    original = TransactionRecord(
        transaction_id="tx_orig_101",
        vendor_id="vendor_001",
        event_timestamp=now,
        amount_inr=300.0,
        payment_method=PaymentMethod.UPI,
        status=TransactionStatus.COMPLETED,
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )
    pipeline.ingest_transaction(original)
    assert pipeline.is_cancelled("tx_orig_101") is False

    # 2. Cancellation of the original order
    cancel_tx = TransactionRecord(
        transaction_id="tx_cancel_102",
        vendor_id="vendor_001",
        event_timestamp=now + timedelta(minutes=5),
        amount_inr=300.0,
        payment_method=PaymentMethod.UPI,
        status=TransactionStatus.CANCELLED,
        correction_of_transaction_id="tx_orig_101",
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )
    res_cancel = pipeline.ingest_transaction(cancel_tx)
    assert res_cancel.success is True
    assert res_cancel.is_cancelled is True
    assert pipeline.is_cancelled("tx_orig_101") is True

    # 3. Corrected replacement transaction
    corrected_tx = TransactionRecord(
        transaction_id="tx_corr_103",
        vendor_id="vendor_001",
        event_timestamp=now + timedelta(minutes=10),
        amount_inr=250.0,
        payment_method=PaymentMethod.UPI,
        status=TransactionStatus.CORRECTED,
        correction_of_transaction_id="tx_orig_101",
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )
    res_corr = pipeline.ingest_transaction(corrected_tx)
    assert res_corr.success is True
    assert res_corr.is_correction is True


def test_missing_location_fallback_to_registered_vendor(pipeline):
    """Test missing GPS falls back to registered vendor coordinates (Task 14)."""
    now = datetime.now(timezone.utc)
    # vendor_001 is registered at (16.5176, 80.5278)
    record = TransactionRecord(
        transaction_id="tx_nolocation_001",
        vendor_id="vendor_001",
        event_timestamp=now,
        amount_inr=95.0,
        payment_method=PaymentMethod.WALLET,
        location=None,  # No location provided
    )

    result = pipeline.ingest_transaction(record)
    assert result.success is True
    assert result.record.location is not None
    assert result.record.location.latitude == 16.5176
    assert result.record.location.longitude == 80.5278
    assert result.record.location.is_fallback_location is True
    assert len(result.record.location.h3_cell_id) == 15


def test_missing_location_fallback_to_city_default(pipeline):
    """Test missing GPS with unknown vendor falls back to default city cell (Task 14)."""
    now = datetime.now(timezone.utc)
    record = TransactionRecord(
        transaction_id="tx_nolocation_unknown",
        vendor_id="vendor_unregistered_999",
        event_timestamp=now,
        amount_inr=50.0,
        payment_method=PaymentMethod.CASH,
        location=None,
    )

    result = pipeline.ingest_transaction(record)
    assert result.success is True
    assert result.record.location.is_fallback_location is True
    assert result.record.location.h3_cell_id == DEFAULT_FALLBACK_H3_CELL
