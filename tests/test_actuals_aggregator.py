"""
Unit and Integration Tests for Actuals Aggregator & Real Data Telemetry (Phase 11.1 Part 2).
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from src.ingestion.schema import (
    TransactionRecord,
    TransactionItem,
    TransactionLocation,
    PaymentMethod,
    TransactionStatus,
)
from src.data.actuals_aggregator import (
    ActualsAggregator,
    RAW_TELEMETRY_RETENTION_DAYS,
    AGGREGATED_METRICS_RETENTION_DAYS,
)


@pytest.fixture
def aggregator():
    return ActualsAggregator()


def test_hourly_bucketing_and_sales_aggregation(aggregator):
    """Test transactions within the same hour bucket into [t, t+1h) (Tasks 16, 21)."""
    base_time = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    txs = [
        TransactionRecord(
            transaction_id="tx_1",
            vendor_id="vendor_001",
            event_timestamp=base_time + timedelta(minutes=5),  # 12:05
            amount_inr=100.0,
            payment_method=PaymentMethod.UPI,
            items=[TransactionItem(item_name="Item A", quantity=2, unit_price_inr=50.0)],
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
        TransactionRecord(
            transaction_id="tx_2",
            vendor_id="vendor_001",
            event_timestamp=base_time + timedelta(minutes=35),  # 12:35
            amount_inr=150.0,
            payment_method=PaymentMethod.CASH,
            items=[TransactionItem(item_name="Item B", quantity=1, unit_price_inr=150.0)],
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
    ]

    df = aggregator.aggregate_transactions(txs)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["vendor_id"] == "vendor_001"
    assert row["h3_cell_id"] == "88619aa6c7fffff"
    assert row["hour_timestamp"] == base_time
    assert row["transaction_count"] == 2
    assert row["total_sales_inr"] == 250.0
    assert row["total_items_sold"] == 3
    assert row["average_order_value"] == 125.0
    assert row["source"] == "real:pos_actuals"
    assert row["source_type"] == "real_telemetry"


def test_separation_of_transactions_customers_and_items(aggregator):
    """Test clear distinction between transaction count, unique customer count, and items count (Tasks 17, 18, 20)."""
    base_time = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)
    
    # 3 transactions from 2 unique customers, buying 7 total items
    txs = [
        TransactionRecord(
            transaction_id="tx_cust_1",
            vendor_id="vendor_001",
            customer_id="cust_alice",
            event_timestamp=base_time + timedelta(minutes=10),
            amount_inr=120.0,
            items=[TransactionItem(item_name="Dosa", quantity=2, unit_price_inr=60.0)],
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
        TransactionRecord(
            transaction_id="tx_cust_2",
            vendor_id="vendor_001",
            customer_id="cust_bob",
            event_timestamp=base_time + timedelta(minutes=20),
            amount_inr=180.0,
            items=[TransactionItem(item_name="Idli", quantity=4, unit_price_inr=45.0)],
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
        TransactionRecord(
            transaction_id="tx_cust_3",
            vendor_id="vendor_001",
            customer_id="cust_alice",  # Repeated customer in same hour
            event_timestamp=base_time + timedelta(minutes=40),
            amount_inr=40.0,
            items=[TransactionItem(item_name="Coffee", quantity=1, unit_price_inr=40.0)],
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
    ]

    df = aggregator.aggregate_transactions(txs)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["transaction_count"] == 3
    assert row["unique_customer_count"] == 2
    assert row["actual_customer_count"] == 2
    assert row["total_items_sold"] == 7
    assert row["total_sales_inr"] == 340.0


def test_customer_count_unmeasured_when_missing(aggregator):
    """Test actual_customer_count is null/None when customer IDs cannot be tracked (Task 20)."""
    base_time = datetime(2026, 10, 15, 16, 0, 0, tzinfo=timezone.utc)
    
    txs = [
        TransactionRecord(
            transaction_id="tx_anon_1",
            vendor_id="vendor_002",
            customer_id=None,  # Cash transaction without customer loyalty/id
            event_timestamp=base_time + timedelta(minutes=15),
            amount_inr=50.0,
            payment_method=PaymentMethod.CASH,
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        )
    ]

    df = aggregator.aggregate_transactions(txs)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["transaction_count"] == 1
    assert pd.isna(row["actual_customer_count"]) or row["actual_customer_count"] is None
    assert pd.isna(row["unique_customer_count"]) or row["unique_customer_count"] is None


def test_cancellation_exclusion_from_aggregations(aggregator):
    """Test cancelled or refunded transactions are excluded from positive revenue metrics (Task 21)."""
    base_time = datetime(2026, 10, 15, 18, 0, 0, tzinfo=timezone.utc)
    
    txs = [
        TransactionRecord(
            transaction_id="tx_valid",
            vendor_id="vendor_001",
            event_timestamp=base_time + timedelta(minutes=10),
            amount_inr=200.0,
            status=TransactionStatus.COMPLETED,
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
        TransactionRecord(
            transaction_id="tx_cancelled",
            vendor_id="vendor_001",
            event_timestamp=base_time + timedelta(minutes=25),
            amount_inr=150.0,
            status=TransactionStatus.CANCELLED,
            correction_of_transaction_id="tx_valid",
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
    ]

    df = aggregator.aggregate_transactions(txs)
    assert len(df) == 1
    row = df.iloc[0]
    # Only the valid transaction should be included in gross sales
    assert row["transaction_count"] == 1
    assert row["total_sales_inr"] == 200.0


def test_actuals_quality_report_reconciliation(aggregator):
    """Test quality report calculates reconciliation discrepancy and coverage (Task 24)."""
    base_time = datetime(2026, 10, 15, 20, 0, 0, tzinfo=timezone.utc)
    
    txs = [
        TransactionRecord(
            transaction_id="tx_rep_1",
            vendor_id="vendor_001",
            customer_id="cust_x",
            event_timestamp=base_time + timedelta(minutes=10),
            amount_inr=100.0,
            status=TransactionStatus.COMPLETED,
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
        TransactionRecord(
            transaction_id="tx_rep_2",
            vendor_id="vendor_001",
            customer_id=None,
            event_timestamp=base_time + timedelta(minutes=20),
            amount_inr=50.0,
            status=TransactionStatus.CANCELLED,
            location=TransactionLocation(h3_cell_id="88619aa6c7fffff"),
        ),
    ]

    df = aggregator.aggregate_transactions(txs)
    report = aggregator.generate_quality_report(txs, df)

    assert report.total_raw_transactions == 2
    assert report.valid_transactions == 1
    assert report.cancelled_transactions == 1
    assert report.total_gross_sales_inr == 100.0
    assert report.reconciliation_discrepancy_inr == 0.0
    assert report.customer_id_coverage_pct == 50.0
    assert report.null_value_count == 0


def test_data_retention_and_privacy_policy():
    """Verify retention policy limits are explicitly declared (Task 26)."""
    assert RAW_TELEMETRY_RETENTION_DAYS == 90
    assert AGGREGATED_METRICS_RETENTION_DAYS >= 365
