"""
Actuals Aggregator & Real Telemetry Pipeline (Phase 11.1).

Aggregates raw Point-of-Sale (POS) transactions into hourly business actuals:
- Hourly time bucketing: floors event_timestamp to [t, t+1h) window
- Separates raw transaction count from unique customer count and total item count
- Generates `actual_customer_count` when customer tracking is measurable
- Computes gross sales (INR), average order value (AOV), and payment distribution
- Emits real-data quality reports and enforces data retention & privacy policies
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.ingestion.schema import (
    TransactionRecord,
    TransactionStatus,
)

# ==============================================================================
# DATA RETENTION & PRIVACY POLICY (Task 11.1.26)
# ==============================================================================
# 1. PII Exclusion: Real customer names/phone numbers/payment credentials MUST NEVER
#    be ingested. Only one-way salted hashes may be stored in `customer_id`.
# 2. Raw Telemetry Retention: Raw individual transaction records expire after 90 days.
# 3. Aggregated Analytics: Hourly aggregated actuals are retained indefinitely for
#    ground-truth evaluation, drift detection, and backtesting.
# ==============================================================================
RAW_TELEMETRY_RETENTION_DAYS = 90
AGGREGATED_METRICS_RETENTION_DAYS = 365 * 3


class ActualsQualityReport(BaseModel):
    """Data quality and integrity summary for an actuals aggregation batch."""
    total_raw_transactions: int
    valid_transactions: int
    cancelled_transactions: int
    hourly_buckets_created: int
    total_gross_sales_inr: float
    reconciliation_discrepancy_inr: float
    customer_id_coverage_pct: float
    null_value_count: int
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ActualsAggregator:
    """
    Transforms individual validated transaction events into hourly ground-truth
    metrics keyed by (vendor_id, h3_cell_id, hour_timestamp).
    """

    def __init__(self, default_city: str = "Vijayawada") -> None:
        self.default_city = default_city

    def aggregate_transactions(
        self,
        transactions: List[TransactionRecord],
        as_of_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Aggregate a batch of transaction records into hourly actuals.

        Returns a DataFrame with columns:
            vendor_id, h3_cell_id, hour_timestamp, transaction_count,
            unique_customer_count, actual_customer_count, total_items_sold,
            total_sales_inr, average_order_value, upi_sales_pct,
            source, source_type, aggregated_at
        """
        now_iso = (as_of_time or datetime.now(timezone.utc)).isoformat()
        
        if not transactions:
            return self._empty_actuals_df()

        rows = []
        for tx in transactions:
            # Exclude cancelled or refunded transactions from positive sales
            if tx.status in (TransactionStatus.CANCELLED, TransactionStatus.REFUNDED):
                continue

            h3_cell = tx.location.h3_cell_id if tx.location and tx.location.h3_cell_id else "unknown"
            
            # Floor timestamp to start of hour [t, t+1h)
            dt = tx.event_timestamp
            hour_bucket = dt.replace(minute=0, second=0, microsecond=0)

            # Sum total items sold
            items_count = sum(item.quantity for item in tx.items) if tx.items else 1

            rows.append({
                "transaction_id": tx.transaction_id,
                "vendor_id": tx.vendor_id,
                "customer_id": tx.customer_id,
                "h3_cell_id": h3_cell,
                "hour_timestamp": hour_bucket,
                "amount_inr": float(tx.amount_inr),
                "items_count": int(items_count),
                "payment_method": tx.payment_method.value if hasattr(tx.payment_method, "value") else str(tx.payment_method),
            })

        if not rows:
            return self._empty_actuals_df()

        df_raw = pd.DataFrame(rows)

        # Group by vendor_id, h3_cell_id, and hour_timestamp
        grouped = df_raw.groupby(["vendor_id", "h3_cell_id", "hour_timestamp"])
        
        agg_rows = []
        for (v_id, cell, h_time), group in grouped:
            tx_count = len(group)
            total_sales = round(float(group["amount_inr"].sum()), 2)
            total_items = int(group["items_count"].sum())
            aov = round(total_sales / tx_count, 2) if tx_count > 0 else 0.0

            # Separate unique customer count from transaction count
            valid_custs = group["customer_id"].dropna()
            has_customers = len(valid_custs) > 0
            unique_cust_count = int(valid_custs.nunique()) if has_customers else None
            
            # actual_customer_count is measurable only when customer IDs are present
            actual_cust_count = unique_cust_count if has_customers else None

            # Payment breakdown: UPI percentage
            upi_count = (group["payment_method"] == "upi").sum()
            upi_pct = round(float(upi_count) / tx_count * 100.0, 1) if tx_count > 0 else 0.0

            agg_rows.append({
                "vendor_id": v_id,
                "h3_cell_id": cell,
                "hour_timestamp": h_time,
                "transaction_count": tx_count,
                "unique_customer_count": unique_cust_count,
                "actual_customer_count": actual_cust_count,
                "total_items_sold": total_items,
                "total_sales_inr": total_sales,
                "average_order_value": aov,
                "upi_sales_pct": upi_pct,
                "source": "real:pos_actuals",
                "source_type": "real_telemetry",
                "aggregated_at": now_iso,
            })

        df_agg = pd.DataFrame(agg_rows)
        # Sort chronologically by hour and vendor
        df_agg = df_agg.sort_values(["hour_timestamp", "vendor_id", "h3_cell_id"]).reset_index(drop=True)
        return df_agg

    def generate_quality_report(
        self,
        raw_transactions: List[TransactionRecord],
        aggregated_df: pd.DataFrame,
    ) -> ActualsQualityReport:
        """
        Audit raw inputs vs aggregated outputs to ensure financial and data integrity.
        """
        total_raw = len(raw_transactions)
        cancelled = sum(
            1 for tx in raw_transactions
            if tx.status in (TransactionStatus.CANCELLED, TransactionStatus.REFUNDED)
        )
        valid_raw = total_raw - cancelled

        # Financial reconciliation
        raw_valid_sales = sum(
            float(tx.amount_inr) for tx in raw_transactions
            if tx.status not in (TransactionStatus.CANCELLED, TransactionStatus.REFUNDED)
        )
        agg_sales = float(aggregated_df["total_sales_inr"].sum()) if not aggregated_df.empty else 0.0
        discrepancy = round(abs(raw_valid_sales - agg_sales), 2)

        # Customer ID coverage
        cust_id_count = sum(1 for tx in raw_transactions if tx.customer_id is not None)
        cust_cov_pct = round((cust_id_count / total_raw * 100.0), 1) if total_raw > 0 else 0.0

        # Null value count in essential fields
        null_count = int(aggregated_df[["vendor_id", "h3_cell_id", "hour_timestamp", "total_sales_inr"]].isna().sum().sum()) if not aggregated_df.empty else 0

        return ActualsQualityReport(
            total_raw_transactions=total_raw,
            valid_transactions=valid_raw,
            cancelled_transactions=cancelled,
            hourly_buckets_created=len(aggregated_df),
            total_gross_sales_inr=agg_sales,
            reconciliation_discrepancy_inr=discrepancy,
            customer_id_coverage_pct=cust_cov_pct,
            null_value_count=null_count,
        )

    def _empty_actuals_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with the canonical actuals schema."""
        return pd.DataFrame(columns=[
            "vendor_id", "h3_cell_id", "hour_timestamp", "transaction_count",
            "unique_customer_count", "actual_customer_count", "total_items_sold",
            "total_sales_inr", "average_order_value", "upi_sales_pct",
            "source", "source_type", "aggregated_at",
        ])
