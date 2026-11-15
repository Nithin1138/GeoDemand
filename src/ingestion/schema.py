"""
Transaction Schemas and Data Models for GeoDemand AI.

Standard telemetry and Point-of-Sale (POS) event schemas (Phase 11.1).
Supports transaction ingestion, vendor location spatial mapping (H3 Res 8),
and status tracking (completed, cancelled, refunded, corrected).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PaymentMethod(str, Enum):
    """Supported real-world payment modes."""
    UPI = "upi"
    CASH = "cash"
    CARD = "card"
    WALLET = "wallet"
    OTHER = "other"


class TransactionStatus(str, Enum):
    """Lifecycle state of the transactional event."""
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    CORRECTED = "corrected"


class TransactionItem(BaseModel):
    """Individual line item sold within a transaction."""
    item_name: str = Field(..., min_length=1, description="Product or dish name")
    quantity: int = Field(..., gt=0, description="Number of units purchased")
    unit_price_inr: float = Field(..., ge=0.0, description="Price per unit in Indian Rupees (INR)")
    total_price_inr: Optional[float] = Field(None, description="Calculated line total in INR")
    product_id: Optional[str] = Field(None, description="Optional foreign key to canonical products catalog")

    @model_validator(mode="after")
    def compute_total(self) -> TransactionItem:
        if self.total_price_inr is None:
            self.total_price_inr = round(float(self.quantity) * float(self.unit_price_inr), 2)
        return self


class TransactionLocation(BaseModel):
    """Geographic positioning of the transaction event."""
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="GPS latitude")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="GPS longitude")
    h3_cell_id: Optional[str] = Field(None, description="Uber H3 resolution 8 spatial cell ID")
    accuracy_meters: Optional[float] = Field(None, ge=0.0, description="GPS accuracy radius in meters")
    is_fallback_location: bool = Field(False, description="Flag indicating location was inferred from fallback")


class TransactionRecord(BaseModel):
    """
    Standard incoming transaction record adhering to Phase 11.1 requirements.
    
    Includes mandatory vendor identification, monetary valuation in INR,
    timestamp provenance, payment method, optional line-item breakdown,
    and geospatial context.
    """
    transaction_id: str = Field(..., min_length=1, description="Unique transaction event identifier")
    vendor_id: str = Field(..., min_length=1, description="Registered mobile vendor identifier")
    customer_id: Optional[str] = Field(None, description="Hashed unique customer ID (for footfall calculation)")
    event_timestamp: datetime = Field(..., description="Timestamp when transaction occurred (UTC)")
    amount_inr: float = Field(..., ge=0.0, description="Net monetary amount in Indian Rupees (INR)")
    payment_method: PaymentMethod = Field(PaymentMethod.UPI, description="Payment channel used")
    status: TransactionStatus = Field(TransactionStatus.COMPLETED, description="Status of transaction")
    items: List[TransactionItem] = Field(default_factory=list, description="Purchased items")
    location: Optional[TransactionLocation] = Field(None, description="Geographic coordinates of transaction")
    correction_of_transaction_id: Optional[str] = Field(
        None, description="Referenced transaction_id for corrected or cancelled orders"
    )
    ingestion_timestamp: Optional[datetime] = Field(
        None, description="Timestamp when record entered ingestion pipeline"
    )
    source: str = Field("real:pos_telemetry", description="Data source origin")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary vendor/device metadata")

    @field_validator("event_timestamp", mode="before")
    @classmethod
    def parse_event_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        elif isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        raise ValueError(f"Invalid timestamp value: {v}")
