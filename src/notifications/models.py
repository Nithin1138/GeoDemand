"""
Data models, schemas, and enums for the GeoDemand AI Notification Subsystem.
Supports impending-rain, event-demand, and demand-spike alerts with physical
relocation constraints, forecast freshness verification, preferences, and delivery telemetry.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional


class AlertType(str, enum.Enum):
    IMPENDING_RAIN = "IMPENDING_RAIN"
    EVENT_DEMAND = "EVENT_DEMAND"
    DEMAND_SPIKE = "DEMAND_SPIKE"
    GENERAL_ALERT = "GENERAL_ALERT"


class AlertSeverity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class NotificationChannel(str, enum.Enum):
    FCM = "FCM"
    WHATSAPP = "WHATSAPP"
    TELEGRAM = "TELEGRAM"
    SMS = "SMS"


class DeliveryStatus(str, enum.Enum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    SUPPRESSED_OPT_OUT = "SUPPRESSED_OPT_OUT"
    SUPPRESSED_DUPLICATE = "SUPPRESSED_DUPLICATE"
    SUPPRESSED_STALE_DATA = "SUPPRESSED_STALE_DATA"
    SUPPRESSED_QUIET_HOURS = "SUPPRESSED_QUIET_HOURS"


@dataclass
class ForecastFreshnessMetadata:
    """Telemetry capturing the provenance, age, and reliability of weather & demand signals."""
    source_type: str = "real_live"
    source_name: str = "open_meteo"
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cache_age_seconds: float = 0.0
    is_reliable: bool = True
    reliability_reason: str = "Live verified forecast observation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": self.retrieved_at,
            "cache_age_seconds": round(self.cache_age_seconds, 1),
            "is_reliable": self.is_reliable,
            "reliability_reason": self.reliability_reason,
        }


@dataclass
class RelocationDetails:
    """Actionable relocation instructions attached to an alert notification."""
    destination_h3_cell: str
    destination_name: str
    distance_km: float
    travel_time_minutes: int
    expected_profit_uplift_inr: float
    recommended_departure_time: str   # ISO or HH:MM
    expected_arrival_time: str        # ISO or HH:MM
    is_move_recommended: bool
    forecast_freshness: ForecastFreshnessMetadata
    decision_reason: str = "High customer demand uplift"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "destination_h3_cell": self.destination_h3_cell,
            "destination_name": self.destination_name,
            "distance_km": round(self.distance_km, 2),
            "travel_time_minutes": self.travel_time_minutes,
            "expected_profit_uplift_inr": round(self.expected_profit_uplift_inr, 2),
            "recommended_departure_time": self.recommended_departure_time,
            "expected_arrival_time": self.expected_arrival_time,
            "is_move_recommended": self.is_move_recommended,
            "forecast_freshness": self.forecast_freshness.to_dict(),
            "decision_reason": self.decision_reason,
        }


@dataclass
class NotificationPreferences:
    """Vendor configuration governing alert delivery channels, quiet hours, and opt-outs."""
    vendor_id: str
    opt_out: bool = False
    channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.FCM, NotificationChannel.WHATSAPP]
    )
    fcm_token: Optional[str] = None
    whatsapp_phone: Optional[str] = None
    alert_impending_rain: bool = True
    alert_event_demand: bool = True
    alert_demand_spike: bool = True
    min_profit_uplift_inr: float = 40.0
    quiet_hours_start: Optional[str] = "23:00"  # HH:MM local
    quiet_hours_end: Optional[str] = "06:00"    # HH:MM local
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor_id": self.vendor_id,
            "opt_out": self.opt_out,
            "channels": [c.value if isinstance(c, NotificationChannel) else str(c) for c in self.channels],
            "fcm_token": self.fcm_token,
            "whatsapp_phone": self.whatsapp_phone,
            "alert_impending_rain": self.alert_impending_rain,
            "alert_event_demand": self.alert_event_demand,
            "alert_demand_spike": self.alert_demand_spike,
            "min_profit_uplift_inr": self.min_profit_uplift_inr,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class NotificationMessage:
    """Structured alert message entity dispatched to messaging adapters."""
    notification_id: str
    vendor_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    body: str
    relocation: Optional[RelocationDetails] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=45))
    idempotency_key: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: Optional[datetime] = None) -> bool:
        now = current_time or datetime.now(timezone.utc)
        return now >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "vendor_id": self.vendor_id,
            "alert_type": self.alert_type.value if isinstance(self.alert_type, AlertType) else str(self.alert_type),
            "severity": self.severity.value if isinstance(self.severity, AlertSeverity) else str(self.severity),
            "title": self.title,
            "body": self.body,
            "relocation": self.relocation.to_dict() if self.relocation else None,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "idempotency_key": self.idempotency_key,
            "metadata": self.metadata,
        }


@dataclass
class DeliveryRecord:
    """Delivery telemetry record tracking channel status, timestamps, and failures."""
    record_id: str
    notification_id: str
    vendor_id: str
    channel: NotificationChannel
    recipient: str
    status: DeliveryStatus
    error_message: Optional[str] = None
    dispatched_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    retry_count: int = 0
    upstream_message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "notification_id": self.notification_id,
            "vendor_id": self.vendor_id,
            "channel": self.channel.value if isinstance(self.channel, NotificationChannel) else str(self.channel),
            "recipient": self.recipient,
            "status": self.status.value if isinstance(self.status, DeliveryStatus) else str(self.status),
            "error_message": self.error_message,
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
            "retry_count": self.retry_count,
            "upstream_message_id": self.upstream_message_id,
        }
