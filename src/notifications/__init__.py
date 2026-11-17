"""
GeoDemand AI — Notifications & Alert Subsystem.
"""

from notifications.models import (
    AlertType,
    AlertSeverity,
    NotificationChannel,
    DeliveryStatus,
    ForecastFreshnessMetadata,
    RelocationDetails,
    NotificationPreferences,
    NotificationMessage,
    DeliveryRecord,
)
from notifications.fcm_adapter import FCMAdapter
from notifications.whatsapp_adapter import WhatsAppNotificationAdapter
from notifications.notification_engine import (
    NotificationTriggerService,
    notification_service,
)

__all__ = [
    "AlertType",
    "AlertSeverity",
    "NotificationChannel",
    "DeliveryStatus",
    "ForecastFreshnessMetadata",
    "RelocationDetails",
    "NotificationPreferences",
    "NotificationMessage",
    "DeliveryRecord",
    "FCMAdapter",
    "WhatsAppNotificationAdapter",
    "NotificationTriggerService",
    "notification_service",
]
