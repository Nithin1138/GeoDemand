"""
GeoDemand Webhook Ingestion Package.
Provides cryptographic validation, idempotency, dead-letter recovery,
and adapters for Razorpay, Paytm, PhonePe, WhatsApp, and Telegram.
"""

import sys
try:
    import src.ingestion.schema
    import src.ingestion.validator
    sys.modules["ingestion"] = sys.modules.get("src.ingestion")
    sys.modules["ingestion.schema"] = sys.modules.get("src.ingestion.schema")
    sys.modules["ingestion.validator"] = sys.modules.get("src.ingestion.validator")
except Exception:
    pass

from api.webhooks.security import (
    verify_razorpay_signature,
    verify_paytm_signature,
    verify_phonepe_signature,
    verify_whatsapp_signature,
    verify_telegram_secret,
    validate_event_timestamp,
    mask_sensitive_payload,
    WHATSAPP_VERIFY_TOKEN,
)
from api.webhooks.payment_adapters import (
    parse_razorpay_event,
    parse_paytm_event,
    parse_phonepe_event,
    WebhookParseError,
)
from api.webhooks.messaging_adapters import (
    parse_whatsapp_webhook,
    parse_telegram_update,
    parse_sales_text,
    vendor_channel_registry,
    VendorChannelRegistry,
    WhatsAppMessageEvent,
    TelegramMessageEvent,
)
from api.webhooks.dead_letter import (
    dead_letter_queue,
    DeadLetterQueue,
    DeadLetterRecord,
)
from api.webhooks.service import (
    webhook_service,
    WebhookIngestionService,
)

__all__ = [
    "verify_razorpay_signature",
    "verify_paytm_signature",
    "verify_phonepe_signature",
    "verify_whatsapp_signature",
    "verify_telegram_secret",
    "validate_event_timestamp",
    "mask_sensitive_payload",
    "WHATSAPP_VERIFY_TOKEN",
    "parse_razorpay_event",
    "parse_paytm_event",
    "parse_phonepe_event",
    "WebhookParseError",
    "parse_whatsapp_webhook",
    "parse_telegram_update",
    "parse_sales_text",
    "vendor_channel_registry",
    "VendorChannelRegistry",
    "WhatsAppMessageEvent",
    "TelegramMessageEvent",
    "dead_letter_queue",
    "DeadLetterQueue",
    "DeadLetterRecord",
    "webhook_service",
    "WebhookIngestionService",
]
