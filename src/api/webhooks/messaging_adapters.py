"""
Messaging Channel Adapters for WhatsApp Business API and Telegram Bot.
Parses conversational vendor sales reports and GPS location pins,
authenticating vendor identities and formatting records for pipeline ingestion.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

try:
    from src.ingestion.schema import (
        TransactionRecord,
        TransactionLocation,
        PaymentMethod,
        TransactionStatus,
        TransactionItem,
    )
except ImportError:
    from ingestion.schema import (
        TransactionRecord,
        TransactionLocation,
        PaymentMethod,
        TransactionStatus,
        TransactionItem,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vendor Channel Identity Registry
# ---------------------------------------------------------------------------

class VendorChannelRegistry:
    """
    Thread-safe mapping of WhatsApp phone numbers and Telegram user IDs
    to verified vendor identities (vendor_id).
    """

    _instance: Optional[VendorChannelRegistry] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._phone_to_vendor: Dict[str, str] = {}
                cls._instance._telegram_to_vendor: Dict[str, str] = {}
                cls._instance._seed_default_mappings()
            return cls._instance

    def _seed_default_mappings(self):
        # Default test vendor mappings for demo / test suites
        self._phone_to_vendor["919876543210"] = "vendor_001"
        self._phone_to_vendor["+919876543210"] = "vendor_001"
        self._phone_to_vendor["918765432109"] = "vendor_002"
        self._telegram_to_vendor["99887766"] = "vendor_001"
        self._telegram_to_vendor["@raju_dosa"] = "vendor_001"
        self._telegram_to_vendor["77665544"] = "vendor_002"

    def register(self, vendor_id: str, phone: Optional[str] = None, telegram_id: Optional[str] = None):
        with self._lock:
            if phone:
                clean_phone = phone.strip().replace(" ", "").replace("-", "")
                self._phone_to_vendor[clean_phone] = vendor_id
                if not clean_phone.startswith("+"):
                    self._phone_to_vendor["+" + clean_phone] = vendor_id
            if telegram_id:
                clean_tg = str(telegram_id).strip()
                self._telegram_to_vendor[clean_tg] = vendor_id

    def resolve_by_phone(self, phone: str) -> Optional[str]:
        clean = phone.strip().replace(" ", "").replace("-", "")
        return self._phone_to_vendor.get(clean) or self._phone_to_vendor.get("+" + clean)

    def resolve_by_telegram(self, telegram_id_or_user: str) -> Optional[str]:
        clean = str(telegram_id_or_user).strip()
        return self._telegram_to_vendor.get(clean) or self._telegram_to_vendor.get("@" + clean.lstrip("@"))


vendor_channel_registry = VendorChannelRegistry()


# ---------------------------------------------------------------------------
# Sales Report Parsing Heuristics
# ---------------------------------------------------------------------------

# Matches e.g.:
# "SALE 250 INR 3 items"
# "SOLD 15 dosa 300 INR"
# "350 INR 5 chai"
# "/sale 450"
# "Cash 200"
CURRENCY_TAGGED_REGEX = re.compile(
    r"(?:(?:rs\.?|inr|₹)\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*(?:rs\.?|inr|₹))",
    re.IGNORECASE,
)

SALES_TEXT_REGEX = re.compile(
    r"(?:(?:sale|sold|order|cash|upi|amount|bill)\s+)?(?:(?:rs|inr|₹)\s*)?(\d+(?:\.\d{1,2})?)\s*(?:(?:rs|inr|₹))?",
    re.IGNORECASE,
)

ITEM_QUANTITY_REGEX = re.compile(
    r"(\d+)\s+([a-zA-Z\s]{3,20})",
    re.IGNORECASE,
)


def parse_sales_text(text: str) -> Tuple[Optional[float], list[TransactionItem]]:
    """
    Extracts total amount and purchased items from free-form vendor messages.
    """
    clean = text.strip()
    amount_inr: Optional[float] = None

    # Check for currency-tagged amount first (e.g. "450 INR", "Rs 450", "₹350")
    curr_match = CURRENCY_TAGGED_REGEX.search(clean)
    if curr_match:
        val = curr_match.group(1) or curr_match.group(2)
        if val:
            amount_inr = round(float(val), 2)
    else:
        # Fall back to sales keyword prefix (e.g. "/sale 250", "SALE 200")
        match = SALES_TEXT_REGEX.search(clean)
        if match:
            amount_inr = round(float(match.group(1)), 2)

    if amount_inr is None:
        return None, []

    items: list[TransactionItem] = []

    # Look for items: e.g. "15 dosa", "2 chai"
    for im in ITEM_QUANTITY_REGEX.finditer(clean):
        qty = int(im.group(1))
        name = im.group(2).strip().lower()
        if name not in ("inr", "rs", "items", "item", "order", "sale", "cash", "upi"):
            items.append(TransactionItem(
                item_name=name.capitalize(),
                quantity=qty,
                unit_price_inr=round(amount_inr / max(1, qty), 2),
            ))

    return amount_inr, items


# ---------------------------------------------------------------------------
# WhatsApp Adapter
# ---------------------------------------------------------------------------

class WhatsAppMessageEvent:
    """Normalized inbound WhatsApp event."""
    def __init__(
        self,
        sender_phone: str,
        vendor_id: Optional[str],
        message_type: str,  # 'text', 'location', 'unsupported'
        text: Optional[str] = None,
        location: Optional[Tuple[float, float, str]] = None,
        raw_message_id: str = "",
    ):
        self.sender_phone = sender_phone
        self.vendor_id = vendor_id
        self.message_type = message_type
        self.text = text
        self.location = location  # (lat, lng, name)
        self.raw_message_id = raw_message_id


def parse_whatsapp_webhook(payload: Dict[str, Any]) -> WhatsAppMessageEvent:
    """
    Extracts message details from a Meta WhatsApp Business Webhook event.
    """
    entry_list = payload.get("entry", [])
    if not entry_list:
        raise ValueError("Invalid WhatsApp webhook: missing 'entry' array")

    changes = entry_list[0].get("changes", [])
    if not changes:
        raise ValueError("Invalid WhatsApp webhook: missing 'changes' array")

    value = changes[0].get("value", {})
    messages = value.get("messages", [])
    if not messages:
        # Status update or delivery receipt
        return WhatsAppMessageEvent(
            sender_phone="",
            vendor_id=None,
            message_type="receipt",
            raw_message_id="",
        )

    msg = messages[0]
    msg_id = msg.get("id", f"WA-{uuid.uuid4().hex[:8]}")
    sender_phone = msg.get("from", "")
    vendor_id = vendor_channel_registry.resolve_by_phone(sender_phone)
    msg_type = msg.get("type", "text")

    if msg_type == "text":
        body = msg.get("text", {}).get("body", "")
        return WhatsAppMessageEvent(
            sender_phone=sender_phone,
            vendor_id=vendor_id,
            message_type="text",
            text=body,
            raw_message_id=msg_id,
        )
    elif msg_type == "location":
        loc = msg.get("location", {})
        lat = float(loc.get("latitude", 0.0))
        lng = float(loc.get("longitude", 0.0))
        name = loc.get("name") or loc.get("address") or "WhatsApp GPS Pin"
        return WhatsAppMessageEvent(
            sender_phone=sender_phone,
            vendor_id=vendor_id,
            message_type="location",
            location=(lat, lng, name),
            raw_message_id=msg_id,
        )

    return WhatsAppMessageEvent(
        sender_phone=sender_phone,
        vendor_id=vendor_id,
        message_type="unsupported",
        raw_message_id=msg_id,
    )


# ---------------------------------------------------------------------------
# Telegram Adapter
# ---------------------------------------------------------------------------

class TelegramMessageEvent:
    """Normalized inbound Telegram bot update."""
    def __init__(
        self,
        user_id: str,
        username: str,
        vendor_id: Optional[str],
        message_type: str,  # 'text', 'location', 'unsupported'
        text: Optional[str] = None,
        location: Optional[Tuple[float, float]] = None,
        update_id: int = 0,
    ):
        self.user_id = user_id
        self.username = username
        self.vendor_id = vendor_id
        self.message_type = message_type
        self.text = text
        self.location = location
        self.update_id = update_id


def parse_telegram_update(update: Dict[str, Any]) -> TelegramMessageEvent:
    """
    Extracts message details from a Telegram Bot API update payload.
    """
    upd_id = update.get("update_id", 0)
    message = update.get("message") or update.get("edited_message")
    if not message:
        return TelegramMessageEvent(
            user_id="",
            username="",
            vendor_id=None,
            message_type="unsupported",
            update_id=upd_id,
        )

    sender = message.get("from", {})
    user_id = str(sender.get("id", ""))
    username = sender.get("username", "")

    # Resolve vendor identity via user_id or @username
    vendor_id = (
        vendor_channel_registry.resolve_by_telegram(user_id)
        or (vendor_channel_registry.resolve_by_telegram(username) if username else None)
    )

    if "location" in message:
        loc = message["location"]
        lat = float(loc.get("latitude", 0.0))
        lng = float(loc.get("longitude", 0.0))
        return TelegramMessageEvent(
            user_id=user_id,
            username=username,
            vendor_id=vendor_id,
            message_type="location",
            location=(lat, lng),
            update_id=upd_id,
        )
    elif "text" in message:
        return TelegramMessageEvent(
            user_id=user_id,
            username=username,
            vendor_id=vendor_id,
            message_type="text",
            text=message["text"],
            update_id=upd_id,
        )

    return TelegramMessageEvent(
        user_id=user_id,
        username=username,
        vendor_id=vendor_id,
        message_type="unsupported",
        update_id=upd_id,
    )
