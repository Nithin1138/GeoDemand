"""
Webhook Security, Cryptographic Signature Verification & Data Sanitization.

Provides verification for:
- Razorpay: HMAC-SHA256 (X-Razorpay-Signature)
- Paytm: SHA256 Checksum (X-Paytm-Signature or body.head.signature)
- PhonePe: Base64 + SHA256 + Salt Key & Index (X-VERIFY)
- WhatsApp Business: HMAC-SHA256 (X-Hub-Signature-256)
- Telegram: Secret token header (X-Telegram-Bot-Api-Secret-Token)
- Timestamp freshness validation (prevents replay and clock manipulation)
- Sensitive credential masking (protects PII and payment data in logs/DLQ)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Default Secrets & Environment Keys
# ---------------------------------------------------------------------------

RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "test_razorpay_secret_123")
PAYTM_MERCHANT_KEY = os.getenv("PAYTM_MERCHANT_KEY", "test_paytm_key_123")
PHONEPE_SALT_KEY = os.getenv("PHONEPE_SALT_KEY", "test_phonepe_salt_123")
PHONEPE_SALT_INDEX = os.getenv("PHONEPE_SALT_INDEX", "1")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "geodemand_wa_token_2026")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "test_wa_secret_123")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "geodemand_tg_secret_2026")

SENSITIVE_KEY_PATTERNS = re.compile(
    r"(secret|signature|token|password|key|auth|cvv|pin|card_num|pan|upi_pin|access_token)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Signature Verification Functions
# ---------------------------------------------------------------------------

def verify_razorpay_signature(
    raw_body: bytes,
    signature: str,
    secret: Optional[str] = None,
) -> bool:
    """
    Verifies Razorpay HMAC-SHA256 signature from X-Razorpay-Signature header.
    """
    if not signature:
        return False
    sec = (secret or RAZORPAY_WEBHOOK_SECRET).encode("utf-8")
    computed = hmac.new(sec, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed.lower(), signature.strip().lower())


def verify_paytm_signature(
    raw_body: bytes,
    signature: str,
    secret: Optional[str] = None,
) -> bool:
    """
    Verifies Paytm checksum signature from X-Paytm-Signature header or head.
    Supports HMAC-SHA256 checksum verification.
    """
    if not signature:
        return False
    sec = (secret or PAYTM_MERCHANT_KEY).encode("utf-8")
    computed = hmac.new(sec, raw_body, hashlib.sha256).hexdigest()
    # Support both direct hmac and sha256(raw + "|" + secret)
    if hmac.compare_digest(computed.lower(), signature.strip().lower()):
        return True
    alt_hash = hashlib.sha256(raw_body + b"|" + sec).hexdigest()
    return hmac.compare_digest(alt_hash.lower(), signature.strip().lower())


def verify_phonepe_signature(
    base64_payload: str,
    x_verify: str,
    salt_key: Optional[str] = None,
    salt_index: Optional[str] = None,
    endpoint: str = "/v3/transaction/callback",
) -> bool:
    """
    Verifies PhonePe signature from X-VERIFY header:
    SHA256(base64_payload + endpoint + salt_key) + "###" + salt_index
    """
    if not x_verify or "###" not in x_verify:
        return False

    sig_hash, sig_index = x_verify.split("###", 1)
    target_salt = salt_key or PHONEPE_SALT_KEY
    target_index = salt_index or PHONEPE_SALT_INDEX

    if sig_index.strip() != target_index.strip():
        return False

    to_hash = f"{base64_payload}{endpoint}{target_salt}".encode("utf-8")
    computed_hash = hashlib.sha256(to_hash).hexdigest()
    return hmac.compare_digest(computed_hash.lower(), sig_hash.strip().lower())


def verify_whatsapp_signature(
    raw_body: bytes,
    signature_header: str,
    app_secret: Optional[str] = None,
) -> bool:
    """
    Verifies Meta WhatsApp Business HMAC-SHA256 signature (X-Hub-Signature-256).
    Header format: 'sha256=<hex_digest>'
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    provided_sig = signature_header[7:].strip()
    sec = (app_secret or WHATSAPP_APP_SECRET).encode("utf-8")
    computed = hmac.new(sec, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed.lower(), provided_sig.lower())


def verify_telegram_secret(
    secret_token_header: str,
    expected_secret: Optional[str] = None,
) -> bool:
    """
    Verifies Telegram Bot Webhook secret token from X-Telegram-Bot-Api-Secret-Token.
    """
    if not secret_token_header:
        return False
    expected = (expected_secret or TELEGRAM_WEBHOOK_SECRET).strip()
    return hmac.compare_digest(secret_token_header.strip(), expected)


# ---------------------------------------------------------------------------
# Timestamp Freshness & Replay Attack Defense
# ---------------------------------------------------------------------------

def validate_event_timestamp(
    event_timestamp: datetime,
    max_future_minutes: float = 5.0,
    max_age_hours: float = 24.0,
    ref_time: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """
    Validates event timestamp to prevent replay attacks and future-dating:
    - Rejects events with future timestamps (> max_future_minutes ahead of server clock)
    - Rejects events older than max_age_hours
    """
    now = ref_time or datetime.now(timezone.utc)
    if event_timestamp.tzinfo is None:
        event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Future check
    if event_timestamp > now + timedelta(minutes=max_future_minutes):
        diff_sec = (event_timestamp - now).total_seconds()
        return False, f"Event timestamp is {diff_sec:.0f}s in the future (exceeds {max_future_minutes}m tolerance)"

    # Age check
    if now - event_timestamp > timedelta(hours=max_age_hours):
        diff_hours = (now - event_timestamp).total_seconds() / 3600.0
        return False, f"Event timestamp is stale ({diff_hours:.1f} hours old, max allowed: {max_age_hours}h)"

    return True, "Timestamp valid"


# ---------------------------------------------------------------------------
# Sensitive Data Sanitization (Masking)
# ---------------------------------------------------------------------------

def mask_sensitive_payload(data: Any) -> Any:
    """
    Recursively masks sensitive payment credentials, tokens, secrets,
    and CVVs before writing to logs, Dead-Letter Queues, or diagnostics.
    """
    if isinstance(data, dict):
        sanitized: Dict[str, Any] = {}
        for k, v in data.items():
            if SENSITIVE_KEY_PATTERNS.search(str(k)):
                sanitized[k] = "***MASKED***"
            else:
                sanitized[k] = mask_sensitive_payload(v)
        return sanitized
    elif isinstance(data, list):
        return [mask_sensitive_payload(item) for item in data]
    elif isinstance(data, (str, bytes)):
        # Check if raw string resembles a 16-digit card number or JWT
        val_str = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data
        if re.match(r"^\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}$", val_str.strip()):
            return "****-****-****-" + val_str.strip()[-4:]
        if val_str.startswith("eyJh") and len(val_str) > 50:
            return "jwt.***MASKED***"
        return data
    return data
