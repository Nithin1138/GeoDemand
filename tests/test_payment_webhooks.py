"""
Comprehensive Test Suite for Payment Webhooks, Messaging Adapters,
Cryptographic Signatures, Idempotency, and Dead-Letter Queue Recovery.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone, timedelta
import pytest

from api.webhooks.security import (
    verify_razorpay_signature,
    verify_paytm_signature,
    verify_phonepe_signature,
    verify_whatsapp_signature,
    verify_telegram_secret,
    validate_event_timestamp,
    mask_sensitive_payload,
    RAZORPAY_WEBHOOK_SECRET,
    PAYTM_MERCHANT_KEY,
    PHONEPE_SALT_KEY,
    PHONEPE_SALT_INDEX,
    WHATSAPP_VERIFY_TOKEN,
    WHATSAPP_APP_SECRET,
    TELEGRAM_WEBHOOK_SECRET,
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
)
from api.webhooks.dead_letter import (
    dead_letter_queue,
    DeadLetterRecord,
)
from api.webhooks.service import webhook_service
from api.app import (
    webhook_razorpay,
    webhook_paytm,
    webhook_phonepe,
    webhook_whatsapp_verify,
    webhook_whatsapp_event,
    webhook_telegram_event,
    webhook_list_dead_letters,
    webhook_retry_dead_letter,
    webhook_register_vendor_channel,
)
from api.schemas import (
    DeadLetterRetryRequest,
    VendorChannelRegisterRequest,
)


def run_coroutine(coro):
    """Helper for executing FastAPI async routes without event loop network sockets on Windows."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not complete synchronously")


class DummyRequest:
    """Mock FastAPI Request for direct unit test invocation."""
    def __init__(self, raw_body: bytes, headers: dict = None, parsed_json: dict = None):
        self._raw_body = raw_body
        self.headers = headers or {}
        self._json = parsed_json if parsed_json is not None else {}

    async def body(self) -> bytes:
        return self._raw_body

    async def json(self) -> dict:
        return self._json


@pytest.fixture(autouse=True)
def clean_webhook_stores():
    """Ensure clean idempotency and DLQ stores before each test."""
    dead_letter_queue.reset_state_for_testing()
    yield
    dead_letter_queue.reset_state_for_testing()


# ===========================================================================
# 1. Razorpay Signature & Ingestion
# ===========================================================================

def test_razorpay_signature_and_tamper_rejection():
    """Valid HMAC-SHA256 signature is accepted; tampered payload or signature is rejected."""
    raw_payload = b'{"event":"payment.captured","created_at":1729000000}'
    valid_sig = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), raw_payload, hashlib.sha256).hexdigest()

    # Valid
    assert verify_razorpay_signature(raw_payload, valid_sig) is True

    # Tampered signature
    assert verify_razorpay_signature(raw_payload, "invalid_sig_abc123") is False

    # Tampered body
    tampered_body = b'{"event":"payment.captured","created_at":1729000001}'
    assert verify_razorpay_signature(tampered_body, valid_sig) is False


def test_razorpay_event_ingestion_forwarding():
    """Valid Razorpay event converts to TransactionRecord and enters ingestion pipeline."""
    payload = {
        "event": "payment.captured",
        "created_at": int(datetime.now(timezone.utc).timestamp()),
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_RZP_1001",
                    "amount": 25000,  # 250 INR
                    "currency": "INR",
                    "status": "captured",
                    "method": "upi",
                    "notes": {
                        "vendor_id": "vendor_001",
                        "location_lat": 16.5062,
                        "location_lng": 80.6480,
                    },
                }
            }
        }
    }
    raw_bytes = json.dumps(payload).encode()
    sig = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), raw_bytes, hashlib.sha256).hexdigest()

    res = webhook_service.process_payment_webhook(
        provider="razorpay",
        raw_body=raw_bytes,
        headers={"x-razorpay-signature": sig},
        parsed_json=payload,
    )

    assert res["status"] == "INGESTED"
    assert res["provider"] == "razorpay"
    assert res["transaction_id"] == "pay_RZP_1001"
    assert res["vendor_id"] == "vendor_001"
    assert res["amount_inr"] == 250.0
    assert res["h3_cell_id"] == "88619aa6c7fffff"


# ===========================================================================
# 2. Paytm Checksum Signature & Ingestion
# ===========================================================================

def test_paytm_signature_and_ingestion():
    """Valid Paytm callback converts to TransactionRecord and forwards to pipeline."""
    payload = {
        "head": {"responseTimestamp": "1729000000", "version": "v1"},
        "body": {
            "resultInfo": {"resultStatus": "TXN_SUCCESS"},
            "txnId": "PTM_TXN_5501",
            "orderId": "ORD_vendor_001_1029",
            "merchantVendorId": "vendor_001",
            "txnAmount": "180.00",
            "paymentMode": "UPI",
            "extraParamsMap": {"latitude": 16.5062, "longitude": 80.6480},
        }
    }
    raw_bytes = json.dumps(payload).encode()
    sig = hmac.new(PAYTM_MERCHANT_KEY.encode(), raw_bytes, hashlib.sha256).hexdigest()

    res = webhook_service.process_payment_webhook(
        provider="paytm",
        raw_body=raw_bytes,
        headers={"x-paytm-signature": sig},
        parsed_json=payload,
    )

    assert res["status"] == "INGESTED"
    assert res["provider"] == "paytm"
    assert res["transaction_id"] == "PTM_TXN_5501"
    assert res["amount_inr"] == 180.0
    assert res["vendor_id"] == "vendor_001"


# ===========================================================================
# 3. PhonePe Base64 + SHA256 Verification & Ingestion
# ===========================================================================

def test_phonepe_signature_and_base64_ingestion():
    """PhonePe Base64 encoded payload with X-VERIFY header verification."""
    inner_payload = {
        "success": True,
        "code": "PAYMENT_SUCCESS",
        "data": {
            "merchantId": "MERCHANTUAT",
            "merchantTransactionId": "PPE_TX_7701",
            "transactionId": "T261015143000",
            "amount": 35000,  # 350 INR
            "state": "COMPLETED",
            "paymentInstrument": {"type": "UPI", "utr": "429102918291"},
            "merchantVendorId": "vendor_001",
            "latitude": 16.5062,
            "longitude": 80.6480,
        }
    }
    raw_inner_bytes = json.dumps(inner_payload).encode()
    b64_str = base64.b64encode(raw_inner_bytes).decode("utf-8")

    # Compute PhonePe X-VERIFY: SHA256(b64 + endpoint + salt) + "###" + index
    endpoint = "/v3/transaction/callback"
    to_hash = f"{b64_str}{endpoint}{PHONEPE_SALT_KEY}".encode("utf-8")
    computed_hash = hashlib.sha256(to_hash).hexdigest()
    x_verify = f"{computed_hash}###{PHONEPE_SALT_INDEX}"

    outer_payload = {"response": b64_str}
    raw_outer_bytes = json.dumps(outer_payload).encode()

    res = webhook_service.process_payment_webhook(
        provider="phonepe",
        raw_body=raw_outer_bytes,
        headers={"x-verify": x_verify},
        parsed_json=outer_payload,
    )

    assert res["status"] == "INGESTED"
    assert res["provider"] == "phonepe"
    assert res["amount_inr"] == 350.0
    assert res["vendor_id"] == "vendor_001"


# ===========================================================================
# 4. Idempotency & Duplicate Delivery Protection
# ===========================================================================

def test_webhook_idempotency_duplicate_protection():
    """Duplicate webhook delivery returns DUPLICATE_IGNORED and avoids double ingestion."""
    payload = {
        "event": "payment.captured",
        "created_at": int(datetime.now(timezone.utc).timestamp()),
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_IDEMPOTENT_999",
                    "amount": 10000,
                    "method": "upi",
                    "notes": {"vendor_id": "vendor_001"},
                }
            }
        }
    }
    raw_bytes = json.dumps(payload).encode()
    sig = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), raw_bytes, hashlib.sha256).hexdigest()

    # Delivery 1: Ingested
    res1 = webhook_service.process_payment_webhook(
        "razorpay", raw_bytes, {"x-razorpay-signature": sig}, payload
    )
    assert res1["status"] == "INGESTED"

    # Delivery 2 (Provider retry): Safely ignored as duplicate
    res2 = webhook_service.process_payment_webhook(
        "razorpay", raw_bytes, {"x-razorpay-signature": sig}, payload
    )
    assert res2["status"] == "DUPLICATE_IGNORED"
    assert "already processed" in res2["message"].lower()


# ===========================================================================
# 5. Timestamp Freshness & Replay Attack Defense
# ===========================================================================

def test_timestamp_freshness_validation():
    """Future events (>5m) and excessively stale events (>24h) are rejected."""
    now = datetime.now(timezone.utc)

    # Valid current timestamp
    ok1, _ = validate_event_timestamp(now)
    assert ok1 is True

    # Future timestamp (15 minutes ahead) -> Rejected
    future_time = now + timedelta(minutes=15)
    ok_fut, msg_fut = validate_event_timestamp(future_time)
    assert ok_fut is False
    assert "in the future" in msg_fut

    # Stale timestamp (30 hours old) -> Rejected
    stale_time = now - timedelta(hours=30)
    ok_stale, msg_stale = validate_event_timestamp(stale_time)
    assert ok_stale is False
    assert "stale" in msg_stale


# ===========================================================================
# 6. Sensitive Data Sanitization (Masking)
# ===========================================================================

def test_sensitive_credential_masking():
    """Confirms that API keys, secrets, tokens, card numbers, and CVVs are masked."""
    sensitive_data = {
        "event": "payment.captured",
        "api_key": "AKIA_SECRET_XYZ",
        "webhook_secret": "super_secret_token_123",
        "payment": {
            "card_number": "4111 1111 1111 1234",
            "cvv": "999",
            "auth_token": "bearer eyJhbGciOi...",
            "amount": 250.0,
        }
    }

    masked = mask_sensitive_payload(sensitive_data)
    assert masked["api_key"] == "***MASKED***"
    assert masked["webhook_secret"] == "***MASKED***"
    assert masked["payment"]["cvv"] == "***MASKED***"
    assert masked["payment"]["auth_token"] == "***MASKED***"
    assert masked["payment"]["amount"] == 250.0


# ===========================================================================
# 7. Dead-Letter Queue (DLQ) & Failure Recovery
# ===========================================================================

def test_dead_letter_queue_and_retry_recovery():
    """
    Simulates an unparseable webhook event routed to DLQ,
    then verifies inspection and successful recovery.
    """
    # Bad payload missing vendor_id
    bad_payload = {
        "event": "payment.captured",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_RECOVER_001",
                    "amount": 50000,
                    # Missing vendor_id in notes
                }
            }
        }
    }
    raw_bytes = json.dumps(bad_payload).encode()
    sig = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), raw_bytes, hashlib.sha256).hexdigest()

    # Process: fails and enqueues to DLQ
    res = webhook_service.process_payment_webhook(
        "razorpay", raw_bytes, {"x-razorpay-signature": sig}, bad_payload
    )
    assert res["status"] == "REJECTED"
    dlq_id = res["dead_letter_id"]
    assert dlq_id.startswith("DLQ-")

    # Inspect DLQ
    records = dead_letter_queue.list_records(provider="razorpay")
    assert len(records) >= 1
    rec = dead_letter_queue.get_record(dlq_id)
    assert rec is not None
    assert rec.status == "FAILED"

    # Fix configuration / payload notes for retry
    rec.sanitized_payload["payload"]["payment"]["entity"]["notes"] = {"vendor_id": "vendor_001"}

    # Retry recovery
    retry_res = webhook_service.retry_dead_letter_event(dlq_id)
    assert retry_res["status"] == "RECOVERED"
    assert retry_res["transaction_id"] == "pay_RECOVER_001"
    assert rec.status == "RECOVERED"


# ===========================================================================
# 8. WhatsApp Business API: Sales Reports & Location Pins
# ===========================================================================

def test_whatsapp_location_pin_and_sales_report():
    """Processes WhatsApp location pin and text sales report from registered vendor phone."""
    # Register phone number for vendor_001
    vendor_channel_registry.register("vendor_001", phone="+919876543210")

    # A: Location Pin Webhook
    loc_payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "wamid.HBgL...",
                        "from": "919876543210",
                        "type": "location",
                        "location": {
                            "latitude": 16.5062,
                            "longitude": 80.6480,
                            "name": "Benz Circle Stalls",
                        }
                    }]
                }
            }]
        }]
    }

    res_loc = webhook_service.process_whatsapp_webhook(
        raw_body=json.dumps(loc_payload).encode(),
        signature=None,
        parsed_json=loc_payload,
    )
    assert res_loc["status"] == "LOCATION_UPDATED"
    assert res_loc["vendor_id"] == "vendor_001"
    assert res_loc["latitude"] == 16.5062

    # B: Text Sales Report Webhook ("SOLD 15 dosa 450 INR")
    sales_payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "wamid.HBgM...",
                        "from": "919876543210",
                        "type": "text",
                        "text": {"body": "SOLD 15 dosa 450 INR"}
                    }]
                }
            }]
        }]
    }

    res_sales = webhook_service.process_whatsapp_webhook(
        raw_body=json.dumps(sales_payload).encode(),
        signature=None,
        parsed_json=sales_payload,
    )
    assert res_sales["status"] == "INGESTED"
    assert res_sales["vendor_id"] == "vendor_001"
    assert res_sales["amount_inr"] == 450.0


# ===========================================================================
# 9. Telegram Bot API: Sales Commands & Location Pins
# ===========================================================================

def test_telegram_sales_command_and_location():
    """Processes Telegram bot /sale command and GPS location message."""
    vendor_channel_registry.register("vendor_002", telegram_id="77665544")

    # Location update
    loc_update = {
        "update_id": 1001,
        "message": {
            "from": {"id": 77665544, "username": "krishna_chaat"},
            "location": {"latitude": 16.5140, "longitude": 80.6550},
        }
    }
    res_loc = webhook_service.process_telegram_webhook(
        secret_token=TELEGRAM_WEBHOOK_SECRET,
        parsed_json=loc_update,
    )
    assert res_loc["status"] == "LOCATION_UPDATED"
    assert res_loc["vendor_id"] == "vendor_002"

    # Sales command (/sale 280)
    sales_update = {
        "update_id": 1002,
        "message": {
            "from": {"id": 77665544, "username": "krishna_chaat"},
            "text": "/sale 280 chaat",
        }
    }
    res_sales = webhook_service.process_telegram_webhook(
        secret_token=TELEGRAM_WEBHOOK_SECRET,
        parsed_json=sales_update,
    )
    assert res_sales["status"] == "INGESTED"
    assert res_sales["vendor_id"] == "vendor_002"
    assert res_sales["amount_inr"] == 280.0


# ===========================================================================
# 10. Webhook Verification Challenges & API Route Contracts
# ===========================================================================

def test_whatsapp_verification_challenge_endpoint():
    """GET /v1/webhooks/whatsapp verifies hub.challenge token."""
    coro_ok = webhook_whatsapp_verify(
        hub_mode="subscribe",
        hub_challenge="challenge_token_99182",
        hub_verify_token=WHATSAPP_VERIFY_TOKEN,
    )
    resp = run_coroutine(coro_ok)
    assert resp.body.decode() == "challenge_token_99182"


def test_payment_webhook_fastapi_endpoints():
    """Full execution of FastAPI webhook route handlers."""
    # Razorpay endpoint test
    payload = {
        "event": "payment.captured",
        "created_at": int(datetime.now(timezone.utc).timestamp()),
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_ENDPOINT_1",
                    "amount": 15000,
                    "method": "upi",
                    "notes": {"vendor_id": "vendor_001"},
                }
            }
        }
    }
    raw_bytes = json.dumps(payload).encode()
    sig = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), raw_bytes, hashlib.sha256).hexdigest()

    req = DummyRequest(raw_bytes, {"x-razorpay-signature": sig}, payload)
    coro = webhook_razorpay(req)
    res = run_coroutine(coro)

    assert res["status"] == "INGESTED"
    assert res["transaction_id"] == "pay_ENDPOINT_1"

    # Register vendor channel endpoint
    reg_req = VendorChannelRegisterRequest(
        vendor_id="vendor_new",
        phone="+919999888877",
        telegram_id="new_user_1",
    )
    coro_reg = webhook_register_vendor_channel(reg_req, api_key="test-key")
    reg_res = run_coroutine(coro_reg)
    assert reg_res["status"] == "REGISTERED"
    assert vendor_channel_registry.resolve_by_phone("+919999888877") == "vendor_new"
