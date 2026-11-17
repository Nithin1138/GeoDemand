# Payment Webhooks, Messaging Channel Ingestion & Dead-Letter Recovery Report

**Author / Maintainer:** Pujitha & Team  
**Date:** November 16, 2026  
**Status:** Implemented, Tested, & Production-Ready  
**Test Suite:** `tests/test_payment_webhooks.py` (12/12 passed in 2.60s; total suite: 121/121 passed)

---

## 1. Executive Summary

To enable real-time demand updates and continuous training telemetry for the GeoDemand AI platform, street vendor Point-of-Sale (POS) transactions and field movements must be captured instantaneously. However, urban Indian micro-vendors rely on a heterogeneous mix of payment gateways (Razorpay, Paytm, PhonePe) and conversational reporting tools (WhatsApp Business, Telegram).

To bridge this operational gap, we implemented the **Webhook Ingestion Subsystem (`src/api/webhooks/`)**:
1. **Cryptographic Webhook Adapters**: Real-time webhook ingestion for **Razorpay**, **Paytm**, and **PhonePe**, verifying HMAC-SHA256, checksums, and salt-based signatures.
2. **Conversational Messaging Adapters**: Ingestion of vendor sales text reports and live GPS location pins via **WhatsApp Business API** and **Telegram Bot API**.
3. **Idempotency & Replay Protection**: Thread-safe duplicate event detection ensuring network retries never double-count vendor sales.
4. **Dead-Letter Queue (DLQ) & Failure Recovery**: Isolated, credential-sanitized storage for malformed or unauthenticated events with automated/manual replay capabilities.
5. **Seamless Pipeline Forwarding**: Transformation of raw payloads into canonical `TransactionRecord`s for Member 1's `TransactionIngestionPipeline` and `FleetManager` spatial tracking.

---

## 2. Payment Gateway Adapters & Signature Verification

### 2.1 Razorpay Webhook Adapter (`api/webhooks/payment_adapters.py`)
- **Supported Events**: `payment.captured`, `order.paid`, `refund.processed`, `payment.failed`.
- **Signature Verification**: HMAC-SHA256 of raw request body using `RAZORPAY_WEBHOOK_SECRET` compared against `X-Razorpay-Signature` via constant-time digest comparison (`hmac.compare_digest`).
- **Paise to INR Normalization**: Razorpay sends amounts in integer paise ($100\text{ paise} = ₹1.00$). Normalized to float INR.
- **Vendor Identity Extraction**: Sourced from `payload.payment.entity.notes.vendor_id` or `notes.merchant_vendor_id`.
- **Lifecycle Mapping**: `captured` $\to$ `COMPLETED`, `refunded` $\to$ `REFUNDED`, `failed` $\to$ `CANCELLED`.

### 2.2 Paytm Webhook Adapter
- **Supported Events**: `TXN_SUCCESS`, `TXN_FAILURE`, `REFUND_SUCCESS`.
- **Signature Verification**: SHA256 checksum comparison against `X-Paytm-Signature` or `head.signature` using `PAYTM_MERCHANT_KEY`.
- **Vendor Identity Extraction**: Sourced from `merchantVendorId` or order ID prefix (`ORD_<vendor_id>_<timestamp>`).
- **Location Extraction**: Reads GPS coordinates from `extraParamsMap.latitude`/`longitude`.

### 2.3 PhonePe Webhook Adapter
- **Supported Events**: `PAYMENT_SUCCESS`, `PAYMENT_ERROR`, `PAYMENT_DECLINED`, `REFUND`.
- **Signature Verification**: Validates `X-VERIFY` header formatted as:
  $$\text{SHA256}(\text{Base64Payload} + \text{"/v3/transaction/callback"} + \text{SaltKey}) + \text{"###"} + \text{SaltIndex}$$
- **Envelope Unpacking**: Decodes Base64 JSON payload and normalizes paise to INR.

---

## 3. Messaging Channel Ingestion: WhatsApp & Telegram

Mobile vendors often lack dedicated POS terminals and instead log sales or pin locations through messaging apps:

### 3.1 WhatsApp Business API Adapter (`api/webhooks/messaging_adapters.py`)
- **Webhook Verification**: Implements Meta challenge verification:
  `GET /v1/webhooks/whatsapp?hub.mode=subscribe&hub.challenge=...&hub.verify_token=...`
- **Signature Validation**: HMAC-SHA256 signature verification via `X-Hub-Signature-256`.
- **Vendor Authentication**: Maps sender mobile number (`from`: `+919876543210`) against verified vendor identities in `VendorChannelRegistry`.
- **Sales Report Parsing**: Heuristically extracts monetary values and purchased line items from free-form text:
  - e.g. `"SALE 450 INR 3 items"` $\to$ ₹450.00 cash transaction.
  - e.g. `"SOLD 15 dosa 450 INR"` $\to$ ₹450.00 with line item: 15 Dosa @ ₹30/unit.
- **GPS Location Pins**: Parses WhatsApp location messages (`latitude`, `longitude`, `name`), dynamically updating vendor coordinates in `TransactionIngestionPipeline.register_vendor_location()` and `FleetManager`.

### 3.2 Telegram Bot Adapter
- **Secret Token Verification**: Authenticates `X-Telegram-Bot-Api-Secret-Token` matching `TELEGRAM_WEBHOOK_SECRET`.
- **Vendor Authentication**: Maps Telegram user ID / `@username` to registered vendor identity.
- **Command & Location Parsing**:
  - Accepts `/sale <amount> <optional_item>` sales reports.
  - Accepts native Telegram GPS location pins.

---

## 4. Idempotency & Dead-Letter Queue (DLQ)

### 4.1 Idempotency Store
Payment gateways frequently retry webhook deliveries upon slight network delays.
- Maintains thread-safe processed event keys: `f"{provider}:{transaction_id}"`.
- Subsequent deliveries of an already ingested transaction return HTTP 200 with `status="DUPLICATE_IGNORED"`, completely preventing double-counting of revenue or duplicate pipeline entries.

### 4.2 Timestamp Freshness Validation
Prevents clock manipulation and replay attacks:
- **Future Tolerance**: Rejects events stamped $> 5\text{ minutes}$ in the future.
- **Staleness Tolerance**: Rejects events older than $24\text{ hours}$.

### 4.3 Sensitive Credential Sanitization
All logged or dead-lettered payloads are recursively scrubbed via `mask_sensitive_payload()`:
- Replaces keys matching `secret`, `signature`, `token`, `password`, `key`, `auth`, `cvv`, `pin`, `card_num` with `"***MASKED***"`.
- Never persists or exposes payment credentials or customer PII.

### 4.4 Dead-Letter Queue & Replay Recovery
Events that fail due to missing vendor mapping, corrupt JSON, or temporary downstream outages are automatically stored in the DLQ:
- Stores sanitized payload, error rationale, failure timestamp, and retry counter.
- **Inspection**: `GET /v1/webhooks/dead-letter`
- **Replay / Recovery**: `POST /v1/webhooks/dead-letter/retry` allows re-ingesting events once vendor channels are registered or issues are resolved.

---

## 5. API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/webhooks/payments/razorpay` | POST | Razorpay payment webhook callback |
| `/v1/webhooks/payments/paytm` | POST | Paytm payment webhook callback |
| `/v1/webhooks/payments/phonepe` | POST | PhonePe transaction callback |
| `/v1/webhooks/whatsapp` | GET | WhatsApp Meta verification challenge |
| `/v1/webhooks/whatsapp` | POST | Inbound WhatsApp sales reports and location pins |
| `/v1/webhooks/telegram` | POST | Telegram bot sales commands and location pins |
| `/v1/webhooks/dead-letter` | GET | List failed / unparseable events in DLQ |
| `/v1/webhooks/dead-letter/retry` | POST | Replay and recover an event from DLQ |
| `/v1/webhooks/register-vendor-channel`| POST | Link WhatsApp phone or Telegram ID to vendor |

---

## 6. Test Suite & Validation Results

The webhook ingestion system is thoroughly verified across 12 unit and integration tests in `tests/test_payment_webhooks.py`:
1. `test_razorpay_signature_and_tamper_rejection`: Validates HMAC-SHA256 signature and rejects tampered bodies.
2. `test_razorpay_event_ingestion_forwarding`: Verifies canonical `TransactionRecord` generation and pipeline acceptance.
3. `test_paytm_signature_and_ingestion`: Validates Paytm checksum verification, order ID parsing, and pipeline acceptance.
4. `test_phonepe_signature_and_base64_ingestion`: Verifies Base64 decoding, X-VERIFY calculation, and pipeline forwarding.
5. `test_webhook_idempotency_duplicate_protection`: Proves duplicate deliveries return `DUPLICATE_IGNORED` without pipeline re-ingestion.
6. `test_timestamp_freshness_validation`: Confirms rejection of future (>5m) and stale (>24h) events.
7. `test_sensitive_credential_masking`: Verifies scrubbing of keys, tokens, CVVs, and card numbers.
8. `test_dead_letter_queue_and_retry_recovery`: Verifies routing bad events to DLQ, credential masking, and successful recovery replay.
9. `test_whatsapp_location_pin_and_sales_report`: Validates location pin updates and text sales parsing.
10. `test_telegram_sales_command_and_location`: Validates Telegram `/sale` commands and location pins.
11. `test_whatsapp_verification_challenge_endpoint`: Validates Meta `hub.challenge` handshake.
12. `test_payment_webhook_fastapi_endpoints`: End-to-end execution of FastAPI route handlers and channel registration.

**Execution Results:**
- Webhook Ingestion suite: **12 passed in 2.60s**
- Full application test suite: **121 passed in 11.18s**
