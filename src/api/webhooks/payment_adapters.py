"""
Payment Webhook Adapters for Razorpay, Paytm, and PhonePe.
Transforms incoming raw provider payloads into canonical TransactionRecords
compatible with Member 1's TransactionIngestionPipeline.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

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
from api.webhooks.security import validate_event_timestamp

logger = logging.getLogger(__name__)


class WebhookParseError(Exception):
    """Raised when an incoming payment payload cannot be parsed or lacks mandatory fields."""
    pass


# ---------------------------------------------------------------------------
# Razorpay Adapter
# ---------------------------------------------------------------------------

def parse_razorpay_event(payload: Dict[str, Any]) -> TransactionRecord:
    """
    Parses a Razorpay webhook event into a canonical TransactionRecord.
    Supported events: payment.captured, payment.failed, order.paid, refund.processed
    """
    event_type = payload.get("event")
    if not event_type:
        raise WebhookParseError("Missing 'event' field in Razorpay webhook payload")

    entity_data: Dict[str, Any] = {}
    p_wrapper = payload.get("payload", {})
    if "payment" in p_wrapper and "entity" in p_wrapper["payment"]:
        entity_data = p_wrapper["payment"]["entity"]
    elif "order" in p_wrapper and "entity" in p_wrapper["order"]:
        entity_data = p_wrapper["order"]["entity"]
    elif "refund" in p_wrapper and "entity" in p_wrapper["refund"]:
        entity_data = p_wrapper["refund"]["entity"]
    else:
        # Direct entity payload
        entity_data = payload.get("entity", payload)

    tx_id = entity_data.get("id") or entity_data.get("payment_id")
    if not tx_id:
        raise WebhookParseError("Razorpay payload missing payment transaction ID ('id')")

    # Extract vendor ID from notes or metadata
    notes = entity_data.get("notes") or {}
    vendor_id = (
        notes.get("vendor_id")
        or notes.get("merchant_vendor_id")
        or entity_data.get("vendor_id")
    )
    if not vendor_id:
        raise WebhookParseError(f"Razorpay transaction {tx_id} lacks mandatory 'vendor_id' in notes")

    # Amount in Razorpay is in paise (1 INR = 100 paise)
    amount_paise = entity_data.get("amount", 0)
    amount_inr = round(float(amount_paise) / 100.0, 2)

    # Event timestamp
    created_at_epoch = entity_data.get("created_at") or payload.get("created_at")
    if created_at_epoch:
        event_time = datetime.fromtimestamp(created_at_epoch, tz=timezone.utc)
    else:
        event_time = datetime.now(timezone.utc)

    # Validate timestamp freshness
    valid_ts, ts_msg = validate_event_timestamp(event_time)
    if not valid_ts:
        raise WebhookParseError(f"Timestamp validation failed: {ts_msg}")

    # Map payment method
    method_raw = (entity_data.get("method") or "upi").lower()
    if "upi" in method_raw:
        payment_method = PaymentMethod.UPI
    elif "card" in method_raw:
        payment_method = PaymentMethod.CARD
    elif "wallet" in method_raw:
        payment_method = PaymentMethod.WALLET
    else:
        payment_method = PaymentMethod.OTHER

    # Map status
    status_raw = (entity_data.get("status") or "").lower()
    if event_type in ("payment.captured", "order.paid") or status_raw == "captured":
        tx_status = TransactionStatus.COMPLETED
    elif event_type == "refund.processed" or status_raw == "refunded":
        tx_status = TransactionStatus.REFUNDED
    elif event_type == "payment.failed" or status_raw == "failed":
        tx_status = TransactionStatus.CANCELLED
    else:
        tx_status = TransactionStatus.COMPLETED

    # Extract location if vendor mobile POS injected lat/lng into notes
    location = None
    lat = notes.get("location_lat") or notes.get("latitude")
    lng = notes.get("location_lng") or notes.get("longitude")
    if lat is not None and lng is not None:
        try:
            location = TransactionLocation(
                latitude=float(lat),
                longitude=float(lng),
            )
        except Exception:
            location = None

    # Line items if provided
    items: list[TransactionItem] = []
    raw_items = notes.get("items") or entity_data.get("items")
    if isinstance(raw_items, list):
        for itm in raw_items:
            if isinstance(itm, dict) and "item_name" in itm:
                items.append(TransactionItem(
                    item_name=str(itm["item_name"]),
                    quantity=int(itm.get("quantity", 1)),
                    unit_price_inr=float(itm.get("unit_price_inr", amount_inr)),
                ))

    return TransactionRecord(
        transaction_id=str(tx_id),
        vendor_id=str(vendor_id),
        customer_id=entity_data.get("contact") or entity_data.get("email"),
        event_timestamp=event_time,
        amount_inr=amount_inr,
        payment_method=payment_method,
        status=tx_status,
        items=items,
        location=location,
        source="real:webhook:razorpay",
        metadata={
            "provider": "razorpay",
            "event_type": event_type,
            "currency": entity_data.get("currency", "INR"),
            "fee_inr": round(float(entity_data.get("fee", 0)) / 100.0, 2),
        },
    )


# ---------------------------------------------------------------------------
# Paytm Adapter
# ---------------------------------------------------------------------------

def parse_paytm_event(payload: Dict[str, Any]) -> TransactionRecord:
    """
    Parses a Paytm webhook callback into a canonical TransactionRecord.
    Supports body structure conforming to Paytm standard callback specification.
    """
    body = payload.get("body", payload)
    head = payload.get("head", {})

    tx_id = body.get("txnId") or body.get("TXNID")
    if not tx_id:
        raise WebhookParseError("Paytm callback payload missing transaction ID ('txnId' / 'TXNID')")

    order_id = body.get("orderId") or body.get("ORDERID") or ""
    vendor_id = (
        body.get("merchantVendorId")
        or body.get("vendor_id")
        or (body.get("extraParamsMap", {}).get("vendor_id"))
    )
    if not vendor_id:
        # Fallback: parse vendor ID from orderId if prefixed (e.g., 'ORD_VND-001_1029')
        parts = order_id.split("_")
        if len(parts) >= 2 and any(k in parts[1].lower() for k in ("v", "vnd", "vendor")):
            vendor_id = parts[1]
        else:
            raise WebhookParseError(f"Paytm transaction {tx_id} lacks mandatory 'merchantVendorId'")

    amount_raw = body.get("txnAmount") or body.get("TXNAMOUNT") or 0.0
    try:
        amount_inr = round(float(amount_raw), 2)
    except (ValueError, TypeError):
        raise WebhookParseError(f"Invalid Paytm transaction amount: {amount_raw}")

    # Event timestamp
    date_str = body.get("txnDate") or body.get("TXNDATE")
    if date_str:
        try:
            event_time = datetime.fromisoformat(date_str)
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
        except Exception:
            event_time = datetime.now(timezone.utc)
    else:
        event_time = datetime.now(timezone.utc)

    valid_ts, ts_msg = validate_event_timestamp(event_time)
    if not valid_ts:
        raise WebhookParseError(f"Paytm timestamp validation failed: {ts_msg}")

    # Status
    result_status = (
        body.get("resultInfo", {}).get("resultStatus")
        or body.get("STATUS")
        or ""
    ).upper()

    if result_status in ("TXN_SUCCESS", "SUCCESS"):
        tx_status = TransactionStatus.COMPLETED
    elif result_status in ("TXN_FAILURE", "FAILURE"):
        tx_status = TransactionStatus.CANCELLED
    elif "REFUND" in result_status:
        tx_status = TransactionStatus.REFUNDED
    else:
        tx_status = TransactionStatus.COMPLETED

    # Payment Mode
    mode = (body.get("paymentMode") or body.get("PAYMENTMODE") or "UPI").upper()
    if "UPI" in mode:
        pm = PaymentMethod.UPI
    elif "CARD" in mode or "CC" in mode or "DC" in mode:
        pm = PaymentMethod.CARD
    elif "WALLET" in mode or "PPI" in mode:
        pm = PaymentMethod.WALLET
    else:
        pm = PaymentMethod.OTHER

    # Location
    location = None
    extra = body.get("extraParamsMap") or {}
    lat = extra.get("latitude") or body.get("latitude")
    lng = extra.get("longitude") or body.get("longitude")
    if lat is not None and lng is not None:
        try:
            location = TransactionLocation(latitude=float(lat), longitude=float(lng))
        except Exception:
            location = None

    return TransactionRecord(
        transaction_id=str(tx_id),
        vendor_id=str(vendor_id),
        customer_id=body.get("bankTxnId"),
        event_timestamp=event_time,
        amount_inr=amount_inr,
        payment_method=pm,
        status=tx_status,
        items=[],
        location=location,
        source="real:webhook:paytm",
        metadata={
            "provider": "paytm",
            "order_id": order_id,
            "bank_name": body.get("bankName"),
            "gateway_name": body.get("gatewayName"),
        },
    )


# ---------------------------------------------------------------------------
# PhonePe Adapter
# ---------------------------------------------------------------------------

def parse_phonepe_event(payload: Dict[str, Any]) -> TransactionRecord:
    """
    Parses a PhonePe server-to-server webhook callback into a TransactionRecord.
    Handles standard PhonePe base64-encoded response payload envelope.
    """
    raw_response = payload.get("response")
    data_dict: Dict[str, Any] = {}

    if raw_response and isinstance(raw_response, str):
        try:
            decoded_bytes = base64.b64decode(raw_response)
            data_dict = json.loads(decoded_bytes.decode("utf-8"))
        except Exception as e:
            raise WebhookParseError(f"Failed to decode PhonePe base64 payload: {e}")
    else:
        data_dict = payload

    inner_data = data_dict.get("data") or data_dict
    tx_id = (
        inner_data.get("transactionId")
        or inner_data.get("merchantTransactionId")
    )
    if not tx_id:
        raise WebhookParseError("PhonePe callback missing transactionId")

    vendor_id = (
        inner_data.get("merchantVendorId")
        or inner_data.get("vendor_id")
        or (inner_data.get("metadata", {}).get("vendor_id"))
    )
    if not vendor_id:
        merchant_tx = inner_data.get("merchantTransactionId", "")
        parts = merchant_tx.split("_")
        if len(parts) >= 2 and any(k in parts[1].lower() for k in ("v", "vnd", "vendor")):
            vendor_id = parts[1]
        else:
            raise WebhookParseError(f"PhonePe transaction {tx_id} missing mandatory 'merchantVendorId'")

    # PhonePe amount is in paise (100 paise = 1 INR)
    amount_paise = inner_data.get("amount", 0)
    amount_inr = round(float(amount_paise) / 100.0, 2)

    # State & Status
    state = (inner_data.get("state") or data_dict.get("code") or "COMPLETED").upper()
    if state in ("COMPLETED", "PAYMENT_SUCCESS"):
        tx_status = TransactionStatus.COMPLETED
    elif state in ("FAILED", "PAYMENT_ERROR", "PAYMENT_DECLINED"):
        tx_status = TransactionStatus.CANCELLED
    elif "REFUND" in state:
        tx_status = TransactionStatus.REFUNDED
    else:
        tx_status = TransactionStatus.COMPLETED

    event_time = datetime.now(timezone.utc)
    payment_inst = inner_data.get("paymentInstrument") or {}
    inst_type = (payment_inst.get("type") or "UPI").upper()
    if "UPI" in inst_type:
        pm = PaymentMethod.UPI
    elif "CARD" in inst_type:
        pm = PaymentMethod.CARD
    elif "WALLET" in inst_type:
        pm = PaymentMethod.WALLET
    else:
        pm = PaymentMethod.OTHER

    # Location
    location = None
    loc_data = inner_data.get("location") or {}
    lat = loc_data.get("latitude") or inner_data.get("latitude")
    lng = loc_data.get("longitude") or inner_data.get("longitude")
    if lat is not None and lng is not None:
        try:
            location = TransactionLocation(latitude=float(lat), longitude=float(lng))
        except Exception:
            location = None

    return TransactionRecord(
        transaction_id=str(tx_id),
        vendor_id=str(vendor_id),
        customer_id=payment_inst.get("utr"),
        event_timestamp=event_time,
        amount_inr=amount_inr,
        payment_method=pm,
        status=tx_status,
        items=[],
        location=location,
        source="real:webhook:phonepe",
        metadata={
            "provider": "phonepe",
            "merchant_id": inner_data.get("merchantId"),
            "merchant_tx_id": inner_data.get("merchantTransactionId"),
            "utr": payment_inst.get("utr"),
        },
    )
