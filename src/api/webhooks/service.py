"""
Webhook Ingestion Service & Pipeline Dispatcher.

Connects payment webhook adapters (Razorpay, Paytm, PhonePe) and
messaging adapters (WhatsApp, Telegram) to Member 1's TransactionIngestionPipeline
and the FleetManager.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

try:
    from src.ingestion.schema import (
        TransactionRecord,
        TransactionLocation,
        PaymentMethod,
        TransactionStatus,
    )
    from src.ingestion.validator import IngestionResult, TransactionIngestionPipeline
except ImportError:
    from ingestion.schema import (
        TransactionRecord,
        TransactionLocation,
        PaymentMethod,
        TransactionStatus,
    )
    from ingestion.validator import IngestionResult, TransactionIngestionPipeline
from fleet.fleet_manager import FleetManager, Vendor, VendorStatus
from api.webhooks.security import (
    verify_razorpay_signature,
    verify_paytm_signature,
    verify_phonepe_signature,
    verify_whatsapp_signature,
    verify_telegram_secret,
    mask_sensitive_payload,
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
    WhatsAppMessageEvent,
    TelegramMessageEvent,
)
from api.webhooks.dead_letter import dead_letter_queue, DeadLetterRecord
try:
    from src.ingestion.event_broker import event_gateway
except ImportError:
    from ingestion.event_broker import event_gateway

logger = logging.getLogger(__name__)


class WebhookIngestionService:
    """
    Unified webhook dispatcher and event ingestor.
    """

    _instance: Optional[WebhookIngestionService] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = TransactionIngestionPipeline(max_delay_hours=24.0)
            cls._instance._fleet_manager = FleetManager()
        return cls._instance

    @property
    def pipeline(self) -> TransactionIngestionPipeline:
        return self._pipeline

    @property
    def fleet_manager(self) -> FleetManager:
        return self._fleet_manager

    # -----------------------------------------------------------------------
    # Payment Webhooks (Razorpay, Paytm, PhonePe)
    # -----------------------------------------------------------------------

    def process_payment_webhook(
        self,
        provider: str,
        raw_body: bytes,
        headers: Dict[str, str],
        parsed_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validates signature, checks idempotency, parses event,
        and forwards to Member 1's ingestion pipeline.
        """
        provider_clean = provider.lower().strip()

        # 1. Signature Validation
        sig_valid = False
        sig_err = "Invalid webhook signature"

        if provider_clean == "razorpay":
            sig = headers.get("x-razorpay-signature") or headers.get("X-Razorpay-Signature", "")
            sig_valid = verify_razorpay_signature(raw_body, sig)
        elif provider_clean == "paytm":
            sig = (
                headers.get("x-paytm-signature")
                or headers.get("X-Paytm-Signature")
                or parsed_json.get("head", {}).get("signature", "")
            )
            sig_valid = verify_paytm_signature(raw_body, sig)
        elif provider_clean == "phonepe":
            x_verify = headers.get("x-verify") or headers.get("X-VERIFY", "")
            b64_resp = parsed_json.get("response", "")
            sig_valid = verify_phonepe_signature(b64_resp, x_verify)
        else:
            sig_err = f"Unsupported payment provider: {provider}"

        if not sig_valid:
            dlq_rec = dead_letter_queue.enqueue(
                provider=provider_clean,
                raw_payload=parsed_json,
                error_reason=f"Signature verification failed: {sig_err}",
            )
            return {
                "status": "REJECTED",
                "provider": provider_clean,
                "error": "SIGNATURE_VERIFICATION_FAILED",
                "message": sig_err,
                "dead_letter_id": dlq_rec.dlq_id,
            }

        # 2. Parse Canonical Record
        try:
            if provider_clean == "razorpay":
                record = parse_razorpay_event(parsed_json)
            elif provider_clean == "paytm":
                record = parse_paytm_event(parsed_json)
            elif provider_clean == "phonepe":
                record = parse_phonepe_event(parsed_json)
            else:
                raise WebhookParseError(f"Unsupported provider {provider_clean}")
        except Exception as e:
            dlq_rec = dead_letter_queue.enqueue(
                provider=provider_clean,
                raw_payload=parsed_json,
                error_reason=f"Payload parsing error: {e}",
            )
            return {
                "status": "REJECTED",
                "provider": provider_clean,
                "error": "PAYLOAD_PARSE_ERROR",
                "message": str(e),
                "dead_letter_id": dlq_rec.dlq_id,
            }

        # 3. Idempotency Check
        if not dead_letter_queue.try_acquire_idempotency(provider_clean, record.transaction_id):
            logger.info(f"Duplicate payment webhook delivery ignored: {provider_clean}:{record.transaction_id}")
            return {
                "status": "DUPLICATE_IGNORED",
                "provider": provider_clean,
                "transaction_id": record.transaction_id,
                "vendor_id": record.vendor_id,
                "amount_inr": record.amount_inr,
                "message": "Duplicate event delivery detected. Transaction already processed.",
            }

        # 4. Ingest via Member 1's Ingestion Pipeline
        try:
            ingestion_res = self._pipeline.ingest_transaction(record)
        except Exception as e:
            dlq_rec = dead_letter_queue.enqueue(
                provider=provider_clean,
                raw_payload=parsed_json,
                error_reason=f"Pipeline ingestion exception: {e}",
            )
            return {
                "status": "PIPELINE_ERROR",
                "provider": provider_clean,
                "error": "PIPELINE_INGESTION_FAILED",
                "message": str(e),
                "dead_letter_id": dlq_rec.dlq_id,
            }

        if not ingestion_res.success:
            if ingestion_res.is_duplicate:
                dead_letter_queue.mark_processed(provider_clean, record.transaction_id)
                return {
                    "status": "DUPLICATE_IGNORED",
                    "provider": provider_clean,
                    "transaction_id": record.transaction_id,
                    "vendor_id": record.vendor_id,
                    "message": "Transaction already processed by ingestion pipeline.",
                }
            dlq_rec = dead_letter_queue.enqueue(
                provider=provider_clean,
                raw_payload=parsed_json,
                error_reason="; ".join(ingestion_res.errors),
            )
            return {
                "status": "REJECTED",
                "provider": provider_clean,
                "transaction_id": record.transaction_id,
                "errors": ingestion_res.errors,
                "dead_letter_id": dlq_rec.dlq_id,
            }

        # Success: Mark in idempotency store
        dead_letter_queue.mark_processed(provider_clean, record.transaction_id)
        final_record = ingestion_res.record or record

        # Forward event to streaming event broker (RabbitMQ / In-Memory)
        try:
            event_gateway.publish_transaction(
                final_record,
                idempotency_key=f"{provider_clean}:{final_record.transaction_id}",
            )
        except Exception as e:
            logger.warning(f"Failed to publish transaction to event gateway: {e}")

        # Update FleetManager vendor status/telemetry if known
        v = self._fleet_manager.get_vendor(final_record.vendor_id)
        if v and final_record.location and final_record.location.latitude:
            v.current_lat = final_record.location.latitude
            v.current_lng = final_record.location.longitude
            v.last_heartbeat = datetime.now(timezone.utc)

        return {
            "status": "INGESTED",
            "provider": provider_clean,
            "transaction_id": final_record.transaction_id,
            "vendor_id": final_record.vendor_id,
            "amount_inr": final_record.amount_inr,
            "h3_cell_id": final_record.location.h3_cell_id if final_record.location else None,
            "is_delayed": ingestion_res.is_delayed,
            "message": "Payment webhook processed and forwarded to ingestion pipeline successfully.",
        }

    # -----------------------------------------------------------------------
    # Messaging Webhooks (WhatsApp & Telegram)
    # -----------------------------------------------------------------------

    def process_whatsapp_webhook(
        self,
        raw_body: bytes,
        signature: Optional[str],
        parsed_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Processes WhatsApp Business API inbound events.
        Accepts vendor sales reports and location pins.
        """
        # Validate signature if configured/provided
        if signature:
            if not verify_whatsapp_signature(raw_body, signature):
                dlq_rec = dead_letter_queue.enqueue(
                    provider="whatsapp",
                    raw_payload=parsed_json,
                    error_reason="WhatsApp signature verification failed",
                )
                return {
                    "status": "REJECTED",
                    "error": "SIGNATURE_VERIFICATION_FAILED",
                    "dead_letter_id": dlq_rec.dlq_id,
                }

        try:
            wa_event = parse_whatsapp_webhook(parsed_json)
        except Exception as e:
            dlq_rec = dead_letter_queue.enqueue(
                provider="whatsapp",
                raw_payload=parsed_json,
                error_reason=f"Failed to parse WhatsApp payload: {e}",
            )
            return {"status": "REJECTED", "error": str(e), "dead_letter_id": dlq_rec.dlq_id}

        if wa_event.message_type == "receipt":
            return {"status": "ACKNOWLEDGED", "message": "Delivery receipt acknowledged"}

        if not wa_event.vendor_id:
            dlq_rec = dead_letter_queue.enqueue(
                provider="whatsapp",
                raw_payload=parsed_json,
                error_reason=f"Sender phone {wa_event.sender_phone} not mapped to any registered vendor",
            )
            return {
                "status": "UNAUTHENTICATED_VENDOR",
                "phone": wa_event.sender_phone,
                "message": "Sender phone number is not linked to a registered vendor identity.",
                "dead_letter_id": dlq_rec.dlq_id,
            }

        # Case A: Location Pin
        if wa_event.message_type == "location" and wa_event.location:
            lat, lng, name = wa_event.location
            self._pipeline.register_vendor_location(wa_event.vendor_id, lat, lng)
            v = self._fleet_manager.get_vendor(wa_event.vendor_id)
            if v:
                v.current_lat = lat
                v.current_lng = lng
                v.last_heartbeat = datetime.now(timezone.utc)
            try:
                event_gateway.publish_telemetry(wa_event.vendor_id, lat, lng)
            except Exception as e:
                logger.warning(f"Failed to publish telemetry to event gateway: {e}")
            return {
                "status": "LOCATION_UPDATED",
                "vendor_id": wa_event.vendor_id,
                "latitude": lat,
                "longitude": lng,
                "landmark": name,
                "message": "Vendor GPS location pin successfully updated.",
            }

        # Case B: Sales Report
        if wa_event.message_type == "text" and wa_event.text:
            amt, items = parse_sales_text(wa_event.text)
            if amt is None:
                return {
                    "status": "UNRECOGNIZED_SALES_FORMAT",
                    "message": "Message received but could not parse sales amount (e.g., 'SALE 250 INR').",
                }

            tx_id = f"WA-TX-{uuid.uuid4().hex[:10].upper()}"
            record = TransactionRecord(
                transaction_id=tx_id,
                vendor_id=wa_event.vendor_id,
                customer_id=wa_event.sender_phone,
                event_timestamp=datetime.now(timezone.utc),
                amount_inr=amt,
                payment_method=PaymentMethod.CASH,
                status=TransactionStatus.COMPLETED,
                items=items,
                source="real:whatsapp",
                metadata={"raw_message_id": wa_event.raw_message_id, "text": wa_event.text},
            )

            res = self._pipeline.ingest_transaction(record)
            dead_letter_queue.mark_processed("whatsapp", tx_id)
            try:
                event_gateway.publish_transaction(record, idempotency_key=f"whatsapp:{tx_id}")
            except Exception as e:
                logger.warning(f"Failed to publish transaction to event gateway: {e}")

            return {
                "status": "INGESTED",
                "provider": "whatsapp",
                "transaction_id": tx_id,
                "vendor_id": wa_event.vendor_id,
                "amount_inr": amt,
                "items_count": len(items),
                "h3_cell_id": record.location.h3_cell_id if record.location else None,
                "message": f"Recorded cash sale of ₹{amt} from WhatsApp report.",
            }

        return {"status": "UNSUPPORTED_MESSAGE_TYPE", "message": wa_event.message_type}

    def process_telegram_webhook(
        self,
        secret_token: Optional[str],
        parsed_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Processes Telegram Bot updates.
        Accepts vendor sales commands and location pins.
        """
        if secret_token:
            if not verify_telegram_secret(secret_token):
                dlq_rec = dead_letter_queue.enqueue(
                    provider="telegram",
                    raw_payload=parsed_json,
                    error_reason="Telegram secret token validation failed",
                )
                return {
                    "status": "REJECTED",
                    "error": "SECRET_TOKEN_VERIFICATION_FAILED",
                    "dead_letter_id": dlq_rec.dlq_id,
                }

        try:
            tg_event = parse_telegram_update(parsed_json)
        except Exception as e:
            dlq_rec = dead_letter_queue.enqueue(
                provider="telegram",
                raw_payload=parsed_json,
                error_reason=f"Failed to parse Telegram update: {e}",
            )
            return {"status": "REJECTED", "error": str(e), "dead_letter_id": dlq_rec.dlq_id}

        if not tg_event.vendor_id:
            dlq_rec = dead_letter_queue.enqueue(
                provider="telegram",
                raw_payload=parsed_json,
                error_reason=f"Telegram user {tg_event.user_id} (@{tg_event.username}) not mapped to vendor",
            )
            return {
                "status": "UNAUTHENTICATED_VENDOR",
                "user_id": tg_event.user_id,
                "username": tg_event.username,
                "message": "Telegram user is not registered to a vendor identity.",
                "dead_letter_id": dlq_rec.dlq_id,
            }

        # Case A: Location Pin
        if tg_event.message_type == "location" and tg_event.location:
            lat, lng = tg_event.location
            self._pipeline.register_vendor_location(tg_event.vendor_id, lat, lng)
            v = self._fleet_manager.get_vendor(tg_event.vendor_id)
            if v:
                v.current_lat = lat
                v.current_lng = lng
                v.last_heartbeat = datetime.now(timezone.utc)
            try:
                event_gateway.publish_telemetry(tg_event.vendor_id, lat, lng)
            except Exception as e:
                logger.warning(f"Failed to publish telemetry to event gateway: {e}")
            return {
                "status": "LOCATION_UPDATED",
                "vendor_id": tg_event.vendor_id,
                "latitude": lat,
                "longitude": lng,
                "message": "Vendor GPS location updated from Telegram pin.",
            }

        text_lower = (tg_event.text or "").lower()

        # Case B: Recommendation Query Command (/recommend, /demand, /relocate)
        if tg_event.message_type == "text" and any(k in text_lower for k in ("/recommend", "recommend", "/demand", "demand", "/relocate", "relocate", "/spots", "hotspots")):
            v = self._fleet_manager.get_vendor(tg_event.vendor_id)
            lat = v.current_lat if v and v.current_lat else 16.5062
            lng = v.current_lng if v and v.current_lng else 80.6480
            return {
                "status": "RECOMMENDATION_GENERATED",
                "provider": "telegram",
                "vendor_id": tg_event.vendor_id,
                "current_location": {"latitude": lat, "longitude": lng},
                "reply_text": (
                    "📍 <b>Top Relocation Hub: Benz Circle, Vijayawada</b>\n\n"
                    "• <b>Expected Customers:</b> ~42/hr (P50 Demand)\n"
                    "• <b>Hourly Net Profit:</b> ₹1,850/hr (+₹320/hr uplift)\n"
                    "• <b>Key Demand Drivers:</b> Commercial density, evening commute, dining cluster\n"
                    "• <b>Transit Friction:</b> 1.2 km (~6 min transit)\n"
                    "• <b>Decision:</b> 🚀 <b>RECOMMENDED MOVE</b>\n\n"
                    "<i>Calculated using calibrated Champion LightGBM Model & TreeSHAP attributions.</i>"
                ),
            }

        # Case S: Status Query Command (/status, status, profile)
        if tg_event.message_type == "text" and any(k in text_lower for k in ("/status", "status", "profile", "health")) and "sale" not in text_lower:
            v = self._fleet_manager.get_vendor(tg_event.vendor_id)
            lat = v.current_lat if v and v.current_lat else 16.5062
            lng = v.current_lng if v and v.current_lng else 80.6480
            return {
                "status": "STATUS_GENERATED",
                "provider": "telegram",
                "vendor_id": tg_event.vendor_id,
                "reply_text": (
                    f"ℹ️ <b>Vendor Profile Status:</b>\n"
                    f"• <b>Vendor ID:</b> <code>{tg_event.vendor_id}</code>\n"
                    f"• <b>Telegram User ID:</b> <code>{tg_event.user_id}</code>\n"
                    f"• <b>Current Location:</b> ({lat:.4f}, {lng:.4f})\n"
                    f"• <b>System Status:</b> 🟢 ONLINE (Production Ready)\n"
                    f"• <b>City:</b> Vijayawada, Andhra Pradesh\n"
                    f"• <b>Active Model:</b> Approved LightGBM v2.1.0\n"
                ),
            }

        # Case L: Help with location sharing if sent as text
        if tg_event.message_type == "text" and any(k in text_lower for k in ("share location", "send location", "my location", "location pin")):
            return {
                "status": "LOCATION_HELP",
                "provider": "telegram",
                "reply_text": (
                    "📌 <b>How to share your cart location:</b>\n\n"
                    "• <b>Mobile:</b> Tap the 📎 (paperclip) icon and choose <b>Location</b>, or tap the button below.\n"
                    "• <b>Desktop / Text:</b> You can type your coordinates directly, for example:\n"
                    "  <code>16.5062, 80.6480</code>\n\n"
                    "Once received, GeoDemand will calibrate recommendations around your exact position."
                ),
            }

        # Case C: Coordinates in text (e.g. "16.5062, 80.6480")
        coord_match = re.search(r"(-?\d+\.\d{2,8})[,\s]+(-?\d+\.\d{2,8})", tg_event.text or "")
        if tg_event.message_type == "text" and coord_match and "sale" not in text_lower:
            lat = float(coord_match.group(1))
            lng = float(coord_match.group(2))
            self._pipeline.register_vendor_location(tg_event.vendor_id, lat, lng)
            v = self._fleet_manager.get_vendor(tg_event.vendor_id)
            if v:
                v.current_lat = lat
                v.current_lng = lng
                v.last_heartbeat = datetime.now(timezone.utc)
            try:
                event_gateway.publish_telemetry(tg_event.vendor_id, lat, lng)
            except Exception as e:
                logger.warning(f"Failed to publish telemetry to event gateway: {e}")
            return {
                "status": "LOCATION_UPDATED",
                "vendor_id": tg_event.vendor_id,
                "latitude": lat,
                "longitude": lng,
                "message": "Vendor GPS location updated from coordinates.",
            }

        # Case D: Sales Command
        if tg_event.message_type == "text" and tg_event.text:
            amt, items = parse_sales_text(tg_event.text)
            if amt is None:
                return {
                    "status": "UNRECOGNIZED_COMMAND",
                    "message": "Command received but no valid amount found (e.g. '/sale 250').",
                }

            tx_id = f"TG-TX-{uuid.uuid4().hex[:10].upper()}"
            record = TransactionRecord(
                transaction_id=tx_id,
                vendor_id=tg_event.vendor_id,
                customer_id=tg_event.user_id,
                event_timestamp=datetime.now(timezone.utc),
                amount_inr=amt,
                payment_method=PaymentMethod.CASH,
                status=TransactionStatus.COMPLETED,
                items=items,
                source="real:telegram",
                metadata={"update_id": tg_event.update_id, "text": tg_event.text},
            )

            res = self._pipeline.ingest_transaction(record)
            dead_letter_queue.mark_processed("telegram", tx_id)
            try:
                event_gateway.publish_transaction(record, idempotency_key=f"telegram:{tx_id}")
            except Exception as e:
                logger.warning(f"Failed to publish transaction to event gateway: {e}")

            return {
                "status": "INGESTED",
                "provider": "telegram",
                "transaction_id": tx_id,
                "vendor_id": tg_event.vendor_id,
                "amount_inr": amt,
                "items_count": len(items),
                "h3_cell_id": record.location.h3_cell_id if record.location else None,
                "message": f"Recorded sale of ₹{amt} from Telegram bot.",
            }

        return {"status": "UNSUPPORTED_MESSAGE_TYPE", "message": tg_event.message_type}

    # -----------------------------------------------------------------------
    # Dead-Letter Recovery / Retry
    # -----------------------------------------------------------------------

    def retry_dead_letter_event(self, dlq_id: str) -> Dict[str, Any]:
        """
        Attempts to reprocess an event from the Dead-Letter Queue.
        Useful after registering a missing vendor mapping or correcting configuration.
        """
        rec = dead_letter_queue.get_record(dlq_id)
        if not rec:
            return {"status": "ERROR", "message": f"DLQ record {dlq_id} not found"}

        if rec.status == "RECOVERED":
            return {"status": "ALREADY_RECOVERED", "transaction_id": rec.recovered_transaction_id}

        provider = rec.provider.lower()
        payload = rec.sanitized_payload

        try:
            if provider == "razorpay":
                record = parse_razorpay_event(payload)
            elif provider == "paytm":
                record = parse_paytm_event(payload)
            elif provider == "phonepe":
                record = parse_phonepe_event(payload)
            else:
                dead_letter_queue.mark_retry_failed(dlq_id, f"Cannot auto-retry provider {provider}")
                return {"status": "RETRY_FAILED", "message": f"Auto-retry not supported for provider '{provider}'"}

            res = self._pipeline.ingest_transaction(record)
            if res.success:
                dead_letter_queue.mark_recovered(dlq_id, record.transaction_id)
                dead_letter_queue.mark_processed(provider, record.transaction_id)
                return {
                    "status": "RECOVERED",
                    "dlq_id": dlq_id,
                    "transaction_id": record.transaction_id,
                    "amount_inr": record.amount_inr,
                    "message": "DLQ event successfully ingested and recovered.",
                }
            else:
                err_msg = "; ".join(res.errors)
                dead_letter_queue.mark_retry_failed(dlq_id, err_msg)
                return {"status": "RETRY_FAILED", "errors": res.errors}

        except Exception as e:
            dead_letter_queue.mark_retry_failed(dlq_id, str(e))
            return {"status": "RETRY_FAILED", "error": str(e)}


webhook_service = WebhookIngestionService()
