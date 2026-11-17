"""
Telegram Bot Notification Adapter for GeoDemand AI.
Alternative messaging channel to WhatsApp Business API.
Formats interactive HTML/Markdown alert cards, dispatches relocation recommendations,
interfaces with Telegram Bot API, and records delivery telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from notifications.models import (
    AlertSeverity,
    AlertType,
    DeliveryRecord,
    DeliveryStatus,
    NotificationChannel,
    NotificationMessage,
)

logger = logging.getLogger(__name__)


class TelegramNotificationAdapter:
    """
    Telegram Bot API client.
    Sends formatted HTML relocation alerts and interactive buttons to vendors via Telegram.
    Provides a free, instant alternative to WhatsApp Business API.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        timeout_seconds: float = 5.0,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.timeout_seconds = timeout_seconds
        self._is_live = bool(self.bot_token)
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else ""

    @property
    def is_live(self) -> bool:
        return self._is_live

    def send_notification(
        self,
        chat_id: str,
        message: NotificationMessage,
    ) -> DeliveryRecord:
        """
        Formats and delivers an alert message to a vendor's Telegram chat ID or username.
        """
        rec_id = f"TG-REC-{uuid.uuid4().hex[:10].upper()}"
        now = datetime.now(timezone.utc)

        # 1. Validate Chat ID
        clean_chat_id = str(chat_id).strip()
        if not clean_chat_id:
            logger.warning(f"Telegram delivery aborted: Empty chat ID for vendor {message.vendor_id}")
            return DeliveryRecord(
                record_id=rec_id,
                notification_id=message.notification_id,
                vendor_id=message.vendor_id,
                channel=NotificationChannel.TELEGRAM,
                recipient=str(chat_id),
                status=DeliveryStatus.FAILED,
                error_message="INVALID_CHAT_ID: Telegram chat ID cannot be empty",
                failed_at=now,
            )

        # Explicit failure simulation for testing
        if "invalid" in clean_chat_id.lower() or clean_chat_id == "0":
            return DeliveryRecord(
                record_id=rec_id,
                notification_id=message.notification_id,
                vendor_id=message.vendor_id,
                channel=NotificationChannel.TELEGRAM,
                recipient=clean_chat_id,
                status=DeliveryStatus.FAILED,
                error_message="TELEGRAM_CHAT_NOT_FOUND: The recipient has not started the bot",
                failed_at=now,
            )

        # 2. Format HTML Card
        text_body, reply_markup = self._format_telegram_message(clean_chat_id, message)

        # 3. Dispatch Live or Simulated
        if self._is_live:
            return self._dispatch_live(rec_id, clean_chat_id, message, text_body, reply_markup, now)
        else:
            return self._dispatch_simulated(rec_id, clean_chat_id, message, text_body, now)

    def _format_telegram_message(
        self,
        chat_id: str,
        message: NotificationMessage,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Constructs an eye-catching, structured Telegram HTML message card.
        """
        severity_emoji = "⚠️" if message.severity == AlertSeverity.WARNING else "🚨" if message.severity == AlertSeverity.CRITICAL else "ℹ️"
        
        type_banner = {
            AlertType.IMPENDING_RAIN: "🌧️ <b>RAIN ALERT & SHELTER ADVISORY</b>",
            AlertType.EVENT_DEMAND: "🎪 <b>HIGH FOOTFALL EVENT DETECTED</b>",
            AlertType.DEMAND_SPIKE: "📈 <b>CUSTOMER DEMAND SURGE</b>",
            AlertType.GENERAL_ALERT: "📢 <b>OPERATIONAL ADVISORY</b>",
        }.get(message.alert_type, "📢 <b>GEODEMAND ALERT</b>")

        lines = [
            f"{severity_emoji} {type_banner}",
            f"━━━━━━━━━━━━━━━━━━",
            f"<b>{message.title}</b>",
            f"{message.body}",
            "",
        ]

        reply_markup = None

        if message.relocation and message.relocation.is_move_recommended:
            reloc = message.relocation
            lines.extend([
                "📍 <b>Recommended Action: RELOCATE NOW</b>",
                f"• <b>Destination:</b> {reloc.destination_name}",
                f"• <b>Distance:</b> {reloc.distance_km:.1f} km (~{reloc.travel_time_minutes} min transit)",
                f"• <b>Expected Profit Uplift:</b> +₹{reloc.expected_profit_uplift_inr:.0f}/hr",
                f"• <b>Depart By:</b> {reloc.recommended_departure_time} | <b>Arrive By:</b> {reloc.expected_arrival_time}",
                f"• <b>Reason:</b> {reloc.decision_reason}",
                "",
                f"<i>Forecast Freshness: {reloc.forecast_freshness.source_name} ({reloc.forecast_freshness.reliability_reason})</i>",
            ])
            # Inline keyboard button with Google Maps direction link
            maps_url = f"https://www.google.com/maps/dir/?api=1&destination={reloc.destination_name.replace(' ', '+')}"
            reply_markup = {
                "inline_keyboard": [
                    [
                        {"text": "🗺️ Open Navigation in Maps", "url": maps_url},
                        {"text": "✅ Accept Relocation", "callback_data": f"accept:{reloc.destination_h3_cell}"},
                    ]
                ]
            }
        elif message.relocation and not message.relocation.is_move_recommended:
            reloc = message.relocation
            lines.extend([
                "🛡️ <b>Recommended Action: SHELTER IN PLACE</b>",
                f"• <b>Advisory:</b> {reloc.decision_reason}",
                f"• Deploy cart rain canopy and secure electrical equipment.",
                "",
                f"<i>Forecast Status: {reloc.forecast_freshness.reliability_reason}</i>",
            ])

        lines.append("\n<i>Reply /sale &lt;amount&gt; to report cash sales, or send a location pin to update coordinates.</i>")
        return "\n".join(lines), reply_markup

    def _dispatch_live(
        self,
        rec_id: str,
        chat_id: str,
        message: NotificationMessage,
        text_body: str,
        reply_markup: Optional[Dict[str, Any]],
        now: datetime,
    ) -> DeliveryRecord:
        """Dispatches live message payload to Telegram Bot API."""
        try:
            payload: Dict[str, Any] = {
                "chat_id": chat_id,
                "text": text_body,
                "parse_mode": "HTML",
                "disable_web_page_preview": False,
            }
            if reply_markup:
                payload["reply_markup"] = reply_markup

            resp = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout_seconds,
            )

            if resp.status_code == 200:
                data = resp.json()
                msg_id = str(data.get("result", {}).get("message_id", "TG-MSG-UNKNOWN"))
                logger.info(f"Telegram notification sent to chat {chat_id} (msg_id: {msg_id})")
                return DeliveryRecord(
                    record_id=rec_id,
                    notification_id=message.notification_id,
                    vendor_id=message.vendor_id,
                    channel=NotificationChannel.TELEGRAM,
                    recipient=chat_id,
                    status=DeliveryStatus.DELIVERED,
                    dispatched_at=now,
                    delivered_at=datetime.now(timezone.utc),
                    upstream_message_id=msg_id,
                )
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:150]}"
                logger.warning(f"Telegram Bot API delivery failed: {err}")
                return DeliveryRecord(
                    record_id=rec_id,
                    notification_id=message.notification_id,
                    vendor_id=message.vendor_id,
                    channel=NotificationChannel.TELEGRAM,
                    recipient=chat_id,
                    status=DeliveryStatus.FAILED,
                    error_message=err,
                    failed_at=datetime.now(timezone.utc),
                )
        except Exception as e:
            logger.error(f"Telegram network exception during delivery: {e}")
            return DeliveryRecord(
                record_id=rec_id,
                notification_id=message.notification_id,
                vendor_id=message.vendor_id,
                channel=NotificationChannel.TELEGRAM,
                recipient=chat_id,
                status=DeliveryStatus.FAILED,
                error_message=f"NETWORK_ERROR: {str(e)[:150]}",
                failed_at=datetime.now(timezone.utc),
            )

    def _dispatch_simulated(
        self,
        rec_id: str,
        chat_id: str,
        message: NotificationMessage,
        text_body: str,
        now: datetime,
    ) -> DeliveryRecord:
        """Deterministic simulation for local testing or when no bot token is provided."""
        mock_msg_id = f"MOCK-TG-MSG-{uuid.uuid4().hex[:12].upper()}"
        logger.info(
            f"[Simulated Telegram Bot API] Delivered alert '{message.title}' to chat '{chat_id}' "
            f"(msg_id: {mock_msg_id})"
        )
        return DeliveryRecord(
            record_id=rec_id,
            notification_id=message.notification_id,
            vendor_id=message.vendor_id,
            channel=NotificationChannel.TELEGRAM,
            recipient=chat_id,
            status=DeliveryStatus.DELIVERED,
            dispatched_at=now,
            delivered_at=datetime.now(timezone.utc),
            upstream_message_id=mock_msg_id,
        )

