#!/usr/bin/env python3
"""
GeoDemand AI — Interactive Telegram Bot Polling Runner
Allows vendors, operators, and developers to interact live with @GeoDemandBot:
- /start : Welcome message & registration
- /recommend : Real-time footfall & profit relocation recommendation
- /sale <amount> : Record cash sales directly into the ground-truth ingestion pipeline
- Location Pins : Update vendor GPS coordinates in FleetManager
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "src" / "api"))
sys.path.insert(0, str(ROOT_DIR / "src" / "fleet"))
sys.path.insert(0, str(ROOT_DIR / "src" / "ingestion"))
sys.path.insert(0, str(ROOT_DIR / "src" / "notifications"))

load_dotenv(ROOT_DIR / ".env")

from api.webhooks.service import webhook_service
from api.webhooks.messaging_adapters import vendor_channel_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("telegram_bot_runner")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set in .env! Exiting.")
    sys.exit(1)

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


VIJAYAWADA_HUBS = {
    "benz": (16.5062, 80.6480, "Benz Circle / Labbipet"),
    "station": (16.5186, 80.6198, "Railway Station / Gandhi Nagar"),
    "railway": (16.5186, 80.6198, "Railway Station / Gandhi Nagar"),
    "governorpet": (16.5110, 80.6320, "Governorpet / MG Road"),
    "mg road": (16.5110, 80.6320, "Governorpet / MG Road"),
    "patamata": (16.4965, 80.6710, "Patamata / Auto Nagar"),
    "auto nagar": (16.4965, 80.6710, "Patamata / Auto Nagar"),
    "gunadala": (16.5280, 80.6720, "Gunadala / Ramavarappadu"),
}

LOCATION_PICKER_MARKUP = {
    "inline_keyboard": [
        [{"text": "📍 Benz Circle / Labbipet", "callback_data": "loc:16.5062:80.6480:Benz Circle"}],
        [{"text": "🚉 Railway Station / Gandhi Nagar", "callback_data": "loc:16.5186:80.6198:Railway Station"}],
        [{"text": "🛍️ Governorpet / MG Road", "callback_data": "loc:16.5110:80.6320:MG Road"}],
        [{"text": "🏭 Patamata / Auto Nagar", "callback_data": "loc:16.4965:80.6710:Auto Nagar"}],
        [{"text": "⛪ Gunadala / Ramavarappadu", "callback_data": "loc:16.5280:80.6720:Gunadala"}],
    ]
}

MAIN_KEYBOARD = {
    "keyboard": [
        [{"text": "📍 /recommend"}, {"text": "💰 /sale 100"}],
        [{"text": "📌 Update Location"}, {"text": "ℹ️ /status"}],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": False,
}


def send_telegram_reply(chat_id: int, text: str, reply_markup: dict = None):
    url = f"{BASE_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram reply to {chat_id}: {e}")
        return None


def poll_updates():
    logger.info("🤖 Starting GeoDemand Telegram Bot live polling runner...")
    # Verify bot identity
    me_resp = requests.get(f"{BASE_URL}/getMe", timeout=10).json()
    if not me_resp.get("ok"):
        logger.error(f"Failed to connect to Telegram: {me_resp}")
        sys.exit(1)

    bot_info = me_resp["result"]
    logger.info(f"✅ Connected to @{bot_info.get('username')} ({bot_info.get('first_name')}) [ID: {bot_info.get('id')}]")
    logger.info("Listening for messages... (Press Ctrl+C to stop)")

    last_update_id = 0

    while True:
        try:
            params = {"offset": last_update_id + 1, "timeout": 20}
            resp = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=25)
            if resp.status_code != 200:
                time.sleep(2)
                continue

            data = resp.json()
            updates = data.get("result", [])

            for u in updates:
                last_update_id = u["update_id"]

                # 1. Handle Inline Button Clicks (Callback Queries)
                cb = u.get("callback_query")
                if cb:
                    cb_id = cb.get("id")
                    cb_data = cb.get("data", "")
                    cb_from = cb.get("from", {})
                    cb_user_id = str(cb_from.get("id"))
                    cb_username = cb_from.get("username")
                    cb_chat_id = cb.get("message", {}).get("chat", {}).get("id")

                    try:
                        requests.post(f"{BASE_URL}/answerCallbackQuery", json={"callback_query_id": cb_id}, timeout=5)
                    except Exception as e:
                        logger.warning(f"Failed to answer callback query: {e}")

                    if cb_data.startswith("loc:"):
                        parts = cb_data.split(":")
                        lat = float(parts[1])
                        lng = float(parts[2])
                        loc_name = parts[3] if len(parts) > 3 else "Selected Hub"

                        v_id = vendor_channel_registry.resolve_by_telegram(cb_user_id) or f"vendor_tg_{cb_user_id[-4:]}"
                        vendor_channel_registry.register(v_id, telegram_id=cb_user_id)
                        if cb_username:
                            vendor_channel_registry.register(v_id, telegram_id=cb_username)

                        webhook_service._pipeline.register_vendor_location(v_id, lat, lng)
                        v = webhook_service._fleet_manager.get_vendor(v_id)
                        if v:
                            v.current_lat = lat
                            v.current_lng = lng
                            v.last_heartbeat = datetime.now(timezone.utc)

                        reply = (
                            f"📍 <b>Location Updated to {loc_name}!</b>\n"
                            f"• <b>Coordinates:</b> ({lat:.4f}, {lng:.4f})\n"
                            f"• <b>Fleet Status:</b> Synchronized\n\n"
                            f"Tap <b>📍 /recommend</b> to view top demand hotspots and profit uplift from here!"
                        )
                        send_telegram_reply(cb_chat_id, reply, MAIN_KEYBOARD)
                    continue

                msg = u.get("message")
                if not msg:
                    continue

                chat_id = msg.get("chat", {}).get("id")
                from_user = msg.get("from", {})
                user_id = str(from_user.get("id"))
                username = from_user.get("username")
                text = msg.get("text", "").strip()
                location = msg.get("location")

                logger.info(f"Received from @{username or user_id}: text='{text}', location={bool(location)}")

                # Auto-register user in vendor registry if not present
                existing_vendor = vendor_channel_registry.resolve_by_telegram(user_id)
                if not existing_vendor:
                    assigned_vendor = f"vendor_tg_{user_id[-4:]}"
                    vendor_channel_registry.register(assigned_vendor, telegram_id=user_id)
                    if username:
                        vendor_channel_registry.register(assigned_vendor, telegram_id=username)
                    logger.info(f"Auto-registered Telegram user {user_id} (@{username}) as '{assigned_vendor}'")

                text_lower = text.lower()

                # Handle /start or greeting
                if any(k in text_lower for k in ("/start", "/help", "start", "help", "hello", "hi")) and "sale" not in text_lower:
                    welcome_text = (
                        f"👋 <b>Welcome to GeoDemand AI (@{bot_info.get('username')})!</b>\n\n"
                        f"I am your mobile vendor decision intelligence assistant.\n\n"
                        f"<b>Available Commands:</b>\n"
                        f"• 📍 <b>/recommend</b> — Get the top customer demand hotspot & net profit uplift\n"
                        f"• 💰 <b>/sale &lt;amount&gt;</b> — Log a cash transaction (e.g. <code>/sale 150</code>)\n"
                        f"• 📌 <b>Update Location</b> — Select your trading hub or send GPS coordinates\n"
                        f"• ℹ️ <b>/status</b> — Check your vendor ID and system health\n\n"
                        f"<i>All recommendations account for live weather, traffic friction, and local competition.</i>"
                    )
                    send_telegram_reply(chat_id, welcome_text, MAIN_KEYBOARD)
                    continue

                # Handle /status
                if any(k in text_lower for k in ("/status", "status", "profile", "health")) and "sale" not in text_lower:
                    v_id = vendor_channel_registry.resolve_by_telegram(user_id) or "vendor_default"
                    v = webhook_service._fleet_manager.get_vendor(v_id)
                    lat = v.current_lat if v and v.current_lat else 16.5062
                    lng = v.current_lng if v and v.current_lng else 80.6480
                    status_text = (
                        f"ℹ️ <b>Vendor Profile Status:</b>\n"
                        f"• <b>Vendor ID:</b> <code>{v_id}</code>\n"
                        f"• <b>Telegram Chat ID:</b> <code>{chat_id}</code>\n"
                        f"• <b>Current Location:</b> ({lat:.4f}, {lng:.4f})\n"
                        f"• <b>System Status:</b> 🟢 ONLINE (Production Ready)\n"
                        f"• <b>City:</b> Vijayawada, Andhra Pradesh\n"
                        f"• <b>Active Model:</b> Approved LightGBM v2.1.0\n"
                    )
                    send_telegram_reply(chat_id, status_text, MAIN_KEYBOARD)
                    continue

                # Handle Location Update / Picker
                if any(k in text_lower for k in ("location", "share location", "update location", "gps", "pin", "/location", "/loc")) and "sale" not in text_lower:
                    matched_hub = None
                    for hub_key, (h_lat, h_lng, h_name) in VIJAYAWADA_HUBS.items():
                        if hub_key in text_lower:
                            matched_hub = (h_lat, h_lng, h_name)
                            break

                    if matched_hub:
                        h_lat, h_lng, h_name = matched_hub
                        v_id = vendor_channel_registry.resolve_by_telegram(user_id) or f"vendor_tg_{user_id[-4:]}"
                        webhook_service._pipeline.register_vendor_location(v_id, h_lat, h_lng)
                        v = webhook_service._fleet_manager.get_vendor(v_id)
                        if v:
                            v.current_lat = h_lat
                            v.current_lng = h_lng
                            v.last_heartbeat = datetime.now(timezone.utc)
                        reply = (
                            f"📍 <b>Location Updated to {h_name}!</b>\n"
                            f"• <b>Coordinates:</b> ({h_lat:.4f}, {h_lng:.4f})\n"
                            f"• <b>Fleet Status:</b> Synchronized\n\n"
                            f"Tap <b>📍 /recommend</b> to get nearby demand hotspots from this location!"
                        )
                        send_telegram_reply(chat_id, reply, MAIN_KEYBOARD)
                        continue

                    picker_text = (
                        "📍 <b>Select or Update Your Cart Location:</b>\n\n"
                        "Tap your current trading hub below:\n"
                        "• <i>Or type GPS coordinates:</i> <code>16.5062, 80.6480</code>\n"
                        "• <i>On Mobile:</i> Use 📎 (paperclip) &gt; <b>Location</b>"
                    )
                    send_telegram_reply(chat_id, picker_text, LOCATION_PICKER_MARKUP)
                    continue

                # Process update through Webhook Ingestion Service
                webhook_res = webhook_service.process_telegram_webhook(
                    secret_token=os.getenv("TELEGRAM_WEBHOOK_SECRET"),
                    parsed_json=u,
                )

                status = webhook_res.get("status")

                if status == "RECOMMENDATION_GENERATED":
                    reply = webhook_res.get("reply_text", "Recommendation generated.")
                    markup = {
                        "inline_keyboard": [
                            [{"text": "🗺️ Open Navigation in Maps", "url": "https://www.google.com/maps/dir/?api=1&destination=16.5062,80.6480"}]
                        ]
                    }
                    send_telegram_reply(chat_id, reply, markup)
                elif status in ("STATUS_GENERATED", "LOCATION_HELP"):
                    reply = webhook_res.get("reply_text", "Information:")
                    send_telegram_reply(chat_id, reply)
                elif status == "INGESTED":
                    amt = webhook_res.get("amount_inr")
                    tx_id = webhook_res.get("transaction_id")
                    reply = (
                        f"✅ <b>Transaction Recorded!</b>\n"
                        f"• <b>Amount:</b> ₹{amt:.2f}\n"
                        f"• <b>Transaction ID:</b> <code>{tx_id}</code>\n"
                        f"• <b>Status:</b> Ingested to Feature Store\n"
                    )
                    send_telegram_reply(chat_id, reply)
                elif status == "LOCATION_UPDATED":
                    lat = webhook_res.get("latitude")
                    lng = webhook_res.get("longitude")
                    reply = (
                        f"📍 <b>Location Updated Successfully!</b>\n"
                        f"• Coordinates: ({lat:.4f}, {lng:.4f})\n"
                        f"• Fleet Manager synchronized.\n"
                        f"Type /recommend to see nearby demand hotspots from this location."
                    )
                    send_telegram_reply(chat_id, reply)
                elif status == "UNRECOGNIZED_COMMAND":
                    send_telegram_reply(chat_id, "❓ Unrecognized format. Type <code>/sale &lt;amount&gt;</code> or tap <b>/recommend</b>.")
                else:
                    send_telegram_reply(chat_id, "👍 Message received and logged.")

        except requests.exceptions.Timeout:
            continue
        except KeyboardInterrupt:
            logger.info("Bot polling stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in polling loop: {e}")
            time.sleep(3)


if __name__ == "__main__":
    poll_updates()

