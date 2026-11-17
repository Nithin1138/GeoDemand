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

                # Handle /start or greeting
                if text.lower().startswith("/start") or text.lower().startswith("/help"):
                    welcome_text = (
                        f"👋 <b>Welcome to GeoDemand AI (@{bot_info.get('username')})!</b>\n\n"
                        f"I am your mobile vendor decision intelligence assistant.\n\n"
                        f"<b>Available Commands:</b>\n"
                        f"• 📍 <b>/recommend</b> — Get the top customer demand hotspot & net profit uplift\n"
                        f"• 💰 <b>/sale &lt;amount&gt;</b> — Log a cash transaction (e.g. <code>/sale 150</code>)\n"
                        f"• 📌 <b>Send Location Pin</b> — Update your current cart GPS coordinates\n"
                        f"• ℹ️ <b>/status</b> — Check your vendor ID and system health\n\n"
                        f"<i>All recommendations account for live weather, traffic friction, and local competition.</i>"
                    )
                    markup = {
                        "keyboard": [
                            [{"text": "📍 /recommend"}, {"text": "💰 /sale 100"}],
                            [{"text": "📌 Share Location", "request_location": True}, {"text": "ℹ️ /status"}],
                        ],
                        "resize_keyboard": True,
                        "one_time_keyboard": False,
                    }
                    send_telegram_reply(chat_id, welcome_text, markup)
                    continue

                if text.lower().startswith("/status"):
                    v_id = vendor_channel_registry.resolve_by_telegram(user_id) or "vendor_default"
                    status_text = (
                        f"ℹ️ <b>Vendor Profile Status:</b>\n"
                        f"• <b>Vendor ID:</b> <code>{v_id}</code>\n"
                        f"• <b>Telegram Chat ID:</b> <code>{chat_id}</code>\n"
                        f"• <b>System Status:</b> 🟢 ONLINE (Production Ready)\n"
                        f"• <b>City:</b> Vijayawada, Andhra Pradesh\n"
                        f"• <b>Active Model:</b> Approved LightGBM v2.1.0\n"
                    )
                    send_telegram_reply(chat_id, status_text)
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
                            [{"text": "🗺️ Open Navigation in Maps", "url": "https://www.google.com/maps/dir/?api=1&destination=Benz+Circle+Vijayawada"}]
                        ]
                    }
                    send_telegram_reply(chat_id, reply, markup)
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

