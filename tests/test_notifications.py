"""
Comprehensive unit and integration test suite for the GeoDemand AI Notification Subsystem:
- Forecast freshness verification & safety invariants
- Impending-rain alerts with sheltered zone relocation & profit uplift
- Event-demand alerts with crowd attendance
- Demand-spike alerts
- FCM push notification delivery & error handling
- WhatsApp Business API conversational delivery & error handling
- Vendor preferences, quiet-hours, and opt-out suppression
- Duplicate notification prevention (idempotency cooldown)
- Notification expiry
- Delivery telemetry tracking & FastAPI endpoints
"""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from notifications.models import (
    AlertType,
    AlertSeverity,
    NotificationChannel,
    DeliveryStatus,
    ForecastFreshnessMetadata,
    RelocationDetails,
    NotificationPreferences,
    NotificationMessage,
    DeliveryRecord,
)
from notifications.fcm_adapter import FCMAdapter
from notifications.whatsapp_adapter import WhatsAppNotificationAdapter
from notifications.telegram_adapter import TelegramNotificationAdapter
from notifications.notification_engine import NotificationTriggerService


# ===========================================================================
# 1. Forecast Freshness & Reliability Safety Invariant Tests
# ===========================================================================

def test_forecast_freshness_verified_live():
    service = NotificationTriggerService()
    now_str = datetime.now(timezone.utc).isoformat()
    weather_live = {
        "source_type": "real_live",
        "source_name": "open_meteo",
        "retrieved_at": now_str,
        "cache_age_seconds": 120.0,
    }
    meta = service.verify_forecast_freshness(weather_live)
    assert meta.is_reliable is True
    assert meta.source_type == "real_live"
    assert "VERIFIED_LIVE" in meta.reliability_reason


def test_forecast_freshness_rejects_synthetic_fallback():
    service = NotificationTriggerService()
    weather_synth = {
        "source_type": "synthetic_fallback",
        "source_name": "weather_defaults",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "cache_age_seconds": 10.0,
    }
    meta = service.verify_forecast_freshness(weather_synth)
    assert meta.is_reliable is False
    assert "UNRELIABLE_SYNTHETIC" in meta.reliability_reason


def test_forecast_freshness_rejects_stale_cache():
    service = NotificationTriggerService()
    old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    weather_stale = {
        "source_type": "real_live",
        "source_name": "open_meteo",
        "retrieved_at": old_time,
        "cache_age_seconds": 7200.0,
    }
    meta = service.verify_forecast_freshness(weather_stale)
    assert meta.is_reliable is False
    assert "STALE_FORECAST" in meta.reliability_reason


def test_rain_alert_safety_invariant_blocks_move_on_stale_forecast():
    """
    CRITICAL SAFETY INVARIANT:
    Ensure notifications do not recommend a move using stale or unreliable forecasts.
    """
    service = NotificationTriggerService()
    stale_weather = {
        "source_type": "synthetic_fallback",
        "source_name": "weather_defaults",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "precipitation_probability": 85.0,
        "rainfall": 4.5,
    }
    msg = service.evaluate_rain_alert(
        vendor_id="vendor_safety_test",
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        custom_forecast=stale_weather,
    )
    assert msg is not None
    assert msg.alert_type == AlertType.IMPENDING_RAIN
    assert msg.relocation is not None
    # Crucial assertion: must NOT recommend moving
    assert msg.relocation.is_move_recommended is False
    assert "DO NOT RELOCATE" in msg.body
    assert msg.relocation.forecast_freshness.is_reliable is False


# ===========================================================================
# 2. Impending-Rain Alerts with Sheltered Relocation & Profit Uplift
# ===========================================================================

def test_impending_rain_triggers_sheltered_move_when_reliable():
    service = NotificationTriggerService()
    live_rain = {
        "source_type": "real_live",
        "source_name": "open_meteo",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "cache_age_seconds": 60.0,
        "precipitation_probability": 80.0,
        "rainfall": 2.5,
    }
    msg = service.evaluate_rain_alert(
        vendor_id="vendor_001",
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        custom_forecast=live_rain,
    )
    assert msg is not None
    assert msg.alert_type == AlertType.IMPENDING_RAIN
    assert msg.severity == AlertSeverity.CRITICAL
    assert msg.relocation is not None
    assert msg.relocation.is_move_recommended is True
    assert msg.relocation.destination_h3_cell != ""
    assert msg.relocation.expected_profit_uplift_inr > 0
    assert msg.relocation.recommended_departure_time != ""
    assert msg.relocation.expected_arrival_time != ""
    assert msg.relocation.forecast_freshness.is_reliable is True


def test_clear_weather_does_not_trigger_rain_alert():
    service = NotificationTriggerService()
    clear_weather = {
        "source_type": "real_live",
        "source_name": "open_meteo",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "precipitation_probability": 5.0,
        "rainfall": 0.0,
    }
    msg = service.evaluate_rain_alert(
        vendor_id="vendor_001",
        latitude=16.5062,
        longitude=80.6480,
        custom_forecast=clear_weather,
    )
    assert msg is None


# ===========================================================================
# 3. Event-Demand Alert Tests
# ===========================================================================

def test_event_demand_alert_trigger():
    service = NotificationTriggerService()
    custom_event = {
        "event_active": True,
        "event_importance": 0.85,
        "event_attendance_estimate": 1500,
        "event_name": "Vijayawada Food & Music Carnival",
        "h3_cell": "88619aa6c1fffff",
    }
    msg = service.evaluate_event_alert(
        vendor_id="vendor_001",
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        custom_event=custom_event,
    )
    assert msg is not None
    assert msg.alert_type == AlertType.EVENT_DEMAND
    assert msg.relocation is not None
    assert msg.relocation.destination_h3_cell == "88619aa6c1fffff"
    assert msg.relocation.expected_profit_uplift_inr >= 40.0
    assert "Carnival" in msg.title


# ===========================================================================
# 4. Demand-Spike Alert Tests
# ===========================================================================

def test_demand_spike_alert_trigger():
    service = NotificationTriggerService()
    candidates = [
        {
            "h3_cell": "88619aa6c3fffff",
            "landmark_name": "Railway Hub Plaza",
            "distance_km": 1.1,
            "travel_time_min": 5,
            "realized_net_uplift_inr": 65.0,
        }
    ]
    msg = service.evaluate_demand_spike_alert(
        vendor_id="vendor_001",
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        candidate_zones=candidates,
    )
    assert msg is not None
    assert msg.alert_type == AlertType.DEMAND_SPIKE
    assert msg.relocation.destination_h3_cell == "88619aa6c3fffff"
    assert msg.relocation.expected_profit_uplift_inr == 65.0


def test_demand_spike_ignored_if_uplift_below_preference():
    service = NotificationTriggerService()
    # Set vendor minimum uplift to ₹80
    service.update_preferences(NotificationPreferences(vendor_id="v_high_thresh", min_profit_uplift_inr=80.0))
    candidates = [
        {
            "h3_cell": "88619aa6c3fffff",
            "landmark_name": "Small Corner",
            "realized_net_uplift_inr": 45.0,  # Below 80.0
        }
    ]
    msg = service.evaluate_demand_spike_alert(
        vendor_id="v_high_thresh",
        latitude=16.5062,
        longitude=80.6480,
        candidate_zones=candidates,
    )
    assert msg is None


# ===========================================================================
# 5. FCM Adapter Tests
# ===========================================================================

def test_fcm_adapter_successful_dispatch():
    adapter = FCMAdapter()
    msg = NotificationMessage(
        notification_id="NOTIF-123",
        vendor_id="v_fcm_test",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Demand Spike",
        body="Surge detected",
    )
    record = adapter.send_notification("fcm_token_valid_device_12345", msg)
    assert record.status == DeliveryStatus.DELIVERED
    assert record.channel == NotificationChannel.FCM
    assert record.upstream_message_id is not None


def test_fcm_adapter_invalid_token_failure():
    adapter = FCMAdapter()
    msg = NotificationMessage(
        notification_id="NOTIF-124",
        vendor_id="v_fcm_test",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Demand Spike",
        body="Surge detected",
    )
    record = adapter.send_notification("", msg)
    assert record.status == DeliveryStatus.FAILED
    assert "INVALID_FCM_TOKEN" in record.error_message


def test_fcm_adapter_unregistered_token_failure():
    adapter = FCMAdapter()
    msg = NotificationMessage(
        notification_id="NOTIF-125",
        vendor_id="v_fcm_test",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Demand Spike",
        body="Surge detected",
    )
    record = adapter.send_notification("token_unregistered_device_999", msg)
    assert record.status == DeliveryStatus.FAILED
    assert "FCM_UNREGISTERED" in record.error_message


# ===========================================================================
# 6. WhatsApp Business API Adapter Tests
# ===========================================================================

def test_whatsapp_adapter_successful_dispatch():
    adapter = WhatsAppNotificationAdapter()
    msg = NotificationMessage(
        notification_id="NOTIF-WA-1",
        vendor_id="v_wa_test",
        alert_type=AlertType.IMPENDING_RAIN,
        severity=AlertSeverity.WARNING,
        title="Impending Rain",
        body="Rain expected in 30 minutes",
        relocation=RelocationDetails(
            destination_h3_cell="88619aa6c1fffff",
            destination_name="Benz Circle Hub",
            distance_km=1.2,
            travel_time_minutes=6,
            expected_profit_uplift_inr=70.0,
            recommended_departure_time="14:30",
            expected_arrival_time="14:36",
            is_move_recommended=True,
            forecast_freshness=ForecastFreshnessMetadata(),
        ),
    )
    record = adapter.send_notification("9876543210", msg)
    assert record.status == DeliveryStatus.DELIVERED
    assert record.channel == NotificationChannel.WHATSAPP
    assert record.recipient == "919876543210"
    assert record.upstream_message_id is not None


def test_whatsapp_adapter_invalid_phone_failure():
    adapter = WhatsAppNotificationAdapter()
    msg = NotificationMessage(
        notification_id="NOTIF-WA-2",
        vendor_id="v_wa_test",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Test",
        body="Test",
    )
    record = adapter.send_notification("123", msg)  # Under 10 digits
    assert record.status == DeliveryStatus.FAILED
    assert "INVALID_PHONE_NUMBER" in record.error_message


# ===========================================================================
# 7. Preferences, Quiet Hours & Opt-Out Suppression Tests
# ===========================================================================

def test_notification_suppressed_when_vendor_opts_out():
    service = NotificationTriggerService()
    service.update_preferences(NotificationPreferences(vendor_id="v_optout", opt_out=True))
    msg = NotificationMessage(
        notification_id="NOTIF-OPT",
        vendor_id="v_optout",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Demand Spike",
        body="Surge detected",
    )
    records = service.dispatch_notification(msg)
    assert len(records) == 1
    assert records[0].status == DeliveryStatus.SUPPRESSED_OPT_OUT


def test_notification_suppressed_during_quiet_hours():
    service = NotificationTriggerService()
    now_hh_mm = datetime.now(timezone.utc).strftime("%H:%M")
    # Set quiet hours to cover current time
    service.update_preferences(NotificationPreferences(
        vendor_id="v_quiet",
        quiet_hours_start="00:00",
        quiet_hours_end="23:59",
    ))
    msg = NotificationMessage(
        notification_id="NOTIF-QUIET",
        vendor_id="v_quiet",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,  # Non-critical
        title="Info Spike",
        body="Surge detected",
    )
    records = service.dispatch_notification(msg)
    assert len(records) == 1
    assert records[0].status == DeliveryStatus.SUPPRESSED_QUIET_HOURS


# ===========================================================================
# 8. Duplicate Prevention & Expiry Tests
# ===========================================================================

def test_duplicate_notification_prevention():
    service = NotificationTriggerService()
    # Disable quiet hours for test vendor so delivery succeeds at any hour
    service.update_preferences(NotificationPreferences(
        vendor_id="v_dup_test",
        quiet_hours_start="",
        quiet_hours_end="",
    ))
    msg1 = NotificationMessage(
        notification_id="NOTIF-DUP-1",
        vendor_id="v_dup_test",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Surge",
        body="Surge",
        idempotency_key="v_dup_test:DEMAND_SPIKE:88619aa6c1fffff:2026111612",
    )
    # First dispatch succeeds
    rec1 = service.dispatch_notification(msg1)
    assert any(r.status == DeliveryStatus.DELIVERED for r in rec1)

    # Second dispatch with same idempotency key is suppressed
    msg2 = NotificationMessage(
        notification_id="NOTIF-DUP-2",
        vendor_id="v_dup_test",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Surge",
        body="Surge",
        idempotency_key="v_dup_test:DEMAND_SPIKE:88619aa6c1fffff:2026111612",
    )
    rec2 = service.dispatch_notification(msg2)
    assert len(rec2) == 1
    assert rec2[0].status == DeliveryStatus.SUPPRESSED_DUPLICATE


def test_expired_notification_suppression():
    service = NotificationTriggerService()
    # Notification that expired 10 minutes ago
    msg_expired = NotificationMessage(
        notification_id="NOTIF-EXP",
        vendor_id="v_exp_test",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Old Spike",
        body="Expired",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=40),
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    records = service.dispatch_notification(msg_expired)
    assert len(records) == 1
    assert records[0].status == DeliveryStatus.EXPIRED


# ===========================================================================
# 9. FastAPI Endpoint Integration Tests
# ===========================================================================

@pytest.fixture(scope="module")
def api_client():
    from api.app import app
    return TestClient(app)


def test_notification_preferences_endpoint(api_client):
    r = api_client.post(
        "/v1/notifications/preferences",
        json={
            "vendor_id": "v_api_test",
            "opt_out": False,
            "channels": ["FCM", "WHATSAPP"],
            "fcm_token": "fcm_token_api_test_123456",
            "whatsapp_phone": "919988776655",
            "alert_impending_rain": True,
            "min_profit_uplift_inr": 50.0,
        },
        headers={"X-API-Key": "dev-local-secret-key"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "UPDATED"
    assert data["preferences"]["vendor_id"] == "v_api_test"
    assert data["preferences"]["min_profit_uplift_inr"] == 50.0

    # Query preferences
    r_get = api_client.get(
        "/v1/notifications/preferences/v_api_test",
        headers={"X-API-Key": "dev-local-secret-key"},
    )
    assert r_get.status_code == 200
    assert r_get.json()["fcm_token"] == "fcm_token_api_test_123456"


def test_notification_trigger_check_endpoint(api_client):
    r = api_client.post(
        "/v1/notifications/trigger-check",
        json={
            "vendor_id": "vendor_001",
            "latitude": 16.5062,
            "longitude": 80.6480,
            "vendor_category": "food",
        },
        headers={"X-API-Key": "dev-local-secret-key"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "EVALUATED"
    assert data["vendor_id"] == "vendor_001"
    assert data["alerts_evaluated_count"] == 3


def test_notification_dispatch_endpoint(api_client):
    r = api_client.post(
        "/v1/notifications/dispatch",
        json={
            "vendor_id": "vendor_001",
            "alert_type": "DEMAND_SPIKE",
            "severity": "INFO",
            "title": "Benz Circle Footfall Surge",
            "body": "Surge footfall detected near Benz Circle hub.",
            "destination_h3_cell": "88619aa6c1fffff",
            "destination_name": "Benz Circle Hub",
            "distance_km": 1.4,
            "travel_time_minutes": 7,
            "expected_profit_uplift_inr": 65.0,
            "channel": "FCM",
            "force": True,
        },
        headers={"X-API-Key": "dev-local-secret-key"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "DISPATCHED"
    assert len(data["records"]) >= 1
    assert data["records"][0]["status"] == "DELIVERED"


def test_notification_history_endpoint(api_client):
    r = api_client.get(
        "/v1/notifications/history/vendor_001",
        headers={"X-API-Key": "dev-local-secret-key"},
    )
    assert r.status_code == 200
    records = r.json()
    assert isinstance(records, list)
    assert len(records) >= 1


# ===========================================================================
# Telegram Bot Adapter & Delivery Tests
# ===========================================================================

def test_telegram_notification_adapter_simulated_delivery():
    adapter = TelegramNotificationAdapter()
    assert adapter.is_live is False

    reloc = RelocationDetails(
        destination_h3_cell="88619aa6c1fffff",
        destination_name="Benz Circle Hub",
        distance_km=1.2,
        travel_time_minutes=6,
        expected_profit_uplift_inr=75.0,
        recommended_departure_time="11:30",
        expected_arrival_time="11:36",
        is_move_recommended=True,
        forecast_freshness=ForecastFreshnessMetadata(),
    )
    msg = NotificationMessage(
        notification_id="NOTIF-TG-TEST-001",
        vendor_id="vendor_001",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="High Customer Demand near Benz Circle",
        body="Demand surge detected with +₹75/hr uplift.",
        relocation=reloc,
    )

    rec = adapter.send_notification("99887766", msg)
    assert rec.status == DeliveryStatus.DELIVERED
    assert rec.channel == NotificationChannel.TELEGRAM
    assert rec.recipient == "99887766"
    assert "MOCK-TG-MSG" in rec.upstream_message_id


def test_telegram_notification_adapter_live_mock():
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 43210}}
        mock_post.return_value = mock_resp

        adapter = TelegramNotificationAdapter(bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
        assert adapter.is_live is True

        msg = NotificationMessage(
            notification_id="NOTIF-TG-LIVE-001",
            vendor_id="vendor_001",
            alert_type=AlertType.IMPENDING_RAIN,
            severity=AlertSeverity.WARNING,
            title="Heavy Rain Imminent",
            body="Rain beginning in 20 min. Deploy cart canopy.",
        )

        rec = adapter.send_notification("99887766", msg)
        assert rec.status == DeliveryStatus.DELIVERED
        assert rec.upstream_message_id == "43210"
        assert mock_post.called


def test_telegram_notification_adapter_invalid_chat_id():
    adapter = TelegramNotificationAdapter()
    msg = NotificationMessage(
        notification_id="NOTIF-TG-INV-001",
        vendor_id="vendor_001",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Test Alert",
        body="Test Body",
    )
    rec = adapter.send_notification("", msg)
    assert rec.status == DeliveryStatus.FAILED
    assert "INVALID_CHAT_ID" in rec.error_message


def test_notification_dispatch_via_telegram_channel():
    service = NotificationTriggerService()
    msg = NotificationMessage(
        notification_id=f"NOTIF-TG-DISP-{datetime.now().strftime('%f')}",
        vendor_id="vendor_002",
        alert_type=AlertType.DEMAND_SPIKE,
        severity=AlertSeverity.INFO,
        title="Telegram Dispatch Test",
        body="Testing Telegram channel dispatch.",
    )
    records = service.dispatch_notification(msg, override_channel=NotificationChannel.TELEGRAM, force=True)
    assert len(records) >= 1
    tg_rec = [r for r in records if r.channel == NotificationChannel.TELEGRAM][0]
    assert tg_rec.status == DeliveryStatus.DELIVERED

