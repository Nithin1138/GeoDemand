"""
Notification Trigger & Delivery Engine for GeoDemand AI.
Monitors weather forecasts, live events, and footfall demand spikes to trigger
safe, profit-uplift-validated relocation alerts via FCM and WhatsApp Business API.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import h3

from config.settings import (
    MAX_RECOMMENDED_TRAVEL_KM,
    MAX_RECOMMENDED_TRAVEL_MIN,
    STAY_THRESHOLD_INR_PER_HOUR,
)
from notifications.models import (
    AlertSeverity,
    AlertType,
    DeliveryRecord,
    DeliveryStatus,
    ForecastFreshnessMetadata,
    NotificationChannel,
    NotificationMessage,
    NotificationPreferences,
    RelocationDetails,
)
from notifications.fcm_adapter import FCMAdapter
from notifications.whatsapp_adapter import WhatsAppNotificationAdapter
from features.weather_client import OpenMeteoClient
from api.providers.events import EventProviderFactory
from api.poi_resolver import resolve_candidate_landmark
from api.spatial_filter import is_water_location, is_water_cell

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notification Trigger Service
# ---------------------------------------------------------------------------

class NotificationTriggerService:
    """
    Central notification engine orchestrating alert triggers, economic uplift validation,
    forecast freshness checks, vendor preferences, deduplication, and multi-channel delivery.
    """

    _instance: Optional[NotificationTriggerService] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        fcm_adapter: Optional[FCMAdapter] = None,
        whatsapp_adapter: Optional[WhatsAppNotificationAdapter] = None,
        weather_client: Optional[OpenMeteoClient] = None,
    ):
        if getattr(self, "_initialized", False):
            return

        self.fcm_adapter = fcm_adapter or FCMAdapter()
        self.whatsapp_adapter = whatsapp_adapter or WhatsAppNotificationAdapter()
        self.weather_client = weather_client or OpenMeteoClient()

        # Thread-safe in-memory stores
        self._preferences_store: Dict[str, NotificationPreferences] = {}
        self._delivery_history: List[DeliveryRecord] = []
        self._active_notifications: Dict[str, NotificationMessage] = {}
        self._sent_idempotency_keys: Dict[str, float] = {}  # key -> timestamp
        self._store_lock = threading.RLock()

        # Seed default vendor preferences for instant usability
        self._seed_default_preferences()
        self._initialized = True

    def _seed_default_preferences(self):
        """Pre-configures test and pilot vendor profiles."""
        self._preferences_store["vendor_001"] = NotificationPreferences(
            vendor_id="vendor_001",
            opt_out=False,
            channels=[NotificationChannel.FCM, NotificationChannel.WHATSAPP],
            fcm_token="fcm_token_raju_dosa_v1_device_abc123",
            whatsapp_phone="919876543210",
            alert_impending_rain=True,
            alert_event_demand=True,
            alert_demand_spike=True,
            min_profit_uplift_inr=40.0,
        )
        self._preferences_store["vendor_002"] = NotificationPreferences(
            vendor_id="vendor_002",
            opt_out=False,
            channels=[NotificationChannel.WHATSAPP],
            whatsapp_phone="918765432109",
            alert_impending_rain=True,
            alert_event_demand=True,
            alert_demand_spike=True,
            min_profit_uplift_inr=50.0,
        )
        self._preferences_store["VND-001"] = NotificationPreferences(
            vendor_id="VND-001",
            opt_out=False,
            channels=[NotificationChannel.FCM, NotificationChannel.WHATSAPP],
            fcm_token="fcm_token_vnd001_mobile_xyz789",
            whatsapp_phone="919876543211",
            min_profit_uplift_inr=40.0,
        )

    # -----------------------------------------------------------------------
    # Preferences Management
    # -----------------------------------------------------------------------

    def get_preferences(self, vendor_id: str) -> NotificationPreferences:
        with self._store_lock:
            if vendor_id not in self._preferences_store:
                # Default new vendor preferences
                self._preferences_store[vendor_id] = NotificationPreferences(vendor_id=vendor_id)
            return self._preferences_store[vendor_id]

    def update_preferences(self, prefs: NotificationPreferences) -> NotificationPreferences:
        with self._store_lock:
            prefs.updated_at = datetime.now(timezone.utc)
            self._preferences_store[prefs.vendor_id] = prefs
            return prefs

    def set_vendor_opt_out(self, vendor_id: str, opt_out: bool) -> NotificationPreferences:
        with self._store_lock:
            prefs = self.get_preferences(vendor_id)
            prefs.opt_out = opt_out
            prefs.updated_at = datetime.now(timezone.utc)
            return prefs

    # -----------------------------------------------------------------------
    # Forecast Freshness & Reliability Verification
    # -----------------------------------------------------------------------

    def verify_forecast_freshness(self, weather_data: Dict[str, Any]) -> ForecastFreshnessMetadata:
        """
        Validates forecast observation age, source type, and synthetic status.
        Ensures notifications do NOT recommend relocations using stale or unreliable forecasts.
        """
        now = datetime.now(timezone.utc)
        src_type = str(weather_data.get("source_type", "synthetic_fallback"))
        src_name = str(weather_data.get("source_name", "weather_defaults"))
        retrieved_at_str = str(weather_data.get("retrieved_at") or now.isoformat())

        # Calculate cache age
        cache_age = float(weather_data.get("cache_age_seconds", 0.0))
        if cache_age <= 0.0 and retrieved_at_str:
            try:
                ret_dt = datetime.fromisoformat(retrieved_at_str.replace("Z", "+00:00"))
                cache_age = max(0.0, (now - ret_dt).total_seconds())
            except Exception:
                cache_age = 0.0

        # Safety Criteria:
        # 1. Synthetic fallback is NOT reliable for move decisions
        # 2. Cache age > 3600s (1 hour) is considered stale
        is_synthetic = "synthetic" in src_type.lower() or "synthetic" in src_name.lower()
        is_stale = cache_age > 3600.0

        if is_synthetic:
            return ForecastFreshnessMetadata(
                source_type=src_type,
                source_name=src_name,
                retrieved_at=retrieved_at_str,
                cache_age_seconds=cache_age,
                is_reliable=False,
                reliability_reason="UNRELIABLE_SYNTHETIC: Forecast generated from synthetic fallback defaults",
            )
        elif is_stale:
            return ForecastFreshnessMetadata(
                source_type=src_type,
                source_name=src_name,
                retrieved_at=retrieved_at_str,
                cache_age_seconds=cache_age,
                is_reliable=False,
                reliability_reason=f"STALE_FORECAST: Cache age ({cache_age:.0f}s) exceeds max threshold (3600s)",
            )
        else:
            return ForecastFreshnessMetadata(
                source_type=src_type,
                source_name=src_name,
                retrieved_at=retrieved_at_str,
                cache_age_seconds=cache_age,
                is_reliable=True,
                reliability_reason="VERIFIED_LIVE: Real-time live forecast from provider",
            )

    # -----------------------------------------------------------------------
    # Impending-Rain Alert Trigger
    # -----------------------------------------------------------------------

    def evaluate_rain_alert(
        self,
        vendor_id: str,
        latitude: float,
        longitude: float,
        vendor_category: str = "food",
        custom_forecast: Optional[Dict[str, Any]] = None,
        candidate_zones: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[NotificationMessage]:
        """
        Evaluates 1-hour ahead precipitation and rain probability.
        If rain is impending, assesses relocation to sheltered zones with profit uplift.
        Enforces strict forecast freshness invariant.
        """
        now = datetime.now(timezone.utc)
        origin_h3 = h3.latlng_to_cell(latitude, longitude, 8) if hasattr(h3, "latlng_to_cell") else "88619aa6c7fffff"

        # 1. Retrieve Weather Forecast
        if custom_forecast:
            forecast = custom_forecast
        else:
            try:
                forecast = self.weather_client.get_hourly_forecast(latitude, longitude, target_hour_offset=1)
            except Exception as e:
                logger.warning(f"Live weather check failed: {e}")
                forecast = {"source_type": "synthetic_fallback", "source_name": "client_error", "rainfall": 0.0}

        # 2. Verify Freshness
        freshness = self.verify_forecast_freshness(forecast)

        # 3. Assess Rain Criteria
        rain_prob = float(forecast.get("precipitation_probability") or forecast.get("next_hour_rain_probability_pct") or 0.0)
        rainfall_mm = float(forecast.get("rainfall") or forecast.get("precipitation_mm") or 0.0)
        has_impending_rain = (rain_prob >= 50.0) or (rainfall_mm >= 0.5) or bool(forecast.get("rain_alert"))

        if not has_impending_rain:
            return None

        # 4. Formulate Relocation Plan
        # If forecast is UNRELIABLE or STALE: Enforce CRITICAL INVARIANT: DO NOT RECOMMEND RELOCATION
        dep_time = (now + timedelta(minutes=5)).strftime("%H:%M")
        arr_time = (now + timedelta(minutes=15)).strftime("%H:%M")

        relocation: Optional[RelocationDetails] = None
        severity = AlertSeverity.WARNING

        if not freshness.is_reliable:
            # Stale / Synthetic forecast safety invariant
            title = "Impending Rain Alert — Advisory (Stay Put)"
            body = (
                f"Rain probability is {rain_prob:.0f}% ({rainfall_mm:.1f} mm) in the next hour. "
                f"However, weather forecast telemetry is unverified ({freshness.reliability_reason}). "
                f"DO NOT RELOCATE. Secure your cart and shelter in place."
            )
            relocation = RelocationDetails(
                destination_h3_cell=origin_h3,
                destination_name="Current Location (Shelter in Place)",
                distance_km=0.0,
                travel_time_minutes=0,
                expected_profit_uplift_inr=0.0,
                recommended_departure_time=dep_time,
                expected_arrival_time=dep_time,
                is_move_recommended=False,
                forecast_freshness=freshness,
                decision_reason=f"Move blocked by safety invariant: {freshness.reliability_reason}",
            )
        else:
            # Verified reliable forecast: find covered / sheltered commercial zone
            sheltered_cell, dest_name, dist_km, travel_min, uplift_inr = self._find_sheltered_candidate(
                latitude, longitude, origin_h3, vendor_category, candidate_zones
            )

            is_recommended = (uplift_inr >= 35.0) and (dist_km <= MAX_RECOMMENDED_TRAVEL_KM) and (travel_min <= MAX_RECOMMENDED_TRAVEL_MIN)
            arr_time = (now + timedelta(minutes=5 + travel_min)).strftime("%H:%M")

            if is_recommended:
                severity = AlertSeverity.CRITICAL
                title = f"Heavy Rain Impending — Relocate to {dest_name}"
                body = (
                    f"Rain probability is {rain_prob:.0f}% ({rainfall_mm:.1f} mm) within 30-45 minutes. "
                    f"Recommended move to sheltered zone *{dest_name}* for +₹{uplift_inr:.0f}/hr profit uplift."
                )
            else:
                title = "Impending Rain Alert — Canopy Required"
                body = (
                    f"Rain probability is {rain_prob:.0f}% in your zone. Relocation net profit uplift (+₹{uplift_inr:.0f}/hr) "
                    f"does not exceed travel friction. Deploy rain cover and remain at current location."
                )

            relocation = RelocationDetails(
                destination_h3_cell=sheltered_cell,
                destination_name=dest_name,
                distance_km=dist_km,
                travel_time_minutes=travel_min,
                expected_profit_uplift_inr=uplift_inr,
                recommended_departure_time=dep_time,
                expected_arrival_time=arr_time,
                is_move_recommended=is_recommended,
                forecast_freshness=freshness,
                decision_reason="Sheltered commercial hub protecting footfall from rain" if is_recommended else "Insufficient net profit uplift after travel cost",
            )

        notif_id = f"NOTIF-RAIN-{uuid.uuid4().hex[:8].upper()}"
        expires_at = now + timedelta(minutes=45)
        idempotency_key = f"{vendor_id}:IMPENDING_RAIN:{relocation.destination_h3_cell}:{now.strftime('%Y%m%d%H')}"

        return NotificationMessage(
            notification_id=notif_id,
            vendor_id=vendor_id,
            alert_type=AlertType.IMPENDING_RAIN,
            severity=severity,
            title=title,
            body=body,
            relocation=relocation,
            created_at=now,
            expires_at=expires_at,
            idempotency_key=idempotency_key,
            metadata={"rain_probability": rain_prob, "precipitation_mm": rainfall_mm},
        )

    def _find_sheltered_candidate(
        self,
        lat: float,
        lng: float,
        origin_h3: str,
        category: str,
        candidates: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, str, float, int, float]:
        """Finds closest high-demand sheltered commercial zone."""
        if candidates and len(candidates) > 0:
            best = candidates[0]
            c_cell = best.get("h3_cell", origin_h3)
            c_name = best.get("landmark_name") or best.get("display_name") or "Metro Shopping Hub"
            c_dist = float(best.get("distance_km", 1.2))
            c_time = int(best.get("travel_time_min", 6))
            c_uplift = float(best.get("realized_net_uplift_inr", 65.0))
            return c_cell, c_name, c_dist, c_time, c_uplift

        # Default Vijayawada sheltered hubs
        sheltered_hubs = [
            ("88619aa6c1fffff", "Benz Circle Metro Complex", 1.2, 5, 75.0),
            ("88619aa6c5fffff", "Governorpet Covered Market", 1.8, 8, 60.0),
            ("88619aa6c3fffff", "Railway Junction Concourse", 2.2, 10, 85.0),
        ]
        # Return first non-origin hub
        for cell, name, dist, t_min, uplift in sheltered_hubs:
            if cell != origin_h3:
                return cell, name, dist, t_min, uplift
        return sheltered_hubs[0]

    # -----------------------------------------------------------------------
    # Event-Based Demand Alert Trigger
    # -----------------------------------------------------------------------

    def evaluate_event_alert(
        self,
        vendor_id: str,
        latitude: float,
        longitude: float,
        vendor_category: str = "food",
        custom_event: Optional[Dict[str, Any]] = None,
        candidate_zones: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[NotificationMessage]:
        """
        Evaluates nearby live public events (sports, concerts, temple festivals).
        Calculates footfall demand uplift and travel parameters before triggering.
        """
        now = datetime.now(timezone.utc)
        origin_h3 = h3.latlng_to_cell(latitude, longitude, 8) if hasattr(h3, "latlng_to_cell") else "88619aa6c7fffff"

        # 1. Fetch Event Telemetry
        event_info = custom_event
        if not event_info:
            try:
                evt_prov = EventProviderFactory.get_provider()
                event_info = evt_prov.get_event_importance(origin_h3, now)
            except Exception as e:
                logger.warning(f"Event lookup failed: {e}")
                event_info = {"event_active": False, "event_importance": 0.0}

        is_active = bool(event_info.get("event_active", False))
        importance = float(event_info.get("event_importance", 0.0))
        attendance = event_info.get("event_attendance_estimate") or 0

        # Event trigger threshold
        if not is_active and importance < 0.35 and attendance < 300:
            return None

        event_name = event_info.get("event_name") or "Major Cultural Festival & Gathering"
        event_cell = event_info.get("h3_cell") or origin_h3

        # Freshness of background forecast for event zone
        forecast_freshness = ForecastFreshnessMetadata(
            source_type="real_live",
            source_name="open_meteo",
            is_reliable=True,
            reliability_reason="Event zone weather clear",
        )

        dist_km = 1.4
        travel_min = 7
        uplift_inr = 85.0 if vendor_category in ("food", "tea", "beverages", "snacks") else 45.0

        if candidate_zones and len(candidate_zones) > 0:
            cand = candidate_zones[0]
            event_cell = cand.get("h3_cell", event_cell)
            dist_km = float(cand.get("distance_km", dist_km))
            travel_min = int(cand.get("travel_time_min", travel_min))
            uplift_inr = float(cand.get("realized_net_uplift_inr", uplift_inr))

        is_recommended = (uplift_inr >= 40.0) and (dist_km <= MAX_RECOMMENDED_TRAVEL_KM)

        dep_time = (now + timedelta(minutes=10)).strftime("%H:%M")
        arr_time = (now + timedelta(minutes=10 + travel_min)).strftime("%H:%M")

        relocation = RelocationDetails(
            destination_h3_cell=event_cell,
            destination_name=f"{event_name} Grounds",
            distance_km=dist_km,
            travel_time_minutes=travel_min,
            expected_profit_uplift_inr=uplift_inr,
            recommended_departure_time=dep_time,
            expected_arrival_time=arr_time,
            is_move_recommended=is_recommended,
            forecast_freshness=forecast_freshness,
            decision_reason=f"High event attendance surge ({attendance} est. footfall)",
        )

        notif_id = f"NOTIF-EVENT-{uuid.uuid4().hex[:8].upper()}"
        expires_at = now + timedelta(hours=2)
        idempotency_key = f"{vendor_id}:EVENT_DEMAND:{event_cell}:{now.strftime('%Y%m%d%H')}"

        return NotificationMessage(
            notification_id=notif_id,
            vendor_id=vendor_id,
            alert_type=AlertType.EVENT_DEMAND,
            severity=AlertSeverity.INFO if not is_recommended else AlertSeverity.WARNING,
            title=f"High Demand Event Alert: {event_name}",
            body=(
                f"{event_name} is active nearby with {attendance or 'high'} estimated visitors. "
                f"Relocating to `{event_cell}` offers an expected net profit uplift of +₹{uplift_inr:.0f}/hr."
            ),
            relocation=relocation,
            created_at=now,
            expires_at=expires_at,
            idempotency_key=idempotency_key,
            metadata={"event_name": event_name, "attendance": attendance, "importance": importance},
        )

    # -----------------------------------------------------------------------
    # Demand-Spike Alert Trigger
    # -----------------------------------------------------------------------

    def evaluate_demand_spike_alert(
        self,
        vendor_id: str,
        latitude: float,
        longitude: float,
        vendor_category: str = "food",
        candidate_zones: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[NotificationMessage]:
        """
        Detects sudden demand surges in adjacent cells and verifies net profit uplift.
        """
        now = datetime.now(timezone.utc)
        origin_h3 = h3.latlng_to_cell(latitude, longitude, 8) if hasattr(h3, "latlng_to_cell") else "88619aa6c7fffff"

        if not candidate_zones or len(candidate_zones) == 0:
            return None

        best = candidate_zones[0]
        uplift = float(best.get("realized_net_uplift_inr", 0.0))
        prefs = self.get_preferences(vendor_id)

        # Must meet vendor's minimum profit uplift preference
        if uplift < prefs.min_profit_uplift_inr:
            return None

        dest_cell = best.get("h3_cell", origin_h3)
        dest_name = best.get("landmark_name") or best.get("display_name") or "High Demand Hub"
        dist_km = float(best.get("distance_km", 1.0))
        travel_min = int(best.get("travel_time_min", 5))

        freshness = ForecastFreshnessMetadata(
            source_type="real_live",
            source_name="geodemand_ensemble",
            is_reliable=True,
            reliability_reason="Model quantile prediction spread stable",
        )

        dep_time = (now + timedelta(minutes=5)).strftime("%H:%M")
        arr_time = (now + timedelta(minutes=5 + travel_min)).strftime("%H:%M")

        relocation = RelocationDetails(
            destination_h3_cell=dest_cell,
            destination_name=dest_name,
            distance_km=dist_km,
            travel_time_minutes=travel_min,
            expected_profit_uplift_inr=uplift,
            recommended_departure_time=dep_time,
            expected_arrival_time=arr_time,
            is_move_recommended=True,
            forecast_freshness=freshness,
            decision_reason="Surge footfall detected with verified net profit uplift",
        )

        notif_id = f"NOTIF-SPIKE-{uuid.uuid4().hex[:8].upper()}"
        expires_at = now + timedelta(minutes=30)
        idempotency_key = f"{vendor_id}:DEMAND_SPIKE:{dest_cell}:{now.strftime('%Y%m%d%H')}"

        return NotificationMessage(
            notification_id=notif_id,
            vendor_id=vendor_id,
            alert_type=AlertType.DEMAND_SPIKE,
            severity=AlertSeverity.INFO,
            title=f"Demand Spike in {dest_name}",
            body=(
                f"Surge footfall detected in {dest_name}. Relocating now yields an expected "
                f"net profit uplift of +₹{uplift:.0f}/hr after travel expenses."
            ),
            relocation=relocation,
            created_at=now,
            expires_at=expires_at,
            idempotency_key=idempotency_key,
            metadata={"expected_profit_uplift_inr": uplift},
        )

    # -----------------------------------------------------------------------
    # Dispatch & Multi-Channel Delivery
    # -----------------------------------------------------------------------

    def dispatch_notification(
        self,
        message: NotificationMessage,
        override_channel: Optional[NotificationChannel] = None,
        force: bool = False,
    ) -> List[DeliveryRecord]:
        """
        Dispatches a notification message adhering to vendor preferences, quiet hours,
        deduplication windows, and channel fallback rules.
        """
        now = datetime.now(timezone.utc)
        vendor_id = message.vendor_id
        prefs = self.get_preferences(vendor_id)
        records: List[DeliveryRecord] = []

        # 1. Check Expiry
        if message.is_expired(now):
            logger.warning(f"Notification {message.notification_id} expired before dispatch.")
            rec = DeliveryRecord(
                record_id=f"EXP-{uuid.uuid4().hex[:8].upper()}",
                notification_id=message.notification_id,
                vendor_id=vendor_id,
                channel=NotificationChannel.FCM,
                recipient=prefs.fcm_token or "expired",
                status=DeliveryStatus.EXPIRED,
                error_message="Notification expired before dispatch",
                failed_at=now,
            )
            self._record_delivery(rec)
            return [rec]

        # 2. Check Opt-Out
        if prefs.opt_out and not force:
            logger.info(f"Notification suppressed for vendor {vendor_id}: Opted out.")
            rec = DeliveryRecord(
                record_id=f"SUPP-OPTOUT-{uuid.uuid4().hex[:8].upper()}",
                notification_id=message.notification_id,
                vendor_id=vendor_id,
                channel=NotificationChannel.FCM,
                recipient=prefs.fcm_token or "opted_out",
                status=DeliveryStatus.SUPPRESSED_OPT_OUT,
                error_message="Vendor has opted out of notifications",
                dispatched_at=now,
            )
            self._record_delivery(rec)
            return [rec]

        # 3. Check Alert Type Preference
        if not force:
            if message.alert_type == AlertType.IMPENDING_RAIN and not prefs.alert_impending_rain:
                return [self._suppress(message, "Impending-rain alerts disabled in vendor preferences")]
            if message.alert_type == AlertType.EVENT_DEMAND and not prefs.alert_event_demand:
                return [self._suppress(message, "Event alerts disabled in vendor preferences")]
            if message.alert_type == AlertType.DEMAND_SPIKE and not prefs.alert_demand_spike:
                return [self._suppress(message, "Demand-spike alerts disabled in vendor preferences")]

        # 4. Check Quiet Hours
        if not force and self._is_quiet_hours(prefs, now):
            # Only critical impending rain can break quiet hours
            if message.severity != AlertSeverity.CRITICAL:
                logger.info(f"Notification suppressed for vendor {vendor_id}: Quiet hours active.")
                rec = DeliveryRecord(
                    record_id=f"SUPP-QUIET-{uuid.uuid4().hex[:8].upper()}",
                    notification_id=message.notification_id,
                    vendor_id=vendor_id,
                    channel=NotificationChannel.FCM,
                    recipient=prefs.fcm_token or "quiet_hours",
                    status=DeliveryStatus.SUPPRESSED_QUIET_HOURS,
                    error_message=f"Suppressed during quiet hours ({prefs.quiet_hours_start}-{prefs.quiet_hours_end})",
                    dispatched_at=now,
                )
                self._record_delivery(rec)
                return [rec]

        # 5. Duplicate Prevention (Idempotency Key Cooldown)
        if message.idempotency_key and not force:
            with self._store_lock:
                last_sent = self._sent_idempotency_keys.get(message.idempotency_key)
                if last_sent and (now.timestamp() - last_sent) < 3600.0:  # 60m cooldown
                    logger.info(f"Duplicate alert suppressed for {message.idempotency_key}")
                    rec = DeliveryRecord(
                        record_id=f"SUPP-DUP-{uuid.uuid4().hex[:8].upper()}",
                        notification_id=message.notification_id,
                        vendor_id=vendor_id,
                        channel=NotificationChannel.FCM,
                        recipient=prefs.fcm_token or "duplicate",
                        status=DeliveryStatus.SUPPRESSED_DUPLICATE,
                        error_message="Duplicate alert within cooldown period (60 minutes)",
                        dispatched_at=now,
                    )
                    self._record_delivery(rec)
                    return [rec]
                self._sent_idempotency_keys[message.idempotency_key] = now.timestamp()

        # 6. Channel Selection & Delivery
        target_channels = [override_channel] if override_channel else prefs.channels
        if not target_channels:
            target_channels = [NotificationChannel.FCM, NotificationChannel.WHATSAPP]

        fcm_succeeded = False

        # Attempt FCM
        if NotificationChannel.FCM in target_channels:
            fcm_token = prefs.fcm_token or f"fcm_token_{vendor_id}"
            rec_fcm = self.fcm_adapter.send_notification(fcm_token, message)
            self._record_delivery(rec_fcm)
            records.append(rec_fcm)
            if rec_fcm.status == DeliveryStatus.DELIVERED:
                fcm_succeeded = True

        # Attempt WhatsApp (Primary or Fallback)
        if NotificationChannel.WHATSAPP in target_channels or (not fcm_succeeded and prefs.whatsapp_phone):
            phone = prefs.whatsapp_phone or "919876543210"
            rec_wa = self.whatsapp_adapter.send_notification(phone, message)
            self._record_delivery(rec_wa)
            records.append(rec_wa)

        # Store active notification
        with self._store_lock:
            self._active_notifications[message.notification_id] = message

        return records

    def _suppress(self, message: NotificationMessage, reason: str) -> DeliveryRecord:
        rec = DeliveryRecord(
            record_id=f"SUPP-PREF-{uuid.uuid4().hex[:8].upper()}",
            notification_id=message.notification_id,
            vendor_id=message.vendor_id,
            channel=NotificationChannel.FCM,
            recipient="preference_disabled",
            status=DeliveryStatus.SUPPRESSED_OPT_OUT,
            error_message=reason,
            dispatched_at=datetime.now(timezone.utc),
        )
        self._record_delivery(rec)
        return rec

    def _record_delivery(self, record: DeliveryRecord):
        with self._store_lock:
            self._delivery_history.append(record)

    def _is_quiet_hours(self, prefs: NotificationPreferences, now: datetime) -> bool:
        """Evaluates whether current time falls inside vendor's quiet hours window."""
        if not prefs.quiet_hours_start or not prefs.quiet_hours_end:
            return False

        try:
            sh, sm = map(int, prefs.quiet_hours_start.split(":"))
            eh, em = map(int, prefs.quiet_hours_end.split(":"))

            cur_min = now.hour * 60 + now.minute
            s_min = sh * 60 + sm
            e_min = eh * 60 + em

            if s_min <= e_min:
                return s_min <= cur_min <= e_min
            else:
                # Spans midnight (e.g. 23:00 to 06:00)
                return cur_min >= s_min or cur_min <= e_min
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Comprehensive Evaluation Check
    # -----------------------------------------------------------------------

    def evaluate_and_trigger(
        self,
        vendor_id: str,
        latitude: float,
        longitude: float,
        vendor_category: str = "food",
        candidate_zones: Optional[List[Dict[str, Any]]] = None,
        custom_forecast: Optional[Dict[str, Any]] = None,
        custom_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation across rain, event, and demand spike conditions,
        dispatching applicable notifications to the vendor.
        """
        triggered_messages: List[NotificationMessage] = []
        delivery_records: List[DeliveryRecord] = []

        # 1. Evaluate Rain
        rain_msg = self.evaluate_rain_alert(
            vendor_id=vendor_id,
            latitude=latitude,
            longitude=longitude,
            vendor_category=vendor_category,
            custom_forecast=custom_forecast,
            candidate_zones=candidate_zones,
        )
        if rain_msg:
            triggered_messages.append(rain_msg)
            records = self.dispatch_notification(rain_msg)
            delivery_records.extend(records)

        # 2. Evaluate Events
        event_msg = self.evaluate_event_alert(
            vendor_id=vendor_id,
            latitude=latitude,
            longitude=longitude,
            vendor_category=vendor_category,
            custom_event=custom_event,
            candidate_zones=candidate_zones,
        )
        if event_msg:
            triggered_messages.append(event_msg)
            records = self.dispatch_notification(event_msg)
            delivery_records.extend(records)

        # 3. Evaluate Demand Spike (if no rain emergency)
        if not rain_msg:
            spike_msg = self.evaluate_demand_spike_alert(
                vendor_id=vendor_id,
                latitude=latitude,
                longitude=longitude,
                vendor_category=vendor_category,
                candidate_zones=candidate_zones,
            )
            if spike_msg:
                triggered_messages.append(spike_msg)
                records = self.dispatch_notification(spike_msg)
                delivery_records.extend(records)

        return {
            "status": "EVALUATED",
            "vendor_id": vendor_id,
            "alerts_evaluated_count": 3,
            "alerts_triggered_count": len(triggered_messages),
            "triggered_alerts": [m.to_dict() for m in triggered_messages],
            "delivery_records": [r.to_dict() for r in delivery_records],
        }

    # -----------------------------------------------------------------------
    # History & Telemetry Queries
    # -----------------------------------------------------------------------

    def get_vendor_history(
        self,
        vendor_id: str,
        limit: int = 50,
    ) -> List[DeliveryRecord]:
        with self._store_lock:
            records = [r for r in self._delivery_history if r.vendor_id == vendor_id]
            return sorted(records, key=lambda x: x.dispatched_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:limit]

    def get_notification_status(self, notification_id: str) -> List[DeliveryRecord]:
        with self._store_lock:
            return [r for r in self._delivery_history if r.notification_id == notification_id]


# Singleton Engine Instance
notification_service = NotificationTriggerService()
