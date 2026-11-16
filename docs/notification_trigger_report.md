# Notification Trigger & Delivery Engine Report — GeoDemand AI

**Author:** Member 3: Backend, Business Economics & Decision Intelligence  
**Branch:** `dev/pujitha`  
**Date:** November 16, 2026  
**Status:** Production-Ready & Verified (22/22 Tests Passing)  

---

## 1. Executive Summary

Mobile street vendors in urban India operate under dynamic operational constraints. External events such as sudden heavy rainfall, nearby public gatherings, and localized footfall demand spikes significantly alter unit economics. However, false or poorly-timed relocation advice can cause unnecessary fuel expenditure, transit opportunity loss, and stock spoilage.

This release introduces the **GeoDemand AI Notification Subsystem**, delivering:
1. **Multi-Condition Alert Evaluation:**
   - **Impending-Rain Alerts:** Detects precipitation probability $\ge 50\%$ or rainfall $\ge 0.5\text{ mm}$ in the next hour, identifies sheltered commercial hubs (metro concourses, covered markets), and calculates realized net profit uplift.
   - **Event-Demand Surge Alerts:** Triggers on high-attendance live public gatherings, cultural festivals, or sporting events with footfall estimates and category affinity scaling.
   - **Demand-Spike Alerts:** Flags sudden footfall surges in adjacent commercial zones that clear the vendor's minimum profit uplift threshold.
2. **Economic Uplift & Relocation Safety Contract:**
   - Explicit destination H3 resolution 8 cell.
   - Destination landmark name.
   - Distance (km) and travel duration (minutes).
   - Expected net profit uplift ($\text{INR/hr}$).
   - Recommended departure time ($t + 5\text{m}$) and expected arrival time ($t + 5\text{m} + t_{\text{travel}}$).
   - **CRITICAL SAFETY INVARIANT:** Guarantees that alerts **never** recommend relocation if weather telemetry or demand forecasts are stale ($> 3600\text{s}$) or synthetic. In such cases, vendors receive a sheltered shelter-in-place advisory with `is_move_recommended = False`.
3. **Multi-Channel Delivery Infrastructure:**
   - **Firebase Cloud Messaging (FCM):** Push notifications formatted with high-priority data payloads and click-actions, validating device tokens and handling error codes (`UNREGISTERED`, `INVALID_ARGUMENT`).
   - **WhatsApp Business API:** Conversational markdown alerts and structured relocation guidance cards via Meta Graph API, formatting E.164 phone numbers and handling delivery tracking.
4. **Vendor Preference & Delivery Controls:**
   - Granular channel toggles (`FCM`, `WHATSAPP`, `TELEGRAM`, `SMS`).
   - Alert type toggles (`alert_impending_rain`, `alert_event_demand`, `alert_demand_spike`).
   - Full opt-out support (`opt_out = True`).
   - Quiet-hours window suppression (`23:00–06:00`), overrideable only by critical emergency rain.
   - Duplicate prevention with 60-minute idempotency cooldown per `(vendor_id, alert_type, destination_cell, hour_bucket)`.
   - Expiry timestamps marking un-dispatched messages `EXPIRED`.
   - Full delivery telemetry and status tracking (`PENDING`, `DISPATCHED`, `DELIVERED`, `FAILED`, `EXPIRED`, `SUPPRESSED_*`).

---

## 2. Architecture & Data Flow

```mermaid
flowchart TD
    Signals["External Signals\n(Weather, Events, ML Predictions)"] --> Engine["NotificationTriggerService"]
    
    subgraph FreshnessGate["Forecast Freshness & Safety Invariant Gate"]
        Engine --> CheckFreshness{"Is Forecast Fresh & Live?\n(cache_age <= 3600s, real_live)"}
        CheckFreshness -->|No: Stale / Synthetic| StayPut["Set is_move_recommended = False\nShelter-in-Place Advisory"]
        CheckFreshness -->|Yes: Verified Live| EvalUplift{"Expected Profit Uplift >= Threshold\n& Travel <= 3km, <= 12min?"}
        EvalUplift -->|Yes| RecommendMove["Set is_move_recommended = True\nTarget Cell, Departure & Arrival Times"]
        EvalUplift -->|No| StayPutCanopy["Set is_move_recommended = False\nDeploy Canopy Advisory"]
    end

    StayPut --> MsgBuilder["NotificationMessage Builder"]
    RecommendMove --> MsgBuilder
    StayPutCanopy --> MsgBuilder

    subgraph PolicyEngine["Vendor Preference & Suppression Engine"]
        MsgBuilder --> OptOut{"Vendor Opted Out?"}
        OptOut -->|Yes| SuppOptOut["DeliveryStatus.SUPPRESSED_OPT_OUT"]
        OptOut -->|No| QuietHours{"Quiet Hours Active?"}
        QuietHours -->|Yes (Non-Critical)| SuppQuiet["DeliveryStatus.SUPPRESSED_QUIET_HOURS"]
        QuietHours -->|No| DupCheck{"Duplicate Alert in 60m?"}
        DupCheck -->|Yes| SuppDup["DeliveryStatus.SUPPRESSED_DUPLICATE"]
        DupCheck -->|No| ExpCheck{"Message Expired?"}
        ExpCheck -->|Yes| SuppExp["DeliveryStatus.EXPIRED"]
        ExpCheck -->|No| Dispatcher["Multi-Channel Dispatcher"]
    end

    Dispatcher --> FCM["FCMAdapter\n(Push Notification)"]
    Dispatcher --> WA["WhatsAppNotificationAdapter\n(Conversational Relocation Card)"]

    FCM --> Log["DeliveryRecord Store & Telemetry History"]
    WA --> Log
```

---

## 3. Component Details

### 3.1 Data Contracts (`src/notifications/models.py`)

- **`AlertType`**: `IMPENDING_RAIN`, `EVENT_DEMAND`, `DEMAND_SPIKE`, `GENERAL_ALERT`.
- **`AlertSeverity`**: `INFO`, `WARNING`, `CRITICAL`.
- **`NotificationChannel`**: `FCM`, `WHATSAPP`, `TELEGRAM`, `SMS`.
- **`DeliveryStatus`**: `PENDING`, `DISPATCHED`, `DELIVERED`, `FAILED`, `EXPIRED`, `SUPPRESSED_OPT_OUT`, `SUPPRESSED_DUPLICATE`, `SUPPRESSED_STALE_DATA`, `SUPPRESSED_QUIET_HOURS`.
- **`ForecastFreshnessMetadata`**: Captures `source_type`, `source_name`, `retrieved_at`, `cache_age_seconds`, and `is_reliable`.
- **`RelocationDetails`**: Structured parameters including `destination_h3_cell`, `destination_name`, `distance_km`, `travel_time_minutes`, `expected_profit_uplift_inr`, `recommended_departure_time`, `expected_arrival_time`, `is_move_recommended`, and `forecast_freshness`.

### 3.2 Safety Invariant Implementation

Street vendors face substantial loss if prompted to relocate based on stale or inaccurate weather data. In [src/notifications/notification_engine.py](file:///d:/GeoDemand/src/notifications/notification_engine.py):

```python
def verify_forecast_freshness(self, weather_data: Dict[str, Any]) -> ForecastFreshnessMetadata:
    is_synthetic = "synthetic" in src_type.lower() or "synthetic" in src_name.lower()
    is_stale = cache_age > 3600.0  # 1 hour max age
    
    if is_synthetic or is_stale:
        return ForecastFreshnessMetadata(
            is_reliable=False,
            reliability_reason="Stale forecast or synthetic fallback",
        )
```

When `is_reliable == False`:
- `relocation.is_move_recommended = False`
- Recommended destination defaults to current origin location.
- Distance and travel time set to 0.
- Alert title is flagged as an advisory warning: *"DO NOT RELOCATE. Weather telemetry unverified — shelter in place."*

### 3.3 Delivery Adapters

1. **FCM Adapter ([src/notifications/fcm_adapter.py](file:///d:/GeoDemand/src/notifications/fcm_adapter.py)):**
   - Supports FCM HTTP v1 / legacy protocols with high-priority data payloads containing serialized relocation parameters.
   - Detects revoked device tokens (`UNREGISTERED`) and logs failure telemetry.
   - Automatically switches between live HTTP dispatch and mock test simulation.

2. **WhatsApp Adapter ([src/notifications/whatsapp_adapter.py](file:///d:/GeoDemand/src/notifications/whatsapp_adapter.py)):**
   - Integrates with Meta WhatsApp Business Cloud API (`/v20.0/{phone_number_id}/messages`).
   - Normalizes 10-digit Indian numbers to E.164 (`91XXXXXXXXXX`).
   - Generates conversational markdown with emoji severity badges, decision verdicts, departure/arrival schedules, and opt-out instructions.

---

## 4. API Endpoints ([src/api/app.py](file:///d:/GeoDemand/src/api/app.py))

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/notifications/preferences` | `POST` | Register/update vendor notification channels, opt-out, and quiet hours. |
| `/v1/notifications/preferences/{vendor_id}` | `GET` | Retrieve active preferences and notification tokens for a vendor. |
| `/v1/notifications/trigger-check` | `POST` | Evaluate real-time rain, event, and surge conditions for a vendor location. |
| `/v1/notifications/dispatch` | `POST` | Operator/system endpoint to directly dispatch an alert message. |
| `/v1/notifications/history/{vendor_id}` | `GET` | Query past notification delivery attempts, statuses, and timestamps. |
| `/v1/notifications/delivery-status/{notification_id}` | `GET` | Track multi-channel delivery records for a specific alert ID. |

---

## 5. Test Suite Verification

The dedicated test suite [tests/test_notifications.py](file:///d:/GeoDemand/tests/test_notifications.py) validates all 22 test specifications:

| Test Case | Description | Result |
| :--- | :--- | :--- |
| `test_forecast_freshness_verified_live` | Verified live forecast marked reliable | **PASSED** |
| `test_forecast_freshness_rejects_synthetic_fallback` | Synthetic weather marked unreliable | **PASSED** |
| `test_forecast_freshness_rejects_stale_cache` | Cache age > 3600s rejected as stale | **PASSED** |
| `test_rain_alert_safety_invariant_blocks_move_on_stale_forecast` | Move recommendation blocked on stale forecast | **PASSED** |
| `test_impending_rain_triggers_sheltered_move_when_reliable` | Verified rain triggers move to sheltered zone | **PASSED** |
| `test_clear_weather_does_not_trigger_rain_alert` | Clear conditions suppress rain alerts | **PASSED** |
| `test_event_demand_alert_trigger` | Event crowd triggers demand surge alert | **PASSED** |
| `test_demand_spike_alert_trigger` | Neighboring footfall surge triggers spike alert | **PASSED** |
| `test_demand_spike_ignored_if_uplift_below_preference` | Sub-threshold profit uplift ignored | **PASSED** |
| `test_fcm_adapter_successful_dispatch` | FCM mock/live dispatch success | **PASSED** |
| `test_fcm_adapter_invalid_token_failure` | Empty FCM token fails gracefully | **PASSED** |
| `test_fcm_adapter_unregistered_token_failure` | Revoked token recorded as UNREGISTERED | **PASSED** |
| `test_whatsapp_adapter_successful_dispatch` | WhatsApp message formatting & send | **PASSED** |
| `test_whatsapp_adapter_invalid_phone_failure` | Malformed phone numbers fail gracefully | **PASSED** |
| `test_notification_suppressed_when_vendor_opts_out` | Opt-out suppresses alert dispatch | **PASSED** |
| `test_notification_suppressed_during_quiet_hours` | Quiet hours suppress non-critical alerts | **PASSED** |
| `test_duplicate_notification_prevention` | Duplicate alert in 60m cooldown suppressed | **PASSED** |
| `test_expired_notification_suppression` | Expired alerts marked EXPIRED | **PASSED** |
| `test_notification_preferences_endpoint` | `POST /v1/notifications/preferences` contract | **PASSED** |
| `test_notification_trigger_check_endpoint` | `POST /v1/notifications/trigger-check` contract | **PASSED** |
| `test_notification_dispatch_endpoint` | `POST /v1/notifications/dispatch` contract | **PASSED** |
| `test_notification_history_endpoint` | `GET /v1/notifications/history/{vendor_id}` contract | **PASSED** |

**Summary: 22 passed in 4.93s.**
Repository total: **549 tests collected, 0 collection errors.**
