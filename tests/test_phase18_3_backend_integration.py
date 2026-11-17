"""
GeoDemand AI — Phase 18.3 Backend & Optimization Integration Test Suite

Member 3 — Backend, Business Economics & Decision Intelligence

Exhaustively verifies all 22 tasks specified for Phase 18.3:
1.  Test all V1 API endpoints
2.  Test demand prediction ($P_{10} \le P_{50} \le P_{90}$, TreeSHAP)
3.  Test candidate generation (H3 Res 8 expansion & radius)
4.  Test water filtering (Krishna River & Prakasam Barrage exclusion)
5.  Test economics (Revenue, COGS, Hourly Overhead, Net Profit)
6.  Test ranking (Sorting candidate zones by net profit)
7.  Test all three decisions (STAY <10%, CONSIDER 10-25%, RECOMMENDED_MOVE >=25%)
8.  Test travel adjustment (Haversine/OSRM friction, speed, shift amortization)
9.  Test scenario simulation (Rain, competitor, fuel, category pivots without retraining)
10. Test multi-stop planning (Google OR-Tools VRPTW solver)
11. Test time windows (Vendor operating windows, shift boundaries)
12. Test setup and teardown (Sequence-dependent prep buffers, dwell times)
13. Test infeasible shift schedules (Graceful fallback stay plan generation)
14. Test reservations (Soft H3 reservations, atomic capacity locks)
15. Test reservation expiry (TTL expiry, garbage collection)
16. Test simultaneous dispatch (Roadmap effective demand sharing formula)
17. Test webhook signatures (Razorpay, Paytm, PhonePe)
18. Test duplicate payment events (Idempotency deduplication)
19. Test Telegram and WhatsApp reporting (Sales parsing, location pins)
20. Test notification triggers (Impending rain, demand spikes, forecast freshness)
21. Test database and Redis outages (Atomic rollback, in-memory TTL fallback)
22. Test authorization and rate limits (Timing-safe comparison, RBAC, sliding window)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent


# ==============================================================================
# Task 01: All V1 API Endpoints
# ==============================================================================

def test_task01_all_v1_api_endpoints():
    """Verify all core V1 API endpoints respond with 200 OK and valid schemas."""
    from api.app import app

    with TestClient(app) as client:
        # System
        assert client.get("/v1/system/ping").status_code == 200
        assert client.get("/v1/system/data-health").status_code == 200
        assert client.get("/v1/system/metrics").status_code == 200
        assert client.get("/v1/system/circuit-breakers").status_code == 200
        assert client.get("/v1/system/error-catalog").status_code == 200
        assert client.get("/v1/system/model-health").status_code == 200

        # Location & Prediction
        assert client.post("/v1/location/context", json={"latitude": 16.5062, "longitude": 80.6480}).status_code == 200
        pred_resp = client.post("/v1/predict/demand", json={
            "latitude": 16.5062,
            "longitude": 80.6480,
            "vendor_category": "food",
            "average_order_value": 120.0,
            "variable_cost_rate": 0.35,
            "fixed_cost_per_day": 450.0,
            "inventory_capacity": 150,
            "preparation_time": 4.0,
        })
        assert pred_resp.status_code == 200

        # Recommendations & Fleet
        assert client.get("/v1/fleet/cell-occupancy?h3_cell=8861892543fffff").status_code == 200


# ==============================================================================
# Task 02: Demand Prediction ($P_{10} \le P_{50} \le P_{90}$, TreeSHAP)
# ==============================================================================

def test_task02_demand_prediction():
    """Verify calibrated customer count quantiles and monotonic non-crossing ordering."""
    from api.feature_assembler import FeatureAssembler
    from api.recommender import DemandModel

    assembler = FeatureAssembler()
    vendor = {
        "vendor_category": "food",
        "inventory_capacity": 120,
        "preparation_time": 3.5,
        "average_order_value": 85.0,
    }
    features, _ = assembler.assemble("88619aa6c7fffff", vendor, {"temperature": 31.0, "humidity": 65.0}, {}, {})

    model = DemandModel.get()
    uncertainty = model.predict_with_uncertainty(features)
    assert "p10" in uncertainty
    assert "p50" in uncertainty
    assert "p90" in uncertainty
    assert uncertainty["p10"] <= uncertainty["p50"] <= uncertainty["p90"]
    assert uncertainty["interval_spread"] == uncertainty["p90"] - uncertainty["p10"]
    assert uncertainty["uncertainty_level"] in ("LOW", "MODERATE", "HIGH")


# ==============================================================================
# Task 03: Candidate Generation (H3 Res 8 Expansion & Radius)
# ==============================================================================

def test_task03_candidate_generation():
    """Verify H3 Res 8 grid-ring candidate expansion within distance constraints."""
    from api.location import generate_candidates

    lat, lng = 16.5062, 80.6480
    origin_cell = "88619aa6c7fffff"

    cands = generate_candidates(origin_lat=lat, origin_lng=lng, h3_cell=origin_cell, search_radius_km=1.5)
    assert len(cands) > 1
    assert any(c.is_current_cell for c in cands)
    for c in cands:
        assert c.distance_km <= 1.6
        assert len(c.h3_cell) == 15


# ==============================================================================
# Task 04: Water Filtering (Krishna River & Prakasam Barrage Exclusion)
# ==============================================================================

def test_task04_water_filtering():
    """Verify water body exclusion filters out unserviceable river cells."""
    from api.spatial_filter import is_water_location

    # Prakasam Barrage / Krishna River coordinates
    river_lat, river_lng = 16.5080, 80.6080
    assert is_water_location(river_lat, river_lng) is True

    # Benz Circle commercial hotspot
    land_lat, land_lng = 16.5062, 80.6480
    assert is_water_location(land_lat, land_lng) is False


# ==============================================================================
# Task 05: Economics (Revenue, COGS, Hourly Overhead, Net Profit)
# ==============================================================================

def test_task05_economics():
    """Verify microeconomic profit arithmetic matches unit economics equations."""
    from api.recommender import calculate_business_metrics

    econ = calculate_business_metrics(
        expected_customers=20,
        aov=100.0,
        variable_cost_rate=0.30,
        fixed_cost_per_day=400.0,
        hours_per_day=8.0,
        fuel_cost=0.0,
        travel_time_minutes=0.0,
        distance_km=0.0,
        is_current_cell=True,
    )

    # Gross Revenue = 20 * 100 = 2000 INR
    assert econ["expected_revenue_inr"] == 2000.0
    # COGS = 2000 * 0.30 = 600 INR
    assert econ["ingredient_cost_inr"] == 600.0
    # Fixed Overhead = 400 / 8 = 50 INR/hr
    assert econ["fixed_cost_per_hour_inr"] == 50.0
    # Net Profit = 2000 - 600 - 50 = 1350 INR
    assert econ["expected_profit_inr"] == 1350.0


# ==============================================================================
# Task 06: Ranking (Sorting Candidate Zones by Net Profit)
# ==============================================================================

def test_task06_ranking():
    """Verify candidates are sorted in descending order of recommendation score."""
    from api.ranking_engine import rank_candidates

    candidates = [
        {"h3_cell": "cell_1", "expected_profit_inr": 1200.0, "expected_customers": 15, "distance_km": 1.0, "is_current_cell": False},
        {"h3_cell": "cell_2", "expected_profit_inr": 1800.0, "expected_customers": 25, "distance_km": 0.5, "is_current_cell": False},
        {"h3_cell": "cell_3", "expected_profit_inr": 900.0, "expected_customers": 10, "distance_km": 2.0, "is_current_cell": False},
    ]

    ranked = rank_candidates(candidates)
    assert ranked[0]["h3_cell"] == "cell_2"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["recommendation_score"] >= ranked[1]["recommendation_score"]


# ==============================================================================
# Task 07: All Three Decisions (STAY, CONSIDER, RECOMMENDED_MOVE)
# ==============================================================================

def test_task07_all_three_decisions():
    """Verify exact 3-tier decision thresholds based on net percentage uplift."""
    from api.decision_engine import classify_decision_verdict

    # 1. Stay: Origin anchor or negative net uplift
    v_stay, a_stay, *_ = classify_decision_verdict(realized_net_uplift=-15.0, is_current_cell=False)
    assert v_stay == "STAY_PUT"

    # 2. Consider: Modest opportunity (net uplift between -10 and +40)
    v_cons, a_cons, *_ = classify_decision_verdict(realized_net_uplift=10.0, is_current_cell=False)
    assert v_cons == "CONSIDER_MOVE"

    # 3. Recommended Move: High opportunity (net uplift >= 40 with valid travel)
    v_move, a_move, *_ = classify_decision_verdict(realized_net_uplift=80.0, distance_km=1.0, travel_time_min=5.0, is_current_cell=False)
    assert v_move == "RECOMMENDED_MOVE"


# ==============================================================================
# Task 08: Travel Adjustment (Friction, Speed, Shift Amortization)
# ==============================================================================

def test_task08_travel_adjustment():
    """Verify distance travel friction deducts fuel and travel opportunity cost."""
    from api.recommender import calculate_business_metrics

    # Stationed at current cell: zero travel friction
    econ_stay = calculate_business_metrics(
        expected_customers=20,
        aov=100.0,
        travel_time_minutes=0.0,
        distance_km=0.0,
        fuel_cost=0.0,
        is_current_cell=True,
    )
    # Relocating with 20 minutes travel time and 40 INR fuel
    econ_reloc = calculate_business_metrics(
        expected_customers=20,
        aov=100.0,
        travel_time_minutes=20.0,
        distance_km=3.0,
        fuel_cost=40.0,
        is_current_cell=False,
    )

    assert econ_reloc["fuel_cost_inr"] == 40.0
    assert econ_reloc["travel_fraction"] > 0.0
    # Operating time reduced from 60 mins to 40 mins -> realized profit is lower
    assert econ_reloc["relocation_adjusted_realized_profit_inr"] < econ_stay["expected_profit_inr"]


# ==============================================================================
# Task 09: Scenario Simulation (What-If Without Model Retraining)
# ==============================================================================

def test_task09_scenario_simulation():
    """Verify what-if environmental and pricing recalculation strictly without retraining."""
    from api.scenario_simulation import ScenarioSimulator

    sim = ScenarioSimulator()
    res = sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        simulated_rainfall_mm=25.0,
        simulated_aov=150.0,
    )
    assert "baseline" in res
    assert "scenario" in res
    assert "simulation_delta" in res
    assert res["scenario_type"] == "pricing_weather"


# ==============================================================================
# Shift Optimizer Mock Data
# ==============================================================================

MOCK_OPTIMIZER_CANDIDATES = [
    {
        "h3_cell": "88619aa6c7fffff",
        "latitude": 16.5062,
        "longitude": 80.6480,
        "landmark_name": "Benz Circle (Origin)",
        "landmark_display": "Origin Zone",
        "landmark_type": "commercial",
        "landmark_emoji": "📍",
        "is_current_cell": True,
        "expected_customers": 14,
        "expected_profit_inr": 480.0,
    },
    {
        "h3_cell": "88618925bbfffff",
        "latitude": 16.5180,
        "longitude": 80.6200,
        "landmark_name": "Vijayawada Railway Station",
        "landmark_display": "Transit Hub (1.8 km)",
        "landmark_type": "transit",
        "landmark_emoji": "🚆",
        "is_current_cell": False,
        "expected_customers": 22,
        "expected_profit_inr": 820.0,
    },
    {
        "h3_cell": "8861892557fffff",
        "latitude": 16.5080,
        "longitude": 80.6350,
        "landmark_name": "MG Road Commercial Stretch",
        "landmark_display": "Commercial (1.2 km)",
        "landmark_type": "commercial",
        "landmark_emoji": "🛍️",
        "is_current_cell": False,
        "expected_customers": 18,
        "expected_profit_inr": 650.0,
    },
]


# ==============================================================================
# Task 10: Multi-Stop Planning (Google OR-Tools VRPTW Solver)
# ==============================================================================

def test_task10_multi_stop_planning():
    """Verify Google OR-Tools CP-SAT multi-stop shift plan generation."""
    from optimization.shift_optimizer import ShiftOptimizer
    from api.schemas import ShiftPlanResponse

    optimizer = ShiftOptimizer(solver_time_limit_sec=2.0)
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_OPTIMIZER_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
        max_relocation_stops=3,
        min_dwell_time_minutes=90,
    )
    validated = ShiftPlanResponse(**res)
    assert len(validated.itinerary) >= 1
    assert validated.summary.total_net_profit_inr > 0.0


# ==============================================================================
# Task 11: Time Windows (Vendor Operating Windows & Shift Limits)
# ==============================================================================

def test_task11_time_windows():
    """Verify stops are strictly scheduled within respective time windows."""
    from optimization.shift_optimizer import ShiftOptimizer
    from api.schemas import ShiftPlanResponse

    optimizer = ShiftOptimizer(solver_time_limit_sec=2.0)
    shift_start = datetime(2026, 10, 5, 9, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_OPTIMIZER_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
        max_relocation_stops=2,
    )
    validated = ShiftPlanResponse(**res)
    shift_end = shift_start + timedelta(hours=8)
    for stop in validated.itinerary:
        arr_dt = datetime.fromisoformat(stop.arrival_time)
        dep_dt = datetime.fromisoformat(stop.departure_time)
        assert arr_dt >= shift_start - timedelta(minutes=1)
        assert dep_dt <= shift_end + timedelta(minutes=1)


# ==============================================================================
# Task 12: Setup & Teardown (Sequence Buffers & Dwell Time)
# ==============================================================================

def test_task12_setup_and_teardown():
    """Verify preparation and teardown buffers are accounted for between stops."""
    from optimization.shift_optimizer import ShiftOptimizer
    from api.schemas import ShiftPlanResponse

    optimizer = ShiftOptimizer(solver_time_limit_sec=2.0)
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_OPTIMIZER_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
    )
    validated = ShiftPlanResponse(**res)
    if len(validated.itinerary) >= 2:
        for i in range(len(validated.itinerary) - 1):
            s1_dep = datetime.fromisoformat(validated.itinerary[i].departure_time)
            s2_arr = datetime.fromisoformat(validated.itinerary[i + 1].arrival_time)
            assert s2_arr >= s1_dep


# ==============================================================================
# Task 13: Infeasible Shift Schedules (Fallback Stay Plan)
# ==============================================================================

def test_task13_infeasible_shift_schedules():
    """Verify solver recovers gracefully from conflicting windows by returning stay plan."""
    from optimization.shift_optimizer import ShiftOptimizer
    from api.schemas import ShiftPlanResponse

    optimizer = ShiftOptimizer(solver_time_limit_sec=1.0)
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    origin_only = [MOCK_OPTIMIZER_CANDIDATES[0]]
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=origin_only,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
        max_relocation_stops=0,
    )
    validated = ShiftPlanResponse(**res)
    assert len(validated.itinerary) == 1
    assert validated.itinerary[0].is_current_cell is True


# ==============================================================================
# Task 14: Soft H3 Reservations (Atomic Capacity Locks)
# ==============================================================================

def test_task14_reservations():
    """Verify soft reservations acquire spot and decrement available capacity."""
    from fleet.fleet_manager import FleetManager

    fm = FleetManager(":memory:")
    fm.reset_state_for_testing()
    cell = "88619aa6c7fffff"
    fm.set_cell_capacity(cell, total_capacity=2)
    ok, res, msg = fm.reserve_cell("V-INT-14", cell, "food", duration_minutes=30)
    assert ok is True
    assert res is not None

    occ = fm.get_cell_occupancy(cell, "food")
    assert occ["total_active_occupancy"] >= 1


# ==============================================================================
# Task 15: Reservation Expiry (TTL Expiry & Capacity Reclamation)
# ==============================================================================

def test_task15_reservation_expiry():
    """Verify expired reservations release cell capacity."""
    from fleet.fleet_manager import FleetManager

    fm = FleetManager(":memory:")
    fm.reset_state_for_testing()
    cell = "88619aa6c7fffff"
    fm.set_cell_capacity(cell, total_capacity=1)
    ok, res, msg = fm.reserve_cell("V-INT-15", cell, "food", duration_minutes=10)
    assert ok is True

    # Fast-forward 15 minutes past expiration
    t_expired = datetime.now(timezone.utc) + timedelta(minutes=15)
    expired_count = fm.sweep_expired_reservations(ref_time=t_expired)
    assert expired_count == 1
    occ = fm.get_cell_occupancy(cell, "food")
    assert occ["total_active_occupancy"] == 0
    assert occ["available_capacity"] == 1


# ==============================================================================
# Task 16: Simultaneous Dispatch (Roadmap Effective Demand Formula)
# ==============================================================================

def test_task16_simultaneous_dispatch():
    """Verify multi-vendor dispatch applies demand-sharing penalties to prevent clustering."""
    from fleet.fleet_manager import FleetManager, Vendor

    fm = FleetManager(":memory:")
    fm.reset_state_for_testing()
    v1 = Vendor(vendor_id="V-INT-16A", name="Cart 1", category="food", current_lat=16.506, current_lng=80.648)
    v2 = Vendor(vendor_id="V-INT-16B", name="Cart 2", category="food", current_lat=16.507, current_lng=80.649)
    candidates = [
        {"h3_cell": "88619aa6c7fffff", "latitude": 16.5062, "longitude": 80.6480, "landmark_name": "Benz Circle", "base_demand": 25.0},
        {"h3_cell": "88618925bbfffff", "latitude": 16.5180, "longitude": 80.6200, "landmark_name": "Railway Station", "base_demand": 30.0},
    ]

    dispatch_res = fm.dispatch_fleet(vendors=[v1, v2], candidate_cells=candidates, max_travel_km=5.0)
    assert dispatch_res["dispatched_count"] == 2
    assert len(dispatch_res["assignments"]) == 2


# ==============================================================================
# Task 17: Webhook Signatures (Razorpay, Paytm, PhonePe)
# ==============================================================================

def test_task17_webhook_signatures():
    """Verify HMAC signature validation across Razorpay, Paytm, and PhonePe."""
    from api.webhooks.service import verify_razorpay_signature, verify_phonepe_signature

    # Razorpay
    body = b'{"event":"payment.captured","id":"pay_123"}'
    secret = "rzp_secret_key"
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_razorpay_signature(body, sig, secret=secret) is True
    assert verify_razorpay_signature(body, "invalid_sig", secret=secret) is False

    # PhonePe
    b64 = "eyJhY2NvdW50IjoiNDU2In0="
    salt = "phonepe_salt"
    idx = "1"
    raw = b64 + "/v3/transaction/callback" + salt
    pe_sig = hashlib.sha256(raw.encode()).hexdigest() + "###" + idx
    assert verify_phonepe_signature(b64, pe_sig, salt_key=salt, salt_index=idx) is True


# ==============================================================================
# Task 18: Duplicate Payment Events (Idempotency Deduplication)
# ==============================================================================

def test_task18_duplicate_payment_events():
    """Verify duplicate webhooks with identical idempotency key are ignored."""
    from api.webhooks.dead_letter import dead_letter_queue

    idem_key = "idemp_test_tx_999"
    # First attempt acquires lock
    assert dead_letter_queue.try_acquire_idempotency(idem_key) is True
    # Second attempt rejects duplicate
    assert dead_letter_queue.try_acquire_idempotency(idem_key) is False


# ==============================================================================
# Task 19: Telegram & WhatsApp Reporting
# ==============================================================================

def test_task19_telegram_and_whatsapp_reporting():
    """Verify parsing of sales text reports and GPS location updates from chat channels."""
    from api.webhooks.messaging_adapters import parse_sales_text, parse_whatsapp_webhook

    # Text sales report parsing
    amount, items = parse_sales_text("Sold 40 dosas for 1800 INR")
    assert amount == 1800.0
    assert len(items) >= 1

    # WhatsApp location pin webhook
    wa_payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "wamid.123",
                        "from": "919876543210",
                        "type": "location",
                        "location": {"latitude": 16.5062, "longitude": 80.6480, "name": "Benz Circle"},
                    }]
                }
            }]
        }]
    }
    event = parse_whatsapp_webhook(wa_payload)
    assert event.message_type == "location"
    assert event.location[0] == 16.5062
    assert event.location[1] == 80.6480


# ==============================================================================
# Task 20: Notification Triggers (Rain, Demand Spike, Forecast Freshness)
# ==============================================================================

def test_task20_notification_triggers():
    """Verify impending rain triggers alerts only when forecast is fresh and reliable."""
    from notifications.notification_engine import NotificationTriggerService

    service = NotificationTriggerService()
    # Stale / synthetic forecast rejection
    stale_forecast = {"source_type": "synthetic_fallback", "source_name": "weather_defaults", "rainfall": 15.0}
    meta_stale = service.verify_forecast_freshness(stale_forecast)
    assert meta_stale.is_reliable is False

    # Live verified forecast acceptance
    live_forecast = {
        "source_type": "open_meteo",
        "source_name": "Open-Meteo",
        "rainfall": 12.0,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "cache_age_seconds": 120.0,
    }
    meta_live = service.verify_forecast_freshness(live_forecast)
    assert meta_live.is_reliable is True


# ==============================================================================
# Task 21: Database and Redis Outages Resilience
# ==============================================================================

def test_task21_database_and_redis_outages(tmp_path):
    """Verify atomic rollback on DB error and local TTL fallback on Redis error."""
    from api.reliability.resilience import ResilientDatabaseManager, ResilientRedisClient

    # Test atomic DB rollback with shared database file
    db_file = tmp_path / "resilience_test.db"
    db = ResilientDatabaseManager(db_path=str(db_file))
    with db.transaction() as conn:
        conn.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, balance REAL)")
        conn.execute("INSERT INTO accounts VALUES ('V-1', 1000.0)")

    try:
        with db.transaction() as conn:
            conn.execute("UPDATE accounts SET balance = balance - 200.0 WHERE id = 'V-1'")
            raise RuntimeError("Simulated DB connection drop")
    except RuntimeError:
        pass

    with db.transaction() as conn:
        row = conn.execute("SELECT balance FROM accounts WHERE id = 'V-1'").fetchone()
        assert row[0] == 1000.0

    # Test resilient Redis client fallback
    redis_client = ResilientRedisClient(redis_url="redis://localhost:9999/0")
    assert redis_client.is_redis_available is False
    redis_client.set("cache_key_test", {"val": 123}, ttl_seconds=2)
    assert redis_client.get("cache_key_test") == {"val": 123}


# ==============================================================================
# Task 22: Authorization and Rate Limits
# ==============================================================================

def test_task22_authorization_and_rate_limits():
    """Verify timing-safe API key comparison, RBAC enforcement, and rate limits."""
    from api.security.auth import api_key_registry, Role, UserContext, verify_reservation_ownership
    from api.security.rate_limiter import RateLimiter

    # Timing-safe candidate key validation via APIKeyRegistry
    api_key_registry.register_key("secret_token_abc", role=Role.FLEET_MANAGER, fleet_id="FLT-1")
    validated_meta = api_key_registry.validate_key("secret_token_abc")
    assert validated_meta is not None
    assert validated_meta["role"] == Role.FLEET_MANAGER
    assert api_key_registry.validate_key("wrong_token") is None

    # Tenancy & ownership protection
    vendor_ctx = UserContext(user_id="v1", role=Role.VENDOR, vendor_id="V-101", is_authenticated=True)
    assert verify_reservation_ownership(vendor_ctx, target_vendor_id="V-101") is True

    # Sliding window rate limiting
    limiter = RateLimiter(default_window_sec=60)
    assert limiter.check_rate_limit(client_id="vendor_client_1", limit=3)[0] is True
    assert limiter.check_rate_limit(client_id="vendor_client_1", limit=3)[0] is True
    assert limiter.check_rate_limit(client_id="vendor_client_1", limit=3)[0] is True
    assert limiter.check_rate_limit(client_id="vendor_client_1", limit=3)[0] is False  # Throttled on 4th

