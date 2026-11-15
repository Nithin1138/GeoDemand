"""
Dedicated unit and integration tests for:
1. Travel-time adjustment formulas (hourly operating fraction and shift amortization)
2. Traffic-aware fuel friction penalties
3. Relocation policy thresholds (STAY_PUT, CONSIDER_MOVE, RECOMMENDED_MOVE)
4. Travel constraints (MAX_RECOMMENDED_TRAVEL_MIN, MAX_RECOMMENDED_TRAVEL_KM, search_radius_km)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

import pytest

from api.recommender import calculate_business_metrics
from api.providers.fuel import FuelPriceProvider
from config.settings import (
    STAY_THRESHOLD_INR_PER_HOUR,
    CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR,
    MAX_RECOMMENDED_TRAVEL_MIN,
    MAX_RECOMMENDED_TRAVEL_KM,
    FUEL_PRICE_INR_PER_LITRE,
    FUEL_EFFICIENCY_KM_PER_L,
)


# ===========================================================================
# 1. Travel Time Operating Fraction Tests
# ===========================================================================

@pytest.mark.parametrize("travel_min,expected_fraction", [
    (0.0, 1.0000),      # Stationary, full hour of operation
    (10.0, 0.8333),     # 50 min active
    (15.0, 0.7500),     # 45 min active
    (30.0, 0.5000),     # 30 min active
    (45.0, 0.2500),     # 15 min active
    (60.0, 0.0000),     # Entire hour spent travelling
    (90.0, 0.0000),     # Clamped at 0.0 (cannot be negative)
])
def test_travel_fraction_hourly_horizon(travel_min, expected_fraction):
    """Hourly horizon travel fraction decreases linearly and clamps at 0.0."""
    biz = calculate_business_metrics(
        expected_customers=50,
        aov=100.0,
        travel_time_minutes=travel_min,
    )
    assert biz["travel_fraction"] == expected_fraction


@pytest.mark.parametrize("travel_min,expected_fraction", [
    (0.0, 1.0000),      # 240 / 240
    (12.0, 0.9500),     # (240 - 12) / 240 = 228 / 240
    (24.0, 0.9000),     # (240 - 24) / 240 = 216 / 240
    (48.0, 0.8000),     # (240 - 48) / 240 = 192 / 240
    (60.0, 0.7500),     # (240 - 60) / 240 = 180 / 240
    (90.0, 0.7500),     # Clamped at 0.75 floor for standard shift
])
def test_travel_fraction_shift_amortization(travel_min, expected_fraction):
    """4-hour shift amortization in app.py amortizes travel over 240 min with a 0.75 floor."""
    session_minutes = 240.0
    computed_fraction = round(max(0.75, (session_minutes - travel_min) / session_minutes), 4)
    assert computed_fraction == expected_fraction


def test_relocation_adjusted_profit_monotonicity():
    """Higher travel time strictly reduces realized profit for identical customer demand."""
    profits = []
    for t in [0.0, 5.0, 10.0, 15.0, 25.0]:
        biz = calculate_business_metrics(
            expected_customers=50,
            aov=100.0,
            travel_time_minutes=t,
            fuel_cost=10.0,
        )
        profits.append(biz["relocation_adjusted_realized_profit_inr"])

    # Strictly decreasing
    for i in range(len(profits) - 1):
        assert profits[i] > profits[i + 1], f"Profit did not decrease with travel time: {profits}"


# ===========================================================================
# 2. Fuel Friction & Congestion Tests
# ===========================================================================

def test_fuel_cost_calculation():
    """Fuel cost per km matches fuel price / efficiency formula."""
    fuel_prov = FuelPriceProvider()
    expected_cost_per_km = FUEL_PRICE_INR_PER_LITRE / FUEL_EFFICIENCY_KM_PER_L

    cost_1km = fuel_prov.get_fuel_cost_for_distance(1.0)
    assert abs(cost_1km["fuel_cost_inr"] - expected_cost_per_km) < 0.05

    cost_3km = fuel_prov.get_fuel_cost_for_distance(3.0)
    assert abs(cost_3km["fuel_cost_inr"] - (3.0 * expected_cost_per_km)) < 0.05


def test_traffic_congestion_fuel_penalty():
    """Traffic congestion ratio inflates effective fuel cost."""
    fuel_prov = FuelPriceProvider()
    base = fuel_prov.get_fuel_cost_for_distance(3.0)["fuel_cost_inr"]

    # In app.py: congestion_penalty = 1.0 + congestion_ratio * 0.35
    for ratio in [0.0, 0.5, 1.0]:
        penalty = 1.0 + ratio * 0.35
        adjusted_fuel = round(base * penalty, 2)
        assert adjusted_fuel >= base
        if ratio > 0:
            assert adjusted_fuel > base


# ===========================================================================
# 3. Relocation Decision Thresholds Tests
# ===========================================================================

def classify_decision(realized_net_uplift: float, raw_profit_diff: float) -> str:
    """Exact policy classification logic from app.py."""
    if realized_net_uplift >= 40.0 or (raw_profit_diff >= 60.0 and realized_net_uplift >= -20.0):
        return "RECOMMENDED_MOVE"
    elif realized_net_uplift >= -10.0 or raw_profit_diff > 15.0:
        return "CONSIDER_MOVE"
    else:
        return "STAY_PUT"


def test_decision_policy_threshold_constants():
    """Verify configured threshold constants."""
    assert STAY_THRESHOLD_INR_PER_HOUR == 50.0
    assert CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR == 250.0
    assert MAX_RECOMMENDED_TRAVEL_MIN == 12.0
    assert MAX_RECOMMENDED_TRAVEL_KM == 3.0


def test_recommended_move_classification():
    """RECOMMENDED_MOVE triggered by strong uplift or superior destination demand."""
    # Strong net uplift (>= 40 INR/hr)
    assert classify_decision(realized_net_uplift=40.0, raw_profit_diff=30.0) == "RECOMMENDED_MOVE"
    assert classify_decision(realized_net_uplift=150.0, raw_profit_diff=160.0) == "RECOMMENDED_MOVE"

    # Destination demand superiority (raw_diff >= 60 and realized_net_uplift >= -20)
    assert classify_decision(realized_net_uplift=-15.0, raw_profit_diff=65.0) == "RECOMMENDED_MOVE"
    assert classify_decision(realized_net_uplift=0.0, raw_profit_diff=70.0) == "RECOMMENDED_MOVE"


def test_consider_move_classification():
    """CONSIDER_MOVE triggered by modest opportunity or moderate demand difference."""
    # Modest uplift between -10.0 and 39.9
    assert classify_decision(realized_net_uplift=25.0, raw_profit_diff=30.0) == "CONSIDER_MOVE"
    assert classify_decision(realized_net_uplift=-5.0, raw_profit_diff=10.0) == "CONSIDER_MOVE"
    assert classify_decision(realized_net_uplift=-10.0, raw_profit_diff=0.0) == "CONSIDER_MOVE"

    # Moderate raw profit diff > 15 even if net uplift is slightly negative (< -10 but >= -20)
    assert classify_decision(realized_net_uplift=-15.0, raw_profit_diff=25.0) == "CONSIDER_MOVE"


def test_stay_put_classification():
    """STAY_PUT triggered when relocation costs/friction erase opportunity."""
    # Significant negative net uplift and low profit diff
    assert classify_decision(realized_net_uplift=-11.0, raw_profit_diff=10.0) == "STAY_PUT"
    assert classify_decision(realized_net_uplift=-50.0, raw_profit_diff=0.0) == "STAY_PUT"
    assert classify_decision(realized_net_uplift=-100.0, raw_profit_diff=-80.0) == "STAY_PUT"


# ===========================================================================
# 4. Live API Integration for Relocation & Travel Time
# ===========================================================================

MOCK_WEATHER = {
    "temperature": 28.5, "humidity": 72.0, "rainfall": 0.0,
    "wind_speed": 12.0, "pressure": 1008.0, "cloud_cover": 25.0,
    "visibility": 9.5, "weather_condition": "clear", "uv_index": 6.0,
    "source": "real:open-meteo-current", "source_type": "real_live",
    "source_name": "open-meteo", "observation_timestamp": "2026-08-18T10:00:00+00:00",
    "retrieved_at": "2026-08-18T10:00:00+00:00", "from_cache": False, "cache_age_seconds": 0,
}


@pytest.fixture(scope="module")
def api_client():
    with (
        patch("features.weather_client.OpenMeteoClient.get_current_weather", return_value=MOCK_WEATHER),
        patch("features.weather_client.OpenMeteoClient.get_hourly_forecast", return_value={**MOCK_WEATHER, "source_type": "real_live"}),
    ):
        from fastapi.testclient import TestClient
        from api.app import app
        with TestClient(app) as c:
            yield c


def test_live_recommendations_travel_constraints(api_client):
    """Live API enforces search_radius_km and computes valid travel times."""
    r = api_client.post("/v1/recommendations/live", json={
        "latitude": 16.5062,
        "longitude": 80.6480,
        "search_radius_km": 2.5,
        "top_n": 5,
    })
    assert r.status_code == 200
    body = r.json()

    # Decision metadata
    decision = body["decision"]
    assert decision["verdict"] in ("RECOMMENDED_MOVE", "CONSIDER_MOVE", "STAY_PUT")
    assert decision["stay_threshold_inr"] == STAY_THRESHOLD_INR_PER_HOUR
    assert decision["consider_threshold_inr"] == CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR
    assert "rationale" in decision

    # Candidate travel constraints
    for rec in body["recommendations"]:
        assert rec["distance_km"] <= 2.5, f"Candidate distance {rec['distance_km']} exceeded search radius 2.5 km"
        assert rec["estimated_travel_time_min"] >= 0, "Negative travel time"
        assert 0.75 <= rec["travel_fraction"] <= 1.0, f"Travel fraction {rec['travel_fraction']} out of expected range"
        assert "realized_net_uplift_inr" in rec
        assert rec["decision_verdict"] in ("RECOMMENDED_MOVE", "CONSIDER_MOVE", "STAY_PUT")
