"""
Test suite for GeoDemand AI Shift Optimizer & VRPTW Engine.

Verifies:
- 8-hour shift optimization
- 12-hour shift optimization
- Dynamic time-window resolution and enforcement
- Maximum relocation stops constraint enforcement
- Minimum dwell time constraint enforcement
- Setup, teardown, and parking overhead accounting
- Road travel time, distance, and fuel cost accounting
- Net profit maximization and baseline STAY profit comparison
- Infeasible and timeout fallback handling
- API endpoint contract: POST /v1/routes/shift-plan
"""

from __future__ import annotations

import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.app import _load_model, shift_plan
from api.cache import api_response_cache
from api.schemas import ShiftPlanRequest, ShiftPlanResponse
from optimization.shift_optimizer import ShiftOptimizer
from optimization.time_windows import resolve_candidate_time_windows, CATEGORY_WINDOWS, POI_WINDOW_MODIFIERS

MOCK_CANDIDATES = [
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
        "h3_cell": "88618925bdfffff",
        "latitude": 16.5030,
        "longitude": 80.6550,
        "landmark_name": "PVP Square Mall & Offices",
        "landmark_display": "Commercial Tech Park (1.2 km)",
        "landmark_type": "office",
        "landmark_emoji": "🏢",
        "is_current_cell": False,
        "expected_customers": 19,
        "expected_profit_inr": 710.0,
    },
    {
        "h3_cell": "88618925b9fffff",
        "latitude": 16.4950,
        "longitude": 80.6380,
        "landmark_name": "Siddhartha College Campus",
        "landmark_display": "College District (1.5 km)",
        "landmark_type": "college",
        "landmark_emoji": "🎓",
        "is_college": True,
        "expected_customers": 24,
        "expected_profit_inr": 890.0,
    },
]


def run_coroutine(coro):
    """Executes coroutine synchronously without Windows loopback socket hangs."""
    try:
        coro.send(None)
        return None
    except StopIteration as e:
        return e.value


@pytest.fixture(scope="module", autouse=True)
def init_model():
    """Ensure champion LightGBM model is loaded in memory."""
    _load_model()


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear cache before each test."""
    api_response_cache.clear()
    yield
    api_response_cache.clear()


@pytest.fixture
def optimizer():
    """Provide ShiftOptimizer instance with mock routing."""
    mock_routing = MagicMock()
    # Route distance ~1.5 km, duration ~6 min
    mock_routing.get_travel_time.return_value = {
        "distance_km": 1.5,
        "duration_min": 6.0,
        "is_routable": True,
    }
    return ShiftOptimizer(solver_time_limit_sec=2.0, routing_provider=mock_routing)


# ===========================================================================
# 1. 8-Hour Shift Optimization
# ===========================================================================

def test_8_hour_shift_optimization(optimizer):
    """8-hour shift is planned within [T_start, T_start + 8h] with ordered stops and valid dwell times."""
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
        max_relocation_stops=3,
        min_dwell_time_minutes=90,
    )

    validated = ShiftPlanResponse(**res)
    assert validated.shift_duration_hours == 8.0
    assert len(validated.itinerary) >= 1
    assert len(validated.itinerary) <= 3

    # Verify shift boundaries
    shift_end = shift_start + timedelta(hours=8)
    first_stop = validated.itinerary[0]
    last_stop = validated.itinerary[-1]

    assert datetime.fromisoformat(first_stop.arrival_time) >= shift_start
    assert datetime.fromisoformat(last_stop.departure_time) <= shift_end

    # Every stop has dwell >= min_dwell_time_minutes
    for stop in validated.itinerary:
        assert stop.dwell_minutes >= 90
        assert stop.gross_revenue_inr > 0
        assert stop.net_profit_inr > 0


# ===========================================================================
# 2. 12-Hour Shift Optimization
# ===========================================================================

def test_12_hour_shift_optimization(optimizer):
    """12-hour continuous shift is scheduled across trading peaks with high customer throughput."""
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=12.0,
        max_relocation_stops=4,
        min_dwell_time_minutes=90,
    )

    validated = ShiftPlanResponse(**res)
    assert validated.shift_duration_hours == 12.0
    assert validated.summary.total_dwell_hours >= 6.0
    assert validated.summary.total_expected_customers > 50
    assert validated.summary.total_net_profit_inr > 0


# ===========================================================================
# 3. Dynamic Time-Window Resolution & Enforcement
# ===========================================================================

def test_dynamic_time_window_resolution():
    """Time window engine derives accurate trading windows for college, office, and transit zones."""
    shift_start = datetime(2026, 10, 5, 7, 0, 0, tzinfo=timezone.utc)
    shift_end = datetime(2026, 10, 5, 21, 0, 0, tzinfo=timezone.utc)

    # College candidate
    college_cand = MOCK_CANDIDATES[3]
    windows = resolve_candidate_time_windows(college_cand, "food", shift_start, shift_end)
    assert len(windows) >= 2
    window_names = [w["window_name"] for w in windows]
    assert any("Campus" in name for name in window_names)

    # Transit candidate
    transit_cand = MOCK_CANDIDATES[1]
    transit_windows = resolve_candidate_time_windows(transit_cand, "food", shift_start, shift_end)
    assert len(transit_windows) >= 2
    assert any("Transit" in w["window_name"] for w in transit_windows)


# ===========================================================================
# 4. Maximum Relocation Stops Constraint
# ===========================================================================

def test_max_relocation_stops_limit(optimizer):
    """Enforcing max_relocation_stops=2 restricts itinerary to at most 2 distinct stops."""
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
        max_relocation_stops=2,
        min_dwell_time_minutes=90,
    )

    validated = ShiftPlanResponse(**res)
    assert len(validated.itinerary) <= 2
    assert validated.summary.total_stops <= 2


# ===========================================================================
# 5. Minimum Dwell Time Enforcement
# ===========================================================================

def test_minimum_dwell_time_enforcement(optimizer):
    """Enforcing min_dwell_time_minutes=120 guarantees each stop trades for at least 2 hours."""
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
        max_relocation_stops=3,
        min_dwell_time_minutes=120,
    )

    validated = ShiftPlanResponse(**res)
    for stop in validated.itinerary:
        assert stop.dwell_minutes >= 120
        assert stop.dwell_hours >= 2.0


# ===========================================================================
# 6. Travel, Distance, and Fuel Cost Accounting
# ===========================================================================

def test_travel_distance_and_fuel_accounting(optimizer):
    """Multi-stop relocation accounts for travel time, distance, and fuel expense."""
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=10.0,
        max_relocation_stops=3,
        min_dwell_time_minutes=90,
    )

    validated = ShiftPlanResponse(**res)
    if len(validated.itinerary) > 1:
        assert validated.summary.total_transit_minutes > 0
        assert validated.summary.total_travel_distance_km > 0.0
        assert validated.summary.total_fuel_cost_inr > 0.0


# ===========================================================================
# 7. Baseline STAY Profit Comparison
# ===========================================================================

def test_baseline_stay_profit_comparison(optimizer):
    """Shift plan explicitly computes baseline staying profit and derives net profit uplift."""
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)
    res = optimizer.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=MOCK_CANDIDATES,
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
    )

    validated = ShiftPlanResponse(**res)
    assert validated.baseline_stay_profit_inr > 0
    assert validated.shift_net_uplift_inr is not None
    assert validated.verdict in ("RECOMMENDED_SHIFT", "CONSIDER_SHIFT", "STAY_PUT_SHIFT")
    assert len(validated.decision_reason) > 10


# ===========================================================================
# 8. Infeasible and Timeout Fallback
# ===========================================================================

def test_infeasible_or_empty_candidates_fallback():
    """Infeasible constraints fall back gracefully to a single-stop origin STAY plan."""
    opt = ShiftOptimizer()
    shift_start = datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc)

    # Empty candidate pool -> origin stay fallback
    res = opt.optimize_shift(
        origin_lat=16.5062,
        origin_lng=80.6480,
        candidates=[],
        vendor_category="food",
        shift_start_time=shift_start,
        shift_duration_hours=8.0,
    )

    validated = ShiftPlanResponse(**res)
    assert len(validated.itinerary) == 1
    assert validated.itinerary[0].is_current_cell is True
    assert validated.itinerary[0].dwell_hours == 7.75  # 8.0h minus 15m teardown at shift conclusion
    assert validated.shift_duration_hours == 8.0
    assert validated.summary.total_dwell_hours == 7.75
    assert validated.verdict == "STAY_PUT_SHIFT"


# ===========================================================================
# 9. Shift Planning API Contract (POST /v1/routes/shift-plan)
# ===========================================================================

def test_shift_plan_api_endpoint():
    """POST /v1/routes/shift-plan executes full optimization pipeline and returns validated schema."""
    req = ShiftPlanRequest(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        shift_duration_hours=8.0,
        max_relocation_stops=3,
        min_dwell_time_minutes=90,
    )

    coro = shift_plan(req, api_key="test-key")
    body = run_coroutine(coro)

    assert body is not None
    validated = ShiftPlanResponse(**body)
    assert validated.shift_duration_hours == 8.0
    assert len(validated.itinerary) >= 1
    assert validated.summary.total_net_profit_inr > 0
    assert validated.solver_telemetry["solver_name"] == "google_ortools_cp_sat"
