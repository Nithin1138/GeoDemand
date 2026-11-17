"""
Test suite for GeoDemand AI Scenario Simulation & What-If Counterfactual Engine.

Verifies:
- Rainfall and precipitation changes
- Competitor density changes
- Fuel price changes
- Vendor commercial category changes
- Hour-of-day operational schedule shifts
- Pricing and AOV elasticity sensitivity
- Multi-factor counterfactual simulation
- Recalculation strictly without model retraining
- Strict schema validation against ScenarioSimulateResponse
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.app import _load_model, scenario_simulate
from api.recommender import DemandModel
from api.scenario_simulation import ScenarioSimulator
from api.schemas import (
    ScenarioSimulateRequest,
    ScenarioSimulateResponse,
)

# Mock base telemetry
MOCK_WEATHER = {
    "temperature_c": 31.0,
    "relative_humidity_pct": 65.0,
    "precipitation_mm": 0.0,
    "rainfall": 0.0,
    "wind_speed_kmh": 12.0,
    "solar_radiation_wm2": 350.0,
    "is_rain": False,
    "heat_index_c": 33.0,
    "weather_condition": "clear",
    "source_type": "real_live",
}

MOCK_COMP = {
    "competition_score": 0.40,
    "competition_level": "medium",
    "raw_competitor_count": 4,
    "source_type": "real_periodic",
}

MOCK_EVENT = {
    "event_active": False,
    "event_importance": 0.0,
    "source_type": "real_periodic",
}


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


@pytest.fixture
def mock_providers():
    """Provide fully mocked external provider interfaces."""
    wc = MagicMock()
    wc.get_hourly_forecast.return_value = copy_dict(MOCK_WEATHER)

    comp = MagicMock()
    comp.get_competition_score.return_value = copy_dict(MOCK_COMP)

    ev = MagicMock()
    ev.get_event_importance.return_value = copy_dict(MOCK_EVENT)

    fuel = MagicMock()
    fuel.get_fuel_price.return_value = {"price_inr_per_litre": 95.0}

    return wc, comp, ev, fuel


def copy_dict(d):
    return {k: v for k, v in d.items()}


# ===========================================================================
# 1. Rainfall & Weather Changes
# ===========================================================================

def test_rainfall_changes_simulation(mock_providers):
    """Simulating 15mm heavy monsoon rainfall reduces footfall and updates weather features."""
    wc, comp, ev, fuel = mock_providers
    sim = ScenarioSimulator()

    res = sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        baseline_aov=100.0,
        simulated_rainfall_mm=15.0,
        simulated_is_rain=True,
        weather_client=wc,
        competition_provider=comp,
        event_provider=ev,
        fuel_provider=fuel,
    )

    validated = ScenarioSimulateResponse(**res)
    assert validated.baseline.rainfall_mm == 0.0
    assert validated.baseline.is_rain is False
    assert validated.scenario.rainfall_mm == 15.0
    assert validated.scenario.is_rain is True

    # Heavy rain reduces customer footfall for outdoor food stalls
    assert validated.scenario.expected_customers < validated.baseline.expected_customers
    assert validated.simulation_delta.customer_delta < 0
    assert validated.simulation_delta.profit_delta_inr < 0
    assert "Rain" in validated.strategic_advice or "rain" in validated.strategic_advice


# ===========================================================================
# 2. Competitor Density Changes
# ===========================================================================

def test_competitor_density_changes_simulation(mock_providers):
    """Simulating arrival of +4 competitor stalls increases competition score and dilutes profit."""
    wc, comp, ev, fuel = mock_providers
    sim = ScenarioSimulator()

    res = sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        baseline_aov=100.0,
        simulated_competitor_count_delta=4,
        weather_client=wc,
        competition_provider=comp,
        event_provider=ev,
        fuel_provider=fuel,
    )

    validated = ScenarioSimulateResponse(**res)
    assert validated.scenario.competition_score > validated.baseline.competition_score
    assert "competition" in validated.scenario_type.lower()
    assert validated.scenario.expected_profit_inr <= validated.baseline.expected_profit_inr


# ===========================================================================
# 3. Fuel Price Changes
# ===========================================================================

def test_fuel_price_changes_simulation(mock_providers):
    """Simulating fuel price surge from ₹95 to ₹120/L increases travel costs and reduces net profit."""
    wc, comp, ev, fuel = mock_providers
    sim = ScenarioSimulator()

    res = sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        baseline_aov=100.0,
        simulated_fuel_price_inr=120.0,
        distance_km=2.5,
        weather_client=wc,
        competition_provider=comp,
        event_provider=ev,
        fuel_provider=fuel,
    )

    validated = ScenarioSimulateResponse(**res)
    assert validated.baseline.fuel_price_inr == 95.0
    assert validated.scenario.fuel_price_inr == 120.0
    assert validated.scenario.fuel_cost_inr > validated.baseline.fuel_cost_inr
    assert validated.simulation_delta.fuel_delta_inr > 0
    assert validated.simulation_delta.profit_delta_inr < 0


# ===========================================================================
# 4. Vendor Category Pivot Changes
# ===========================================================================

def test_vendor_category_pivot_simulation(mock_providers):
    """Simulating category shift from food to beverages applies beverage pricing, COGS, and capacity."""
    wc, comp, ev, fuel = mock_providers
    sim = ScenarioSimulator()

    res = sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        simulated_vendor_category="beverages",
        weather_client=wc,
        competition_provider=comp,
        event_provider=ev,
        fuel_provider=fuel,
    )

    validated = ScenarioSimulateResponse(**res)
    assert validated.baseline.vendor_category == "food"
    assert validated.scenario.vendor_category == "beverages"
    assert "category" in validated.scenario_type.lower()
    assert validated.scenario.expected_profit_inr != validated.baseline.expected_profit_inr


# ===========================================================================
# 5. Operational Hour / Schedule Shift
# ===========================================================================

def test_operational_hour_shift_simulation(mock_providers):
    """Simulating operational shift from 09:00 to 19:00 applies evening cyclical peak features."""
    wc, comp, ev, fuel = mock_providers
    sim = ScenarioSimulator()

    ref_morning = datetime(2026, 10, 5, 9, 0, 0, tzinfo=timezone.utc)
    res = sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        simulated_hour=19,
        reference_timestamp=ref_morning,
        weather_client=wc,
        competition_provider=comp,
        event_provider=ev,
        fuel_provider=fuel,
    )

    validated = ScenarioSimulateResponse(**res)
    assert validated.baseline.hour == 9
    assert validated.scenario.hour == 19
    assert "schedule" in validated.scenario_type.lower()


# ===========================================================================
# 6. Pricing & AOV Elasticity
# ===========================================================================

def test_pricing_aov_elasticity_simulation(mock_providers):
    """Simulating an AOV increase from ₹80 to ₹140 increases gross revenue and net profit."""
    wc, comp, ev, fuel = mock_providers
    sim = ScenarioSimulator()

    res = sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        baseline_aov=80.0,
        simulated_aov=140.0,
        weather_client=wc,
        competition_provider=comp,
        event_provider=ev,
        fuel_provider=fuel,
    )

    validated = ScenarioSimulateResponse(**res)
    assert validated.baseline.aov == 80.0
    assert validated.scenario.aov == 140.0
    assert validated.simulation_delta.revenue_delta_inr > 0
    assert validated.simulation_delta.profit_delta_inr > 0
    assert validated.simulation_delta.percentage_change > 0


# ===========================================================================
# 7. Recalculation Without Model Retraining
# ===========================================================================

def test_recalculation_strictly_without_retraining(mock_providers):
    """Inference is completely dynamic; champion model weights and metadata are strictly unchanged."""
    wc, comp, ev, fuel = mock_providers
    model = DemandModel.get()
    initial_meta = copy_dict(model.meta)

    sim = ScenarioSimulator(model=model)
    sim.simulate(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="fruits",
        simulated_rainfall_mm=8.0,
        simulated_competitor_count_delta=2,
        simulated_aov=120.0,
        weather_client=wc,
        competition_provider=comp,
        event_provider=ev,
        fuel_provider=fuel,
    )

    # Model metadata and weights are untouched
    assert model.meta["model_type"] == initial_meta["model_type"]
    assert model.meta.get("trained_at") == initial_meta.get("trained_at")


# ===========================================================================
# 8. API Endpoint Contract (POST /v1/scenario/simulate)
# ===========================================================================

def test_scenario_simulate_endpoint_contract():
    """POST /v1/scenario/simulate accepts full counterfactual payload and returns validated response."""
    mock_wc = MagicMock()
    mock_wc.get_hourly_forecast.return_value = copy_dict(MOCK_WEATHER)
    mock_comp = MagicMock()
    mock_comp.get_competition_score.return_value = copy_dict(MOCK_COMP)
    mock_ev = MagicMock()
    mock_ev.get_event_importance.return_value = copy_dict(MOCK_EVENT)
    mock_fuel = MagicMock()

    with (
        patch("api.app._get_weather_client", return_value=mock_wc),
        patch("api.app._get_providers", return_value=(mock_comp, MagicMock(), mock_ev, mock_fuel)),
    ):
        req = ScenarioSimulateRequest(
            latitude=16.5062,
            longitude=80.6480,
            vendor_category="food",
            baseline_aov=90.0,
            simulated_aov=130.0,
            simulated_rainfall_mm=10.0,
            simulated_competitor_count_delta=1,
            simulated_fuel_price_inr=105.0,
            simulated_hour=20,
        )

        coro = scenario_simulate(req, api_key="test-key")
        body = run_coroutine(coro)

        assert body is not None
        validated = ScenarioSimulateResponse(**body)
        assert validated.baseline.aov == 90.0
        assert validated.scenario.aov == 130.0
        assert validated.scenario.rainfall_mm == 10.0
        assert validated.scenario.hour == 20
        assert validated.simulation_delta.profit_delta_inr is not None
        assert len(validated.explanation) > 10
        assert len(validated.strategic_advice) > 10
