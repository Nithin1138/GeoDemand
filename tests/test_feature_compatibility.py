"""
GeoDemand AI — Feature Compatibility & Inference Integration Tests

Phase 3.2 Deliverable: Comprehensive verification covering all 19 Phase 3.2 tasks:
1. Integrate weather provider.
2. Integrate OSM/POI provider.
3. Integrate holiday provider.
4. Integrate event features.
5. Integrate competition features.
6. Accept vendor location.
7. Accept vendor category.
8. Accept vendor inventory.
9. Assemble model features.
10. Match feature names and ordering.
11. Validate required feature fields.
12. Handle missing contextual features.
13. Handle unavailable external providers.
14. Generate next-hour customer-count predictions.
15. Generate P10/P50/P90 estimates.
16. Include prediction timestamp.
17. Include model version.
18. Include feature freshness metadata.
19. Validate prediction response.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from api.feature_assembler import FeatureAssembler, CANONICAL_FEATURE_COLUMNS, _calendar_features
from api.location import get_location_context
from api.recommender import DemandModel
from api.providers.competition import CompetitionProviderFactory, OSMCompetitionProvider
from api.providers.events import EventProviderFactory, StochasticEventProvider
from features.weather_client import OpenMeteoClient
from api.schemas import PredictDemandRequest, PredictDemandResponse


MOCK_WEATHER = {
    "temperature": 29.0,
    "humidity": 65.0,
    "rainfall": 0.0,
    "wind_speed": 11.0,
    "pressure": 1010.0,
    "cloud_cover": 20.0,
    "visibility": 10000.0,
    "weather_condition": "clear",
    "uv_index": 5.5,
    "source_type": "real_live",
    "source_name": "open-meteo",
    "retrieved_at": "2026-11-16T10:00:00+00:00",
}


@pytest.fixture(scope="module")
def assembler():
    return FeatureAssembler()


@pytest.fixture(scope="module")
def demand_model():
    return DemandModel.get()


@pytest.fixture(scope="module")
def canonical_schema():
    schema_path = ROOT / "models" / "lgbm_v1" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Weather Provider Integration
# ===========================================================================

def test_weather_provider_integration(assembler):
    """Weather provider returns complete hourly forecast and assembler maps all 9 weather features."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    comp = {"competition_score": 0.3}
    event = {"event_active": False, "event_importance": 0.0}

    features, freshness = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, comp, event)

    for field in ["temperature", "humidity", "rainfall", "wind_speed", "cloud_cover", "pressure", "uv_index", "visibility", "is_rainy"]:
        assert field in features, f"Missing weather feature: {field}"
        assert not np.isnan(features[field]), f"NaN weather feature: {field}"

    assert freshness["weather"]["source_name"] == "open-meteo"
    assert freshness["weather"]["source_type"] == "real_live"


# ===========================================================================
# 2. OSM / POI Provider Integration
# ===========================================================================

def test_osm_poi_provider_integration(assembler):
    """Static and dynamic POI features are loaded with non-negative counts and valid ratios."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    features, freshness = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, {}, {})

    poi_fields = [
        "office_count", "college_count", "school_count", "hospital_count",
        "mall_count", "restaurant_count", "park_count", "bus_stop_count",
        "railway_station_count", "metro_station_count", "parking_count",
        "road_density", "building_density", "commercial_ratio",
        "residential_ratio", "industrial_ratio", "population_density",
    ]
    for field in poi_fields:
        assert field in features, f"Missing POI feature: {field}"
        assert features[field] >= 0.0, f"Negative POI feature: {field}={features[field]}"

    assert 0.0 <= features["commercial_ratio"] <= 1.0
    assert 0.0 <= features["residential_ratio"] <= 1.0
    assert freshness["static_poi"]["source_name"] == "openstreetmap"


# ===========================================================================
# 3. Holiday and Calendar Provider Integration
# ===========================================================================

def test_holiday_calendar_provider_integration():
    """Calendar logic derives accurate cyclical sinusoidal features, weekend, and holiday indicators."""
    test_dt = datetime(2026, 8, 15, 14, 30, tzinfo=timezone.utc)  # Independence Day (Saturday)
    cal = _calendar_features(test_dt)

    assert cal["hour"] == 14
    assert cal["day_of_week"] == 5  # Saturday
    assert cal["is_weekend"] == 1
    assert cal["is_holiday"] == 1  # Indian National Holiday

    # Monotonicity & boundary of cyclical trigonometric transformations
    for trig in ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]:
        assert -1.0 <= cal[trig] <= 1.0, f"{trig} out of unit circle bounds: {cal[trig]}"

    # Pythagorean identity verification
    assert abs(cal["hour_sin"]**2 + cal["hour_cos"]**2 - 1.0) < 1e-5


# ===========================================================================
# 4. Event Features Integration
# ===========================================================================

def test_event_features_integration(assembler):
    """Event importance and attendance estimates integrate into model features."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    event_data = {
        "event_active": True,
        "event_importance": 0.85,
        "event_attendance_estimate": 4500,
        "source_type": "real_live",
        "source_name": "eventbrite",
    }
    features, freshness = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, {}, event_data)

    assert features["event_importance"] == 0.85
    assert features["active_events_count"] == 1.0
    assert features["event_max_attendance"] == 4500.0
    assert freshness["events"]["active"] is True


# ===========================================================================
# 5. Competition Features Integration
# ===========================================================================

def test_competition_features_integration(assembler):
    """Competition score correctly integrates from provider."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    comp_data = {
        "competition_score": 0.72,
        "competition_level": "high",
        "source_type": "real_periodic",
        "source_name": "openstreetmap",
    }
    features, freshness = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, comp_data, {})

    assert features["competition_score"] == 0.72
    assert freshness["competition"]["source_type"] == "real_periodic"


# ===========================================================================
# 6. Vendor Location Acceptance
# ===========================================================================

def test_vendor_location_acceptance():
    """GPS coordinates (lat, lng) are accepted and resolved to valid H3 cell."""
    ctx = get_location_context(16.5062, 80.6480)
    assert ctx.h3_cell == "88619aa6c7fffff"
    assert ctx.h3_resolution == 8
    assert ctx.is_water is False


# ===========================================================================
# 7. Vendor Category Acceptance
# ===========================================================================

@pytest.mark.parametrize("category", ["food", "fruits", "repair", "beverages", "snacks"])
def test_vendor_category_acceptance(assembler, category):
    """Various commercial vendor categories are ingested without error."""
    vendor = {"vendor_category": category, "inventory_capacity": 50, "preparation_time": 10.0, "average_order_value": 75.0}
    features, _ = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, {}, {})
    assert features["vendor_category"] == category


# ===========================================================================
# 8. Vendor Inventory Acceptance
# ===========================================================================

def test_vendor_inventory_acceptance(assembler):
    """Vendor inventory capacity and preparation time are correctly ingested."""
    vendor = {
        "vendor_category": "food",
        "inventory_capacity": 350,
        "preparation_time": 12.5,
        "average_order_value": 150.0,
    }
    features, _ = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, {}, {})
    assert features["inventory_capacity"] == 350.0
    assert features["preparation_time"] == 12.5
    assert features["average_order_value"] == 150.0


# ===========================================================================
# 9 & 10. Model Feature Assembly, Canonical Names & Ordering
# ===========================================================================

def test_feature_assembly_canonical_schema_match(assembler, canonical_schema):
    """Assembled features contain all 45 canonical columns with exact matching and ordering."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    features, _ = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, {}, {})

    expected_cols = canonical_schema["feature_columns"]
    assert len(expected_cols) == 45

    # Check all 45 are present
    missing = [c for c in expected_cols if c not in features]
    assert not missing, f"Missing required canonical features: {missing}"

    # Verify align_features strictly preserves the exact canonical order
    aligned = assembler.align_features(features)
    assert list(aligned.keys()) == expected_cols


# ===========================================================================
# 11. Validate Required Feature Fields
# ===========================================================================

def test_feature_validation_rejects_missing_or_nan(assembler):
    """Validation method raises ValueError if required features are missing or NaN."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    features, _ = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, {}, {})

    # Valid dictionary passes
    assembler.validate_features(features)

    # Missing column raises ValueError
    bad_features = features.copy()
    del bad_features["temperature"]
    with pytest.raises(ValueError, match="Missing required features"):
        assembler.validate_features(bad_features)

    # NaN column raises ValueError
    nan_features = features.copy()
    nan_features["population_density"] = float("nan")
    with pytest.raises(ValueError, match="contains invalid numeric value"):
        assembler.validate_features(nan_features)


# ===========================================================================
# 12. Handle Missing Contextual Features
# ===========================================================================

def test_missing_contextual_features_imputation(assembler):
    """When an unknown cell outside grid is passed, median imputation produces valid vector."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    unknown_cell = "8861892433fffff"  # Cell outside Vijayawada static grid

    features, freshness = assembler.assemble(unknown_cell, vendor, MOCK_WEATHER, {}, {})

    assert assembler.validate_features(features) is None
    assert all(not (isinstance(v, float) and np.isnan(v)) for v in features.values())


# ===========================================================================
# 13. Handle Unavailable External Providers
# ===========================================================================

def test_unavailable_external_providers_fallback(assembler):
    """When weather and competition dictionaries are empty, assembler applies safe fallbacks."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    empty_weather = {}
    empty_comp = {}
    empty_event = {}

    features, freshness = assembler.assemble("88618925bbfffff", vendor, empty_weather, empty_comp, empty_event)

    assert features["temperature"] == 28.0
    assert features["humidity"] == 60.0
    assert features["competition_score"] == 0.3
    assert features["event_importance"] == 0.0
    assert assembler.validate_features(features) is None


# ===========================================================================
# 14 & 15. Next-Hour Customer-Count Predictions & Calibrated P10/P50/P90
# ===========================================================================

def test_next_hour_customer_predictions_and_quantiles(assembler, demand_model):
    """Inference generates valid non-negative expected customers and monotonic P10 <= P50 <= P90."""
    vendor = {"vendor_category": "food", "inventory_capacity": 100, "preparation_time": 5.0, "average_order_value": 100.0}
    features, _ = assembler.assemble("88618925bbfffff", vendor, MOCK_WEATHER, {"competition_score": 0.2}, {})

    uncertainty = demand_model.predict_with_uncertainty(features)

    p10 = uncertainty["p10"]
    p50 = uncertainty["p50"]
    p90 = uncertainty["p90"]

    assert p10 >= 0, f"Negative P10: {p10}"
    assert p50 >= 0, f"Negative P50: {p50}"
    assert p90 >= 0, f"Negative P90: {p90}"
    assert p10 <= p50 <= p90, f"Quantile inversion detected: P10={p10}, P50={p50}, P90={p90}"
    assert uncertainty["interval_spread"] == p90 - p10
    assert uncertainty["uncertainty_level"] in ("LOW", "MODERATE", "HIGH")


# ===========================================================================
# 16, 17, 18, 19. Prediction API Schema & Contract Validation
# ===========================================================================

def test_predict_demand_api_contract_validation():
    """POST /v1/predict/demand returns strict PredictDemandResponse schema with all required fields."""
    from unittest.mock import MagicMock
    import asyncio
    from api.app import _load_model, predict_demand
    from api.schemas import PredictDemandRequest

    _load_model()

    mock_w_client = MagicMock()
    mock_w_client.get_hourly_forecast.return_value = {**MOCK_WEATHER, "source_type": "real_live"}

    mock_comp = MagicMock()
    mock_comp.get_competition_score.return_value = {
        "competition_score": 0.5,
        "competition_level": "medium",
        "raw_competitor_count": 5,
        "source_type": "real_periodic",
    }

    mock_traffic = MagicMock()
    mock_events = MagicMock()
    mock_events.get_event_importance.return_value = {
        "event_active": False,
        "event_importance": 0.0,
        "source_type": "real_periodic",
    }
    mock_fuel = MagicMock()

    with (
        patch("api.app._get_weather_client", return_value=mock_w_client),
        patch("api.app._get_providers", return_value=(mock_comp, mock_traffic, mock_events, mock_fuel)),
    ):
        req = PredictDemandRequest(
            latitude=16.5062,
            longitude=80.6480,
            vendor_category="food",
            average_order_value=120.0,
            variable_cost_rate=0.35,
            fixed_cost_per_day=450.0,
            inventory_capacity=150,
            preparation_time=4.0,
        )

        coro = predict_demand(req, api_key="dev-key")
        try:
            coro.send(None)
            body = None
        except StopIteration as e:
            body = e.value

        assert body is not None

        # 16. Prediction timestamp
        assert "prediction_timestamp" in body
        assert datetime.fromisoformat(body["prediction_timestamp"])

        # 17. Model version
        assert "model_version" in body
        assert "model_type" in body
        assert body["model_type"] == "lightgbm"

        # 18. Feature freshness metadata
        assert "data_freshness" in body
        fresh = body["data_freshness"]
        for source in ["weather", "static_poi", "calendar", "competition", "events"]:
            assert source in fresh, f"Missing freshness metadata for {source}"

        # 19. Validate response via Pydantic model
        validated = PredictDemandResponse(**body)
        assert validated.prediction["expected_customers"] > 0
        assert validated.economics.expected_revenue_inr > 0
        assert validated.economics.expected_profit_inr > 0
