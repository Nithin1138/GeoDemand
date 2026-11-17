"""
Comprehensive API endpoint verification and error-handling test suite.

Verifies:
- GET /v1/system/ping
- GET /v1/system/config
- GET /v1/system/data-health
- POST /v1/location/context
- POST /v1/predict/demand
- POST /v1/predict/batch
- POST /v1/recommendations/live
- POST /v1/recommend/zones
- POST /v1/scenario/simulate
- GET & POST /v1/routing/route
- GET /v1/system/cache-stats & POST /v1/system/cache-clear
- API-key authentication enforcement and rejection
- Response caching execution and hit rates
- Batch scoring aggregation and constraints
- Comprehensive input boundary validation and error handling
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.app import (
    _load_model,
    ping,
    system_config,
    data_health,
    location_context,
    predict_demand,
    predict_batch,
    recommendations_live,
    recommend_zones,
    scenario_simulate,
    get_street_route,
    cache_stats,
    cache_clear,
    verify_api_key,
    ingest_transaction,
    ingest_transactions_batch,
    get_telemetry_health,
)
from api.cache import api_response_cache
from api.schemas import (
    LocationRequest,
    LocationContextResponse,
    PredictDemandRequest,
    PredictDemandResponse,
    BatchPredictDemandRequest,
    BatchPredictDemandResponse,
    RecommendationRequest,
    LiveRecommendationsResponse,
    RecommendZonesRequest,
    ScenarioSimulateRequest,
    ScenarioSimulateResponse,
    PingResponse,
    SystemConfigResponse,
    DataHealthResponse,
    APICacheStatsResponse,
    StreetRouteResponse,
)

MOCK_WEATHER = {
    "temperature": 29.0,
    "temperature_c": 29.0,
    "humidity": 68.0,
    "relative_humidity_pct": 68.0,
    "precipitation_mm": 0.0,
    "rainfall": 0.0,
    "wind_speed_kmh": 11.0,
    "solar_radiation_wm2": 300.0,
    "is_rain": False,
    "heat_index_c": 31.0,
    "weather_condition": "clear",
    "source_type": "real_live",
    "source_name": "open-meteo",
}

MOCK_COMP = {
    "competition_score": 0.45,
    "competition_level": "medium",
    "raw_competitor_count": 6,
    "source_type": "real_periodic",
    "source_name": "openstreetmap",
}

MOCK_TRAFFIC = {
    "traffic_speed_kmh": 18.0,
    "traffic_level": "medium",
    "traffic_delay_min": 1.0,
    "congestion_ratio": 0.20,
    "source_type": "synthetic_fallback",
    "source_name": "time_of_day_proxy",
}

MOCK_EVENT = {
    "event_active": False,
    "event_importance": 0.0,
    "source_type": "real_periodic",
    "source_name": "stochastic",
}

MOCK_FUEL = {
    "fuel_cost_inr": 12.50,
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


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear response cache before each test."""
    api_response_cache.clear()
    yield
    api_response_cache.clear()


@pytest.fixture
def mock_all_providers():
    """Mock external providers for reliable sub-second testing."""
    mock_wc = MagicMock()
    mock_wc.get_current_weather.return_value = {**MOCK_WEATHER}
    mock_wc.get_hourly_forecast.return_value = {**MOCK_WEATHER}

    mock_comp = MagicMock()
    mock_comp.get_competition_score.return_value = {**MOCK_COMP}
    mock_comp.source_type = "real_periodic"
    mock_comp.source_name = "openstreetmap"

    mock_traffic = MagicMock()
    mock_traffic.get_traffic.return_value = {**MOCK_TRAFFIC}
    mock_traffic.source_type = "synthetic_fallback"
    mock_traffic.source_name = "time_of_day_proxy"

    mock_events = MagicMock()
    mock_events.get_event_importance.return_value = {**MOCK_EVENT}
    mock_events.source_type = "real_periodic"
    mock_events.source_name = "stochastic"

    mock_fuel = MagicMock()
    mock_fuel.get_fuel_cost_for_distance.return_value = {**MOCK_FUEL}
    mock_fuel.get_fuel_price.return_value = {"price_inr_per_litre": 95.0}

    with (
        patch("api.app._get_weather_client", return_value=mock_wc),
        patch("api.app._get_providers", return_value=(mock_comp, mock_traffic, mock_events, mock_fuel)),
    ):
        yield mock_wc, mock_comp, mock_traffic, mock_events, mock_fuel


# ===========================================================================
# 1. System Endpoints
# ===========================================================================

def test_system_ping():
    """GET /v1/system/ping returns operational status and semantic version."""
    coro = ping()
    body = run_coroutine(coro)
    validated = PingResponse(**body)
    assert validated.status == "ok"
    assert validated.service == "GeoDemand AI"
    assert validated.version == "2.1.0"
    assert datetime.fromisoformat(validated.timestamp)


def test_system_config():
    """GET /v1/system/config returns public deployment coordinates."""
    coro = system_config()
    body = run_coroutine(coro)
    validated = SystemConfigResponse(**body)
    assert validated.city == "Vijayawada"
    assert validated.state == "Andhra Pradesh"
    assert 16.0 <= validated.default_lat <= 17.0
    assert 80.0 <= validated.default_lng <= 81.0


def test_system_data_health(mock_all_providers):
    """GET /v1/system/data-health returns provenance telemetry across all subsystems."""
    coro = data_health()
    body = run_coroutine(coro)
    validated = DataHealthResponse(**body)
    assert validated.model["status"] == "loaded"
    assert validated.weather.source_type in ("real_live", "synthetic_fallback")
    assert validated.competition.source_type in ("real_periodic", "real_live")
    assert validated.traffic.source_type in ("synthetic_fallback", "real_live")
    assert validated.poi.source_type == "real_periodic"
    assert validated.customer_demand.source_type == "derived"


# ===========================================================================
# 2. Location Context Endpoint
# ===========================================================================

def test_location_context_endpoint():
    """POST /v1/location/context resolves GPS coordinates to H3 index at resolution 8."""
    req = LocationRequest(latitude=16.5062, longitude=80.6480)
    coro = location_context(req)
    body = run_coroutine(coro)
    validated = LocationContextResponse(**body)
    assert validated.h3_cell == "88619aa6c7fffff"
    assert validated.h3_resolution == 8
    assert validated.is_water is False
    assert validated.timezone_name == "Asia/Kolkata"


# ===========================================================================
# 3. Demand Prediction Endpoint
# ===========================================================================

def test_predict_demand_endpoint(mock_all_providers):
    """POST /v1/predict/demand returns quantiles, unit economics, and data freshness."""
    req = PredictDemandRequest(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        average_order_value=120.0,
    )
    coro = predict_demand(req, api_key="test-key")
    body = run_coroutine(coro)
    validated = PredictDemandResponse(**body)
    assert validated.prediction["expected_customers"] > 0
    assert validated.prediction["p10"] <= validated.prediction["p50"] <= validated.prediction["p90"]
    assert validated.economics.expected_revenue_inr > 0
    assert validated.economics.expected_profit_inr > 0
    assert "weather" in validated.data_freshness


# ===========================================================================
# 4. Batch Scoring Endpoint
# ===========================================================================

def test_predict_batch_endpoint(mock_all_providers):
    """POST /v1/predict/batch computes batch demand and summary statistics across locations."""
    items = [
        PredictDemandRequest(latitude=16.5062, longitude=80.6480, vendor_category="food"),
        PredictDemandRequest(latitude=16.5150, longitude=80.6320, vendor_category="fruits"),
        PredictDemandRequest(latitude=16.5200, longitude=80.6200, vendor_category="beverages"),
    ]
    batch_req = BatchPredictDemandRequest(items=items)
    coro = predict_batch(batch_req, api_key="test-key")
    body = run_coroutine(coro)
    validated = BatchPredictDemandResponse(**body)

    assert validated.count == 3
    assert len(validated.results) == 3
    assert validated.summary.total_locations == 3
    assert validated.summary.total_expected_customers > 0
    assert validated.summary.total_expected_revenue_inr > 0
    assert validated.summary.best_location_profit_inr > 0
    assert 0 <= validated.summary.best_location_index < 3


# ===========================================================================
# 5. Live Recommendations Endpoint
# ===========================================================================

def test_recommendations_live_endpoint(mock_all_providers):
    """POST /v1/recommendations/live returns ranked candidates, overall decision, and heatmap."""
    req = RecommendationRequest(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        search_radius_km=2.0,
        top_n=5,
    )
    coro = recommendations_live(req, api_key="test-key")
    body = run_coroutine(coro)
    validated = LiveRecommendationsResponse(**body)

    assert len(validated.recommendations) <= 5
    assert validated.decision.verdict in ("RECOMMENDED_MOVE", "CONSIDER_MOVE", "STAY_PUT")
    assert validated.total_candidates_evaluated > 0
    assert len(validated.demand_heatmap) > 0
    assert validated.current_location["h3_cell"] == "88619aa6c7fffff"


# ===========================================================================
# 6. Recommend Zones Endpoint
# ===========================================================================

def test_recommend_zones_endpoint(mock_all_providers):
    """POST /v1/recommend/zones proxies to live recommendations pipeline."""
    req = RecommendZonesRequest(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        search_radius_km=1.5,
        top_n=3,
    )
    coro = recommend_zones(req, api_key="test-key")
    body = run_coroutine(coro)
    validated = LiveRecommendationsResponse(**body)
    assert len(validated.recommendations) <= 3


# ===========================================================================
# 7. Scenario Simulation Endpoint
# ===========================================================================

def test_scenario_simulation_endpoint(mock_all_providers):
    """POST /v1/scenario/simulate executes counterfactual analysis without model retraining."""
    req = ScenarioSimulateRequest(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
        baseline_aov=100.0,
        simulated_aov=150.0,
        simulated_rainfall_mm=10.0,
    )
    coro = scenario_simulate(req, api_key="test-key")
    body = run_coroutine(coro)
    validated = ScenarioSimulateResponse(**body)
    assert validated.baseline.aov == 100.0
    assert validated.scenario.aov == 150.0
    assert validated.scenario.rainfall_mm == 10.0
    assert validated.simulation_delta.profit_delta_inr is not None
    assert len(validated.strategic_advice) > 5


# ===========================================================================
# 8. Routing Route Endpoint
# ===========================================================================

def test_routing_route_endpoint():
    """GET/POST /v1/routing/route returns street coordinate polyline."""
    coro = get_street_route(16.5062, 80.6480, 16.5150, 80.6320)
    body = run_coroutine(coro)
    validated = StreetRouteResponse(**body)
    assert validated.code in ("Ok", "Fallback")
    assert validated.is_routable is True
    assert validated.distance_km > 0
    assert validated.duration_min >= 0
    assert len(validated.coordinates) >= 2
    for pt in validated.coordinates:
        assert len(pt) == 2
        assert 16.0 <= pt[0] <= 17.0
        assert 80.0 <= pt[1] <= 81.0


# ===========================================================================
# 9. API-Key Authentication Security
# ===========================================================================

def test_api_key_authentication_enforcement():
    """When REQUIRE_API_KEY is True, missing or invalid keys raise HTTP 401."""
    with patch("api.app.REQUIRE_API_KEY", True), patch("api.app.API_KEY", "secret-test-key"):
        # Missing key -> raises 401
        with pytest.raises(HTTPException) as exc1:
            verify_api_key(None)
        assert exc1.value.status_code == 401

        # Invalid key -> raises 401
        with pytest.raises(HTTPException) as exc2:
            verify_api_key("wrong-key")
        assert exc2.value.status_code == 401

        # Valid key -> succeeds
        assert verify_api_key("secret-test-key") == "secret-test-key"


# ===========================================================================
# 10. Response Caching Execution & Hit Rates
# ===========================================================================

def test_response_caching_and_telemetry(mock_all_providers):
    """Subsequent requests hit memory cache and cache telemetry reports hits."""
    req = PredictDemandRequest(
        latitude=16.5062,
        longitude=80.6480,
        vendor_category="food",
    )

    # Initial call -> cache miss
    coro1 = predict_demand(req, api_key="test-key")
    res1 = run_coroutine(coro1)

    # Second call -> cache hit
    coro2 = predict_demand(req, api_key="test-key")
    res2 = run_coroutine(coro2)

    assert res1 == res2

    # Check cache stats
    coro_stats = cache_stats()
    stats = run_coroutine(coro_stats)
    validated_stats = APICacheStatsResponse(**stats)
    assert validated_stats.hits >= 1
    assert validated_stats.active_entries >= 1

    # Clear cache
    coro_clear = cache_clear(api_key="test-key")
    clear_res = run_coroutine(coro_clear)
    assert clear_res["status"] == "ok"

    coro_stats2 = cache_stats()
    stats2 = run_coroutine(coro_stats2)
    assert stats2["active_entries"] == 0


# ===========================================================================
# 11. Error Handling & Input Validation Tests
# ===========================================================================

def test_invalid_latitude_validation():
    """Coordinates with latitude > 90 or < -90 are rejected by Pydantic."""
    with pytest.raises(ValidationError):
        LocationRequest(latitude=95.0, longitude=80.6480)

    with pytest.raises(ValidationError):
        LocationRequest(latitude=-95.0, longitude=80.6480)


def test_invalid_longitude_validation():
    """Coordinates with longitude > 180 or < -180 are rejected by Pydantic."""
    with pytest.raises(ValidationError):
        LocationRequest(latitude=16.5062, longitude=185.0)

    with pytest.raises(ValidationError):
        LocationRequest(latitude=16.5062, longitude=-185.0)


def test_negative_search_radius_validation():
    """Search radius <= 0 or > 10 km is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        RecommendZonesRequest(latitude=16.5062, longitude=80.6480, search_radius_km=-1.0)

    with pytest.raises(ValidationError):
        RecommendZonesRequest(latitude=16.5062, longitude=80.6480, search_radius_km=15.0)


def test_negative_aov_validation():
    """Average order value <= 0 is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        PredictDemandRequest(latitude=16.5062, longitude=80.6480, average_order_value=-50.0)


def test_batch_size_constraints():
    """Empty batch or batch exceeding 50 items is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        BatchPredictDemandRequest(items=[])

    too_many = [PredictDemandRequest(latitude=16.5062, longitude=80.6480) for _ in range(51)]
    with pytest.raises(ValidationError):
        BatchPredictDemandRequest(items=too_many)


def test_water_location_handling():
    """Water locations (e.g. Krishna river centroid) return is_water=True."""
    req = LocationRequest(latitude=16.5120, longitude=80.6120)  # Krishna river
    coro = location_context(req)
    body = run_coroutine(coro)
    assert body["is_water"] is True


def test_unloaded_model_raises_503():
    """When ML model is not loaded, prediction endpoints return HTTP 503."""
    with patch("api.app._model_health", {"status": "not_loaded", "loaded_at": None}):
        req = PredictDemandRequest(latitude=16.5062, longitude=80.6480)
        coro = predict_demand(req, api_key="test-key")
        with pytest.raises(HTTPException) as exc:
            run_coroutine(coro)
        assert exc.value.status_code == 503


def test_telemetry_transaction_ingest_endpoint():
    """POST /v1/telemetry/transactions ingests POS events, dedups, and enriches H3."""
    import asyncio
    from src.ingestion.schema import TransactionRecord, TransactionLocation, PaymentMethod
    tx_id = f"TX-INT-{datetime.now(timezone.utc).timestamp()}"
    rec = TransactionRecord(
        transaction_id=tx_id,
        vendor_id="V-TEST-99",
        event_timestamp=datetime.now(timezone.utc),
        amount_inr=350.0,
        payment_method=PaymentMethod.UPI,
        location=TransactionLocation(latitude=16.5062, longitude=80.6480),
    )
    res = asyncio.run(ingest_transaction(rec, api_key="test-key"))
    assert res.status == "ACCEPTED"
    assert res.transaction_id == tx_id
    assert res.h3_cell_id == "88619aa6c7fffff"
    assert res.is_duplicate is False

    # Second ingestion of same ID is rejected with HTTP 409 Conflict
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ingest_transaction(rec, api_key="test-key"))
    assert exc.value.status_code == 409
    assert "Duplicate transaction ID rejected" in str(exc.value.detail)


def test_telemetry_batch_ingest_endpoint():
    """POST /v1/telemetry/transactions/batch handles multiple transactions."""
    import asyncio
    from src.ingestion.schema import TransactionRecord, TransactionLocation, PaymentMethod
    batch = [
        TransactionRecord(
            transaction_id=f"TX-B1-{datetime.now(timezone.utc).timestamp()}",
            vendor_id="V-BATCH-1",
            event_timestamp=datetime.now(timezone.utc),
            amount_inr=120.0,
            payment_method=PaymentMethod.CASH,
            location=TransactionLocation(latitude=16.5062, longitude=80.6480),
        ),
        TransactionRecord(
            transaction_id=f"TX-B2-{datetime.now(timezone.utc).timestamp()}",
            vendor_id="V-BATCH-2",
            event_timestamp=datetime.now(timezone.utc),
            amount_inr=480.0,
            payment_method=PaymentMethod.CARD,
            location=TransactionLocation(latitude=16.5120, longitude=80.6350),
        ),
    ]
    res = asyncio.run(ingest_transactions_batch(batch, api_key="test-key"))
    assert res.status == "PROCESSED"
    assert res.total_received == 2
    assert res.successful_ingestions == 2
    assert len(res.results) == 2


def test_telemetry_health_endpoint():
    """GET /v1/telemetry/health returns telemetry pipeline status and config."""
    import asyncio
    res = asyncio.run(get_telemetry_health())
    assert res["status"] == "HEALTHY"
    assert "seen_transactions_count" in res
    assert res["default_city"] == "Vijayawada"


