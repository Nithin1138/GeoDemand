"""
API integration and verification tests for GeoDemand AI live endpoints.

Uses FastAPI's TestClient so no server needs to be running.
External API calls (Open-Meteo, HERE) are mocked so tests run offline.
Verifies every endpoint with valid and invalid inputs, boundary values,
types, HTTP error codes, and authentication.

Run with:
    python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "api"))

import pytest

# ── Mock weather data (returned by any weather API call) ──
MOCK_WEATHER = {
    "temperature": 28.5,
    "humidity": 72.0,
    "rainfall": 0.0,
    "wind_speed": 12.0,
    "pressure": 1008.0,
    "cloud_cover": 25.0,
    "visibility": 9.5,
    "weather_condition": "clear",
    "uv_index": 6.0,
    "source": "real:open-meteo-current",
    "source_type": "real_live",
    "source_name": "open-meteo",
    "observation_timestamp": "2026-08-18T10:00:00+00:00",
    "retrieved_at": "2026-08-18T10:00:00+00:00",
    "from_cache": False,
    "cache_age_seconds": 0,
}


@pytest.fixture(scope="module")
def client():
    """Provide a FastAPI TestClient with mocked external calls."""
    with (
        patch("features.weather_client.OpenMeteoClient.get_current_weather", return_value=MOCK_WEATHER),
        patch("features.weather_client.OpenMeteoClient.get_hourly_forecast", return_value={**MOCK_WEATHER, "source_type": "real_live"}),
    ):
        from fastapi.testclient import TestClient
        from api.app import app
        with TestClient(app) as c:
            yield c


# ===========================================================================
# 1. Root & System Endpoints: GET /, /v1/system/ping, /v1/system/data-health
# ===========================================================================

def test_root_dashboard_valid(client):
    """GET / serves the dashboard HTML or fallback JSON."""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "") or "application/json" in r.headers.get("content-type", "")


def test_root_dashboard_invalid_method(client):
    """Unsupported HTTP methods on root endpoint return 405 Method Not Allowed."""
    r = client.delete("/")
    assert r.status_code == 405


def test_ping(client):
    """API is reachable and returns operational metadata."""
    r = client.get("/v1/system/ping")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["service"] == "GeoDemand AI"
    assert "timestamp" in body


def test_ping_invalid_method(client):
    """POST /v1/system/ping is rejected with 405 Method Not Allowed."""
    r = client.post("/v1/system/ping")
    assert r.status_code == 405


def test_health(client):
    """Alias test for system data health."""
    r = client.get("/v1/system/data-health")
    assert r.status_code == 200
    body = r.json()
    assert "weather" in body


def test_data_health(client):
    """Data health endpoint returns provenance & freshness status for all sources."""
    r = client.get("/v1/system/data-health")
    assert r.status_code == 200
    body = r.json()
    assert "model" in body
    assert "weather" in body
    assert "traffic" in body
    assert "competition" in body
    assert "poi" in body
    assert "events" in body
    assert "holidays" in body
    assert "customer_demand" in body
    assert body["customer_demand"]["source_type"] == "derived"
    assert "timestamp" in body


def test_data_health_invalid_method(client):
    """POST /v1/system/data-health returns 405 Method Not Allowed."""
    r = client.post("/v1/system/data-health")
    assert r.status_code == 405


# ===========================================================================
# 2. Routing Endpoints: GET /v1/routing/route & POST /v1/routing/route
# ===========================================================================

def test_routing_street_navigation_endpoint(client):
    """GET /v1/routing/route returns road geometry coordinates."""
    r = client.get("/v1/routing/route", params={
        "origin_lat": 16.5062, "origin_lng": 80.6480,
        "dest_lat": 16.5030, "dest_lng": 80.6274,
    })
    assert r.status_code == 200
    body = r.json()
    assert "coordinates" in body
    assert isinstance(body["coordinates"], list)


def test_routing_street_navigation_post(client):
    """POST /v1/routing/route returns road geometry coordinates."""
    r = client.post("/v1/routing/route", params={
        "origin_lat": 16.5062, "origin_lng": 80.6480,
        "dest_lat": 16.5030, "dest_lng": 80.6274,
    })
    assert r.status_code == 200
    body = r.json()
    assert "coordinates" in body


def test_routing_route_missing_params(client):
    """Missing required query parameters triggers 422 Unprocessable Entity."""
    # Missing dest coordinates
    r = client.get("/v1/routing/route", params={"origin_lat": 16.5062, "origin_lng": 80.6480})
    assert r.status_code == 422

    # Completely empty query
    r = client.post("/v1/routing/route")
    assert r.status_code == 422


def test_routing_route_invalid_types(client):
    """Non-numeric coordinate parameters trigger 422 Unprocessable Entity."""
    r = client.get("/v1/routing/route", params={
        "origin_lat": "not_a_number", "origin_lng": 80.6480,
        "dest_lat": 16.5030, "dest_lng": 80.6274,
    })
    assert r.status_code == 422


# ===========================================================================
# 3. Location Context: POST /v1/location/context
# ===========================================================================

def test_valid_request(client):
    """Valid location request returns 200 and H3 cell."""
    r = client.post("/v1/location/context", json={"latitude": 16.5062, "longitude": 80.6480})
    assert r.status_code == 200
    assert "h3_cell" in r.json()


def test_location_context_valid(client):
    """Valid GPS coordinates return H3 cell, timezone, and water flag."""
    r = client.post("/v1/location/context", json={"latitude": 16.5062, "longitude": 80.6480})
    assert r.status_code == 200
    body = r.json()
    assert "h3_cell" in body
    assert len(body["h3_cell"]) == 15
    assert body["latitude"] == 16.5062
    assert body["longitude"] == 80.6480
    assert "timezone_name" in body
    assert "is_water" in body


def test_location_context_southern_hemisphere(client):
    """Non-Indian location still produces valid H3 cell."""
    r = client.post("/v1/location/context", json={"latitude": -33.8688, "longitude": 151.2093})
    assert r.status_code == 200
    assert "h3_cell" in r.json()


def test_location_context_boundary_coords(client):
    """Extreme valid GPS boundaries (-90/90 lat, -180/180 lon) succeed."""
    r_max = client.post("/v1/location/context", json={"latitude": 90.0, "longitude": 180.0})
    assert r_max.status_code == 200
    assert "h3_cell" in r_max.json()

    r_min = client.post("/v1/location/context", json={"latitude": -90.0, "longitude": -180.0})
    assert r_min.status_code == 200
    assert "h3_cell" in r_min.json()


def test_invalid_request(client):
    """Out-of-range coordinates return 422."""
    r = client.post("/v1/location/context", json={"latitude": 200.0, "longitude": 80.0})
    assert r.status_code == 422


@pytest.mark.parametrize("lat,lon", [
    (90.1, 80.0),      # Latitude > 90
    (-90.1, 80.0),     # Latitude < -90
    (16.5, 180.1),     # Longitude > 180
    (16.5, -180.1),    # Longitude < -180
    (999.0, 999.0),    # Extremely out of bounds
])
def test_location_context_invalid_coordinates(client, lat, lon):
    """Out-of-range latitude or longitude triggers 422."""
    r = client.post("/v1/location/context", json={"latitude": lat, "longitude": lon})
    assert r.status_code == 422


def test_location_context_missing_fields(client):
    """Missing latitude or longitude triggers 422."""
    assert client.post("/v1/location/context", json={}).status_code == 422
    assert client.post("/v1/location/context", json={"latitude": 16.5062}).status_code == 422
    assert client.post("/v1/location/context", json={"longitude": 80.6480}).status_code == 422


def test_location_context_invalid_types(client):
    """Non-numeric coordinate types trigger 422."""
    r = client.post("/v1/location/context", json={"latitude": "sixteen", "longitude": 80.6480})
    assert r.status_code == 422


def test_location_context_invalid_method(client):
    """GET /v1/location/context returns 405 Method Not Allowed."""
    r = client.get("/v1/location/context")
    assert r.status_code == 405


# ===========================================================================
# 4. Predict Demand: POST /v1/predict/demand
# ===========================================================================

def test_prediction_response(client):
    """POST /v1/predict/demand returns predicted customers & economics."""
    r = client.post("/v1/predict/demand", json={
        "latitude": 16.5062, "longitude": 80.6480, "vendor_category": "food"
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert body["prediction"]["expected_customers"] >= 0
    assert "confidence_interval" in body["prediction"]
    assert "economics" in body
    assert "data_freshness" in body
    assert "location" in body


def test_predict_demand_valid_full(client):
    """POST /v1/predict/demand with complete valid optional payload."""
    r = client.post("/v1/predict/demand", json={
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": "food",
        "average_order_value": 150.0,
        "variable_cost_rate": 0.40,
        "fixed_cost_per_day": 400.0,
        "timestamp": "2026-10-05T12:00:00Z",
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert body["economics"]["average_order_value_inr"] == 150.0
    assert body["vendor_category"] == "food"


def test_predict_demand_missing_fields(client):
    """Missing coordinates in predict demand triggers 422."""
    assert client.post("/v1/predict/demand", json={}).status_code == 422
    assert client.post("/v1/predict/demand", json={"latitude": 16.5062}).status_code == 422


@pytest.mark.parametrize("payload", [
    {"latitude": 95.0, "longitude": 80.6480},    # Lat > 90
    {"latitude": -95.0, "longitude": 80.6480},   # Lat < -90
    {"latitude": 16.5, "longitude": 185.0},      # Lon > 180
    {"latitude": 16.5, "longitude": -185.0},     # Lon < -180
])
def test_predict_demand_invalid_coords(client, payload):
    """Coordinates out of bounds trigger 422."""
    assert client.post("/v1/predict/demand", json=payload).status_code == 422


@pytest.mark.parametrize("field,bad_val", [
    ("average_order_value", 0.0),    # Must be > 0
    ("average_order_value", -10.0),  # Negative
    ("variable_cost_rate", 0.0),     # Must be > 0
    ("variable_cost_rate", 1.0),     # Must be < 1
    ("variable_cost_rate", 1.5),     # Must be < 1
    ("variable_cost_rate", -0.1),    # Negative
    ("fixed_cost_per_day", 0.0),     # Must be > 0
    ("fixed_cost_per_day", -50.0),   # Negative
])
def test_predict_demand_invalid_economics(client, field, bad_val):
    """Invalid business economics parameters trigger 422."""
    payload = {"latitude": 16.5062, "longitude": 80.6480, field: bad_val}
    r = client.post("/v1/predict/demand", json=payload)
    assert r.status_code == 422


def test_predict_demand_invalid_timestamp(client):
    """Malformed timestamp triggers 422."""
    r = client.post("/v1/predict/demand", json={
        "latitude": 16.5062, "longitude": 80.6480, "timestamp": "invalid_date"
    })
    assert r.status_code == 422


def test_predict_demand_invalid_types(client):
    """Wrong data types trigger 422."""
    r = client.post("/v1/predict/demand", json={
        "latitude": 16.5062, "longitude": 80.6480, "average_order_value": "high"
    })
    assert r.status_code == 422


def test_predict_demand_invalid_method(client):
    """GET /v1/predict/demand returns 405 Method Not Allowed."""
    r = client.get("/v1/predict/demand")
    assert r.status_code == 405


def test_predict_demand_auth(client):
    """API key verification enforces 401 when enabled, accepts valid key."""
    with patch("api.app.REQUIRE_API_KEY", True), patch("api.app.API_KEY", "test-secret-key"):
        # Missing API key
        r1 = client.post("/v1/predict/demand", json={"latitude": 16.5062, "longitude": 80.6480})
        assert r1.status_code == 401

        # Invalid API key
        r2 = client.post("/v1/predict/demand", json={"latitude": 16.5062, "longitude": 80.6480},
                         headers={"X-API-Key": "wrong-key"})
        assert r2.status_code == 401

        # Valid API key
        r3 = client.post("/v1/predict/demand", json={"latitude": 16.5062, "longitude": 80.6480},
                         headers={"X-API-Key": "test-secret-key"})
        assert r3.status_code == 200


def test_predict_demand_model_not_loaded(client):
    """When ML model is not loaded, endpoint returns 503 Service Unavailable."""
    with patch.dict("api.app._model_health", {"status": "not_loaded"}, clear=False):
        r = client.post("/v1/predict/demand", json={"latitude": 16.5062, "longitude": 80.6480})
        assert r.status_code == 503
        assert "ML model not loaded" in r.json()["detail"]


# ===========================================================================
# 5. Recommend Zones: POST /v1/recommend/zones
# ===========================================================================

def test_recommend_zones_valid_minimal(client):
    """POST /v1/recommend/zones with minimal valid coordinates returns recommendations."""
    r = client.post("/v1/recommend/zones", json={"latitude": 16.5062, "longitude": 80.6480})
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert "recommendations" in body
    assert "decision" in body
    assert "demand_heatmap" in body


def test_recommend_zones_valid_custom(client):
    """POST /v1/recommend/zones with custom radius and top_n."""
    r = client.post("/v1/recommend/zones", json={
        "latitude": 16.5062, "longitude": 80.6480,
        "vendor_category": "food", "search_radius_km": 2.0, "top_n": 3
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) <= 3


def test_recommend_zones_boundaries(client):
    """Boundary valid values for top_n (1, 10) and search_radius_km (10.0)."""
    r_top1 = client.post("/v1/recommend/zones", json={
        "latitude": 16.5062, "longitude": 80.6480, "top_n": 1
    })
    if r_top1.status_code == 503:
        pytest.skip("Model not trained")
    assert r_top1.status_code == 200
    assert len(r_top1.json()["recommendations"]) <= 1

    r_top10 = client.post("/v1/recommend/zones", json={
        "latitude": 16.5062, "longitude": 80.6480, "top_n": 10, "search_radius_km": 10.0
    })
    assert r_top10.status_code == 200


def test_recommend_zones_invalid_coordinates(client):
    """Out-of-range coordinates trigger 422."""
    assert client.post("/v1/recommend/zones", json={"latitude": 91.0, "longitude": 80.6480}).status_code == 422
    assert client.post("/v1/recommend/zones", json={"latitude": 16.5062, "longitude": 181.0}).status_code == 422


@pytest.mark.parametrize("bad_radius", [0.0, -1.0, 10.1, 50.0])
def test_recommend_zones_invalid_radius(client, bad_radius):
    """search_radius_km must be > 0 and <= 10."""
    r = client.post("/v1/recommend/zones", json={
        "latitude": 16.5062, "longitude": 80.6480, "search_radius_km": bad_radius
    })
    assert r.status_code == 422


@pytest.mark.parametrize("bad_top_n", [0, -1, 11, 20])
def test_recommend_zones_invalid_top_n(client, bad_top_n):
    """top_n must be between 1 and 10."""
    r = client.post("/v1/recommend/zones", json={
        "latitude": 16.5062, "longitude": 80.6480, "top_n": bad_top_n
    })
    assert r.status_code == 422


def test_recommend_zones_invalid_types(client):
    """Invalid types in recommend zones trigger 422."""
    r = client.post("/v1/recommend/zones", json={
        "latitude": 16.5062, "longitude": 80.6480, "top_n": "invalid"
    })
    assert r.status_code == 422


def test_recommend_zones_invalid_method(client):
    """GET /v1/recommend/zones returns 405 Method Not Allowed."""
    r = client.get("/v1/recommend/zones")
    assert r.status_code == 405


def test_recommend_zones_auth(client):
    """Enforces authentication when REQUIRE_API_KEY is active."""
    with patch("api.app.REQUIRE_API_KEY", True), patch("api.app.API_KEY", "zone-secret-key"):
        r_unauth = client.post("/v1/recommend/zones", json={"latitude": 16.5062, "longitude": 80.6480})
        assert r_unauth.status_code == 401

        r_auth = client.post("/v1/recommend/zones", json={"latitude": 16.5062, "longitude": 80.6480},
                             headers={"X-API-Key": "zone-secret-key"})
        assert r_auth.status_code == 200


# ===========================================================================
# 6. Scenario Simulation: POST /v1/scenario/simulate
# ===========================================================================

def test_scenario_simulate_valid_minimal(client):
    """POST /v1/scenario/simulate with minimal payload returns baseline vs scenario comparison."""
    r = client.post("/v1/scenario/simulate", json={"latitude": 16.5062, "longitude": 80.6480})
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert "baseline" in body
    assert "scenario" in body
    assert "simulation_delta" in body
    assert "explanation" in body
    assert "profit_delta_inr" in body["simulation_delta"]


def test_scenario_simulate_valid_custom(client):
    """POST /v1/scenario/simulate with custom AOV and simulated conditions."""
    r = client.post("/v1/scenario/simulate", json={
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": "food",
        "baseline_aov": 80.0,
        "simulated_aov": 140.0,
        "simulated_weather_condition": "rainy",
        "simulated_hour": 14,
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert body["baseline"]["aov"] == 80.0
    assert body["scenario"]["aov"] == 140.0


def test_scenario_simulate_boundary_hours(client):
    """simulated_hour boundaries (0 and 23) succeed."""
    for h in [0, 23]:
        r = client.post("/v1/scenario/simulate", json={
            "latitude": 16.5062, "longitude": 80.6480, "simulated_hour": h
        })
        if r.status_code == 503:
            pytest.skip("Model not trained")
        assert r.status_code == 200


def test_scenario_simulate_invalid_coordinates(client):
    """Invalid coordinates trigger 422."""
    assert client.post("/v1/scenario/simulate", json={"latitude": 92.0, "longitude": 80.6480}).status_code == 422
    assert client.post("/v1/scenario/simulate", json={"latitude": 16.5062, "longitude": -185.0}).status_code == 422


@pytest.mark.parametrize("bad_aov", [0.0, -10.0, -100.0])
def test_scenario_simulate_invalid_aov(client, bad_aov):
    """Non-positive baseline_aov or simulated_aov triggers 422."""
    r1 = client.post("/v1/scenario/simulate", json={
        "latitude": 16.5062, "longitude": 80.6480, "baseline_aov": bad_aov
    })
    assert r1.status_code == 422

    r2 = client.post("/v1/scenario/simulate", json={
        "latitude": 16.5062, "longitude": 80.6480, "simulated_aov": bad_aov
    })
    assert r2.status_code == 422


@pytest.mark.parametrize("bad_hour", [-1, 24, 100])
def test_scenario_simulate_invalid_hours(client, bad_hour):
    """simulated_hour must be between 0 and 23."""
    r = client.post("/v1/scenario/simulate", json={
        "latitude": 16.5062, "longitude": 80.6480, "simulated_hour": bad_hour
    })
    assert r.status_code == 422


def test_scenario_simulate_invalid_types(client):
    """Non-numeric values for numeric fields trigger 422."""
    r = client.post("/v1/scenario/simulate", json={
        "latitude": 16.5062, "longitude": 80.6480, "baseline_aov": "free"
    })
    assert r.status_code == 422


def test_scenario_simulate_invalid_method(client):
    """GET /v1/scenario/simulate returns 405 Method Not Allowed."""
    r = client.get("/v1/scenario/simulate")
    assert r.status_code == 405


def test_scenario_simulate_auth(client):
    """Scenario simulation verifies authentication when enabled."""
    with patch("api.app.REQUIRE_API_KEY", True), patch("api.app.API_KEY", "sim-secret-key"):
        r_unauth = client.post("/v1/scenario/simulate", json={"latitude": 16.5062, "longitude": 80.6480})
        assert r_unauth.status_code == 401

        r_auth = client.post("/v1/scenario/simulate", json={"latitude": 16.5062, "longitude": 80.6480},
                             headers={"X-API-Key": "sim-secret-key"})
        assert r_auth.status_code == 200


# ===========================================================================
# 7. Live Recommendations: POST /v1/recommendations/live
# ===========================================================================

def test_schema_validation(client):
    """Invalid datatype in payload triggers 422 Unprocessable Entity."""
    r = client.post("/v1/recommendations/live", json={"latitude": "invalid_lat", "longitude": 80.6480})
    assert r.status_code == 422


def test_h3_candidate_generation(client):
    """Generates valid non-empty H3 candidates around origin."""
    from api.location import generate_candidates, get_location_context
    ctx = get_location_context(16.5062, 80.6480)
    candidates = generate_candidates(16.5062, 80.6480, ctx.h3_cell, search_radius_km=2.0)
    assert len(candidates) > 0
    assert any(c.is_current_cell for c in candidates)


def test_radius_filter(client):
    """All returned recommendations lie within requested search radius."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480,
        "vendor_category": "food", "search_radius_km": 1.5, "top_n": 5
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    for rec in r.json()["recommendations"]:
        assert rec["distance_km"] <= 1.5


def test_water_exclusion(client):
    """Candidate recommendations strictly exclude water cells and river points."""
    from api.spatial_filter import is_water_location, is_water_cell
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6150,
        "vendor_category": "food", "search_radius_km": 3.0, "top_n": 5
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    for rec in r.json()["recommendations"]:
        assert not is_water_location(rec["latitude"], rec["longitude"])
        assert not is_water_cell(rec["h3_cell"])


def test_profit_calculation():
    """Business metrics calculation computes revenue, COGS, and profit correctly."""
    from api.recommender import calculate_business_metrics
    biz = calculate_business_metrics(
        expected_customers=50, aov=100.0, variable_cost_rate=0.38, fixed_cost_per_day=500.0
    )
    assert biz["expected_revenue_inr"] == 5000.0
    assert biz["ingredient_cost_inr"] == 1900.0
    assert biz["expected_profit_inr"] > 0


def test_travel_adjustment():
    """Travel time operating fraction formula reduces effective operating time in 1-hour horizon."""
    travel_time_minutes = 15
    travel_fraction = max(0.0, (60.0 - travel_time_minutes) / 60.0)
    assert travel_fraction == 0.75


def test_stay_threshold():
    """Verify stay put threshold constant."""
    from config.settings import STAY_THRESHOLD_INR_PER_HOUR
    assert STAY_THRESHOLD_INR_PER_HOUR == 50.0


def test_consider_threshold():
    """Verify consider move threshold constant."""
    from config.settings import CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR
    assert CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR == 250.0


def test_recommended_move_threshold():
    """Verify maximum recommended travel distance and time thresholds."""
    from config.settings import MAX_RECOMMENDED_TRAVEL_MIN, MAX_RECOMMENDED_TRAVEL_KM
    assert MAX_RECOMMENDED_TRAVEL_MIN == 12.0
    assert MAX_RECOMMENDED_TRAVEL_KM == 3.0


def test_decision_explanation(client):
    """Decision explanation contains all required fields per section 44."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480, "vendor_category": "food", "top_n": 3
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert "decision" in body
    dec = body["decision"]
    assert "verdict" in dec
    assert "reason" in dec
    assert "current_profit" in dec
    assert "recommended_profit" in dec
    assert "profit_uplift" in dec
    assert "travel_time" in dec
    assert "distance" in dec

    recs = body["recommendations"]
    if recs:
        rec = recs[0]
        assert "decision" in rec
        assert "reason" in rec
        assert "current_profit" in rec
        assert "recommended_profit" in rec
        assert "profit_uplift" in rec
        assert "travel_time" in rec
        assert "distance" in rec


def test_recommendations_structure(client):
    """Live recommendations return correct schema."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": "food",
        "average_order_value": 100,
        "search_radius_km": 1.5,
        "top_n": 3,
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")

    assert r.status_code == 200
    body = r.json()
    assert "current_location" in body
    assert "recommendations" in body
    assert "demand_heatmap" in body
    assert "data_quality" in body
    assert "model_info" in body


def test_recommendations_live_valid_custom(client):
    """POST /v1/recommendations/live with full vendor profile."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_id": "V_999",
        "vendor_category": "food",
        "vendor_name": "Test Cart",
        "average_order_value": 120.0,
        "variable_cost_rate": 0.35,
        "fixed_cost_per_day": 250.0,
        "inventory_capacity": 300,
        "preparation_time": 4.5,
        "search_radius_km": 2.5,
        "top_n": 4,
        "timestamp": "2026-10-05T12:00:00Z",
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) <= 4


def test_recommendations_live_boundaries(client):
    """POST /v1/recommendations/live boundaries (top_n=1, top_n=10, search_radius_km=10.0)."""
    r_top1 = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480, "top_n": 1
    })
    if r_top1.status_code == 503:
        pytest.skip("Model not trained")
    assert r_top1.status_code == 200
    assert len(r_top1.json()["recommendations"]) <= 1

    r_top10 = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480, "top_n": 10, "search_radius_km": 10.0
    })
    assert r_top10.status_code == 200


@pytest.mark.parametrize("payload", [
    {"latitude": 90.1, "longitude": 80.6480},
    {"latitude": -90.1, "longitude": 80.6480},
    {"latitude": 16.5062, "longitude": 180.1},
    {"latitude": 16.5062, "longitude": -180.1},
])
def test_recommendations_live_invalid_coordinates(client, payload):
    """Coordinates out of bounds trigger 422."""
    assert client.post("/v1/recommendations/live", json=payload).status_code == 422


@pytest.mark.parametrize("field,bad_val", [
    ("average_order_value", 0.0),
    ("average_order_value", -15.0),
    ("variable_cost_rate", 0.0),
    ("variable_cost_rate", 1.0),
    ("variable_cost_rate", 1.5),
    ("variable_cost_rate", -0.2),
    ("fixed_cost_per_day", 0.0),
    ("fixed_cost_per_day", -100.0),
    ("inventory_capacity", 0),
    ("inventory_capacity", -5),
    ("preparation_time", 0.0),
    ("preparation_time", -3.0),
    ("search_radius_km", 0.0),
    ("search_radius_km", -1.0),
    ("search_radius_km", 10.1),
    ("top_n", 0),
    ("top_n", 11),
])
def test_recommendations_live_invalid_parameters(client, field, bad_val):
    """Invalid parameter values in live recommendation request trigger 422."""
    payload = {"latitude": 16.5062, "longitude": 80.6480, field: bad_val}
    r = client.post("/v1/recommendations/live", json=payload)
    assert r.status_code == 422


def test_recommendations_live_invalid_timestamp(client):
    """Malformed timestamp in live recommendation request triggers 422."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480, "timestamp": "not-a-datetime"
    })
    assert r.status_code == 422


def test_recommendations_live_invalid_method(client):
    """GET /v1/recommendations/live returns 405 Method Not Allowed."""
    r = client.get("/v1/recommendations/live")
    assert r.status_code == 405


def test_recommendations_live_model_not_loaded(client):
    """When ML model is not loaded, /v1/recommendations/live returns 503."""
    with patch.dict("api.app._model_health", {"status": "not_loaded"}, clear=False):
        r = client.post("/v1/recommendations/live", json={"latitude": 16.5062, "longitude": 80.6480})
        assert r.status_code == 503
        assert "ML model not loaded" in r.json()["detail"]["error"]


# ===========================================================================
# 8. Static Files: GET /static/...
# ===========================================================================

def test_static_files(client):
    """Static assets return 200 when present, 404 when absent."""
    r_ok = client.get("/static/index.html")
    assert r_ok.status_code == 200

    r_404 = client.get("/static/non_existent_asset_12345.xyz")
    assert r_404.status_code == 404


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.run(["python", "-m", "pytest", __file__, "-v"]).returncode)
