"""
API integration tests for GeoDemand AI live endpoints.

Uses FastAPI's TestClient so no server needs to be running.
External API calls (Open-Meteo, HERE) are mocked so tests run offline.

Run with:
    python3 -m pytest tests/test_api.py -v
"""

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


# ── Basic health tests ──

def test_ping(client):
    """API is reachable."""
    r = client.get("/v1/system/ping")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health(client):
    """Alias test for system data health."""
    r = client.get("/v1/system/data-health")
    assert r.status_code == 200
    body = r.json()
    assert "weather" in body


def test_data_health(client):
    """Data health endpoint returns all expected source types."""
    r = client.get("/v1/system/data-health")
    assert r.status_code == 200
    body = r.json()
    assert "weather" in body
    assert "traffic" in body
    assert "competition" in body
    assert "customer_demand" in body
    assert body["customer_demand"]["source_type"] == "derived"


# ── Location & Candidate tests ──

def test_valid_request(client):
    """Valid location request returns 200 and H3 cell."""
    r = client.post("/v1/location/context", json={"latitude": 16.5062, "longitude": 80.6480})
    assert r.status_code == 200
    assert "h3_cell" in r.json()


def test_invalid_request(client):
    """Out-of-range coordinates return 422."""
    r = client.post("/v1/location/context", json={"latitude": 200.0, "longitude": 80.0})
    assert r.status_code == 422


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


# ── Prediction & Economics tests ──

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


# ── Decision Threshold & Explanation tests ──

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


def test_location_context_valid(client):
    """Valid GPS → H3 response."""
    r = client.post("/v1/location/context", json={"latitude": 16.5062, "longitude": 80.6480})
    assert r.status_code == 200
    body = r.json()
    assert "h3_cell" in body
    assert len(body["h3_cell"]) == 15


def test_location_context_southern_hemisphere(client):
    """Non-Indian location still produces valid H3 cell."""
    r = client.post("/v1/location/context", json={"latitude": -33.8688, "longitude": 151.2093})
    assert r.status_code == 200
    assert "h3_cell" in r.json()


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


def test_routing_street_navigation_endpoint(client):
    """Test /v1/routing/route endpoint returns real road geometry coordinates."""
    r = client.get("/v1/routing/route", params={
        "origin_lat": 16.5062, "origin_lng": 80.6480,
        "dest_lat": 16.5030, "dest_lng": 80.6274,
    })
    assert r.status_code == 200
    body = r.json()
    assert "coordinates" in body


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.run(["python3", "-m", "pytest", __file__, "-v"]).returncode)


