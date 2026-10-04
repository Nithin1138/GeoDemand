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


# ── Location context tests ──

def test_location_context_valid(client):
    """Valid GPS → H3 response."""
    r = client.post("/v1/location/context", json={"latitude": 16.5062, "longitude": 80.6480})
    assert r.status_code == 200
    body = r.json()
    assert "h3_cell" in body
    assert len(body["h3_cell"]) == 15  # H3 res 8 cell ID length
    assert body["source_type"] == "real_live"
    assert body["source_name"] == "device_gps"


def test_location_context_invalid(client):
    """Out-of-range coordinates return 422."""
    r = client.post("/v1/location/context", json={"latitude": 200.0, "longitude": 80.0})
    assert r.status_code == 422


def test_location_context_southern_hemisphere(client):
    """Non-Indian location still produces valid H3 cell."""
    r = client.post("/v1/location/context", json={"latitude": -33.8688, "longitude": 151.2093})
    assert r.status_code == 200
    assert "h3_cell" in r.json()


# ── Recommendations tests ──

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
        pytest.skip("Model not trained — run train_demand_model.py first")

    assert r.status_code == 200
    body = r.json()

    # Top-level fields
    assert "current_location" in body
    assert "recommendations" in body
    assert "data_freshness" in body
    assert "model_info" in body
    assert "total_candidates_evaluated" in body

    # Current location
    assert "h3_cell" in body["current_location"]

    # Recommendations
    recs = body["recommendations"]
    assert len(recs) <= 3
    if recs:
        rec = recs[0]
        assert "rank" in rec
        assert "h3_cell" in rec
        assert "expected_customers" in rec
        assert "expected_profit_inr" in rec
        assert "confidence_interval" in rec
        ci = rec["confidence_interval"]
        assert "p10" in ci and "p50" in ci and "p90" in ci
        assert ci["p10"] <= ci["p50"] <= ci["p90"]
        assert "top_drivers" in rec
        assert "explanation" in rec
        assert "competition_level" in rec
        assert rec["competition_level"] in ("low", "medium", "high")
        assert "distance_km" in rec
        assert rec["distance_km"] >= 0


def test_recommendations_provenance(client):
    """Every recommendation response has explict source_type tags."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480,
        "vendor_category": "food",
        "search_radius_km": 1.0, "top_n": 1,
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    body = r.json()
    freshness = body["data_freshness"]
    assert "weather" in freshness
    weather_freshness = freshness["weather"]
    assert "source_type" in weather_freshness
    # Should never be an unlabeled source
    assert weather_freshness["source_type"] != ""


def test_recommendations_no_negative_customers(client):
    """ML model should never predict negative customers."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480,
        "vendor_category": "food",
        "search_radius_km": 2.0, "top_n": 5,
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    for rec in r.json()["recommendations"]:
        assert rec["expected_customers"] >= 0
        ci = rec["confidence_interval"]
        assert ci["p10"] >= 0


def test_recommendations_profit_improvement(client):
    """Profit improvement field exists and is correctly signed."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480,
        "vendor_category": "food",
        "search_radius_km": 3.0, "top_n": 5,
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")
    body = r.json()
    for rec in body["recommendations"]:
        if rec.get("profit_improvement_inr") is not None:
            # pct and inr should have the same sign
            pct = rec.get("profit_improvement_pct", 0)
            inr = rec.get("profit_improvement_inr", 0)
            assert (pct >= 0) == (inr >= 0), f"Sign mismatch: {pct} vs {inr}"


def test_no_recommendations_in_water_body(client):
    """Test that candidate destinations never fall into Krishna River / water bodies and have positive expected profit."""
    from api.spatial_filter import is_water_location, is_water_cell

    # Test point right at Prakasam Barrage / river
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6150,
        "vendor_category": "food",
        "search_radius_km": 3.0, "top_n": 5,
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")

    assert r.status_code == 200
    body = r.json()
    recs = body["recommendations"]
    assert len(recs) > 0

    for rec in recs:
        in_water_pt = is_water_location(rec["latitude"], rec["longitude"])
        in_water_cell_bnd = is_water_cell(rec["h3_cell"])
        assert not in_water_pt, f"Recommendation cell {rec['h3_cell']} at ({rec['latitude']}, {rec['longitude']}) is in water point!"
        assert not in_water_cell_bnd, f"Recommendation cell {rec['h3_cell']} overlaps water cell boundary!"
        assert not rec.get("is_water", False)
        assert rec.get("expected_profit_inr", 0) > 0, f"Recommendation {rec['h3_cell']} has invalid non-positive profit!"



def test_recommendations_strategic_decision(client):
    """Decision intelligence returns structured stay/move guidance with explicit rationale."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062, "longitude": 80.6480,
        "vendor_category": "tea_coffee",
        "search_radius_km": 2.0, "top_n": 3,
    })
    if r.status_code == 503:
        pytest.skip("Model not trained")

    assert r.status_code == 200
    body = r.json()
    assert "decision" in body
    decision = body["decision"]
    assert decision["verdict"] in ("STAY_PUT", "CONSIDER_MOVE", "RECOMMENDED_MOVE")
    assert decision["action"] in ("STAY HERE", "CONSIDER MOVE", "RECOMMENDED MOVE")
    assert "headline" in decision
    assert "rationale" in decision

    for rec in body["recommendations"]:
        assert "decision_verdict" in rec
        assert "decision_action" in rec
        assert "decision_rationale" in rec


def test_routing_street_navigation_endpoint(client):
    """Test /v1/routing/route endpoint returns real road geometry coordinates."""
    r = client.get("/v1/routing/route", params={
        "origin_lat": 16.5062, "origin_lng": 80.6480,
        "dest_lat": 16.5030, "dest_lng": 80.6274,
    })
    assert r.status_code == 200
    body = r.json()
    assert "coordinates" in body
    assert len(body["coordinates"]) > 1
    assert "distance_km" in body
    assert "duration_min" in body


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.run(["python3", "-m", "pytest", __file__, "-v"]).returncode)

