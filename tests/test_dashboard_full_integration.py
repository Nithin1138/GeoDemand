"""
Tests verifying every single frontend fetch endpoint and payload used in dashboard/app.js.
Ensures zero 422, zero 400, zero 500 errors across all UI features.
"""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_endpoint_system_config(client):
    """Endpoint: GET /v1/system/config"""
    res = client.get("/v1/system/config")
    assert res.status_code == 200
    data = res.json()
    assert "mapbox_token" in data or "live_mode" in data or "default_lat" in data


def test_endpoint_system_data_health(client):
    """Endpoint: GET /v1/system/data-health"""
    res = client.get("/v1/system/data-health")
    assert res.status_code == 200
    data = res.json()
    assert "model" in data
    assert "weather" in data
    assert "poi" in data


def test_endpoint_system_model_health(client):
    """Endpoint: GET /v1/system/model-health"""
    res = client.get("/v1/system/model-health")
    assert res.status_code == 200
    data = res.json()
    assert "model_version" in data or "status" in data


def test_endpoint_recommendations_live(client):
    """Endpoint: POST /v1/recommendations/live"""
    payload = {
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": "food",
        "average_order_value": 85.0,
        "search_radius_km": 3.0,
        "top_n": 5,
    }
    res = client.post("/v1/recommendations/live", json=payload)
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    top = data["recommendations"][0]
    assert "h3_cell" in top
    assert "latitude" in top
    assert "longitude" in top
    assert "expected_profit_inr" in top or "expected_customers" in top or "net_profit_uplift_inr" in top


def test_endpoint_routing_route(client):
    """Endpoint: GET /v1/routing/route"""
    params = {
        "origin_lat": 16.5062,
        "origin_lng": 80.6480,
        "dest_lat": 16.5150,
        "dest_lng": 80.6350,
    }
    res = client.get("/v1/routing/route", params=params)
    assert res.status_code == 200
    data = res.json()
    assert "coordinates" in data
    assert "distance_km" in data
    assert "duration_min" in data


def test_endpoint_shift_plan(client):
    """Endpoint: POST /v1/routes/shift-plan (The exact UI payload from dashboard/app.js)"""
    payload = {
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": "food",
        "shift_start_time": datetime.now(timezone.utc).isoformat(),
        "shift_duration_hours": 8.0,
        "max_relocation_stops": 2,
        "min_dwell_time_minutes": 90,
        "setup_time_minutes": 15,
        "teardown_time_minutes": 15,
    }
    res = client.post("/v1/routes/shift-plan", json=payload)
    assert res.status_code == 200, f"Failed with {res.status_code}: {res.text}"
    data = res.json()
    assert "summary" in data
    assert "itinerary" in data
    summary = data["summary"]
    assert "total_net_profit_inr" in summary
    assert "total_gross_revenue_inr" in summary
    assert "total_travel_distance_km" in summary
    assert "total_transit_minutes" in summary


def test_endpoint_fleet_cell_occupancy(client):
    """Endpoint: GET /v1/fleet/cell-occupancy"""
    res = client.get("/v1/fleet/cell-occupancy", params={"h3_cell": "886189254dfffff"})
    assert res.status_code == 200
    data = res.json()
    assert "h3_cell" in data
    assert "total_capacity" in data
    assert "total_active_occupancy" in data
    assert "available_capacity" in data
    assert "is_at_capacity" in data


def test_endpoint_fleet_reserve_and_cancel(client):
    """Endpoint: POST /v1/fleet/reserve-cell and POST /v1/fleet/cancel-reservation"""
    reserve_payload = {
        "vendor_id": "vendor-test-001",
        "h3_cell": "886189254dfffff",
        "vendor_category": "food",
        "duration_minutes": 60,
        "is_soft": True,
        "deposit_amount_inr": 0.0,
    }
    res = client.post("/v1/fleet/reserve-cell", json=reserve_payload)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["status"] in ("CONFIRMED", "REJECTED")

    if res_data["status"] == "CONFIRMED":
        res_id = res_data["reservation_id"]
        cancel_payload = {
            "reservation_id": res_id,
            "vendor_id": "vendor-test-001",
            "reason": "vendor_cancellation",
        }
        cres = client.post("/v1/fleet/cancel-reservation", json=cancel_payload)
        assert cres.status_code == 200
        cdata = cres.json()
        assert cdata["status"] in ("CANCELLED", "SUCCESS")


def test_endpoint_telemetry_transaction(client):
    """Endpoint: POST /v1/telemetry/transactions"""
    payload = {
        "transaction_id": f"TX-TEST-{datetime.now().timestamp()}",
        "vendor_id": "vendor-001",
        "amount_inr": 150.0,
        "payment_method": "upi",
        "category": "food",
        "items_count": 1,
        "location": {
            "latitude": 16.5062,
            "longitude": 80.6480,
        },
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    res = client.post(
        "/v1/telemetry/transactions",
        headers={"Content-Type": "application/json", "X-API-Key": "geodemand-prod-live-key"},
        json=payload,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("ingested", "success", "ACCEPTED")
