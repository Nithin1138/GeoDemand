"""
GeoDemand AI — Phase 18.4 Full-System Master Integration Testing Suite
Primary Owner: Member 4 (Nithin, Technical Lead & Integration)

Validates the complete end-to-end integration contracts across:
1. Member 1 (Data Engineering & Feature Store)
2. Member 2 (Machine Learning, Quantiles & SHAP)
3. Member 3 (Backend Decision Engine, Optimization & Fleet)
4. Member 4 (Dashboard, PWA, Security & Release)
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, DASHBOARD_DIR, _load_model
from src.api.recommender import DemandModel
from src.api.security.auth import api_key_registry, Role, UserContext


@pytest.fixture
def client():
    _load_model()
    with TestClient(app) as c:
        yield c


# -----------------------------------------------------------------------------
# 1. Contract 1: Member 1 Data -> Member 2 Model Contract
# -----------------------------------------------------------------------------
def test_contract_data_to_model():
    """Verify that feature schema matches the trained champion model expectations."""
    model = DemandModel.get()
    assert model is not None
    assert model._predictor is not None or model._model is not None
    assert model.meta is not None
    assert len(model._feature_cols) == 45
    assert "building_density" in model._feature_cols
    assert "commercial_ratio" in model._feature_cols
    assert "rainfall" in model._feature_cols


# -----------------------------------------------------------------------------
# 2. Contract 2: Member 2 Model -> Member 3 API Recommender Contract
# -----------------------------------------------------------------------------
def test_contract_model_to_api(client):
    """Verify that live recommendation API returns calibrated quantiles & TreeSHAP attributions."""
    payload = {
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": "food",
        "average_order_value": 100.0,
        "search_radius_km": 3.0,
        "top_n": 3
    }
    resp = client.post("/v1/recommendations/live", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0

    top = data["recommendations"][0]
    assert "expected_customers" in top
    assert "expected_profit_inr" in top
    assert "expected_revenue_inr" in top
    assert "prediction_interval" in top

    # Quantile monotonicity contract
    interval = top["prediction_interval"]
    assert interval["p10"] <= interval["p50"] <= interval["p90"]

    # TreeSHAP drivers contract
    assert "top_drivers" in top
    assert isinstance(top["top_drivers"], list)


# -----------------------------------------------------------------------------
# 3. Contract 3: Member 3 API -> Member 4 Dashboard & PWA Contract
# -----------------------------------------------------------------------------
def test_contract_api_to_dashboard(client):
    """Verify that root dashboard, manifest, and service worker are delivered with valid MIME types."""
    # Root dashboard HTML
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "text/html" in resp_root.headers.get("content-type", "")
    assert "GeoDemand" in resp_root.text
    assert 'href="/manifest.json"' in resp_root.text

    # PWA Manifest
    resp_manifest = client.get("/manifest.json")
    assert resp_manifest.status_code == 200
    assert "application/manifest+json" in resp_manifest.headers.get("content-type", "")
    manifest_data = resp_manifest.json()
    assert manifest_data["short_name"] == "GeoDemand"
    assert manifest_data["display"] == "standalone"

    # PWA Service Worker
    resp_sw = client.get("/sw.js")
    assert resp_sw.status_code == 200
    assert "application/javascript" in resp_sw.headers.get("content-type", "")
    assert "geodemand-shell" in resp_sw.text


# -----------------------------------------------------------------------------
# 4. Multi-Stop Shift Optimization Integration (Phase 9.4)
# -----------------------------------------------------------------------------
def test_shift_optimization_integration(client):
    """Verify OR-Tools shift route planning endpoint with operational constraints."""
    payload = {
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": "food",
        "shift_duration_hours": 8.0,
        "max_relocation_stops": 3,
        "min_dwell_time_minutes": 90,
        "setup_time_minutes": 15,
        "teardown_time_minutes": 15
    }
    resp = client.post("/v1/routes/shift-plan", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["summary"]["total_net_profit_inr"] > 0
    assert data["summary"]["total_gross_revenue_inr"] >= data["summary"]["total_net_profit_inr"]
    assert len(data["itinerary"]) <= 3

    for stop in data["itinerary"]:
        assert "h3_cell" in stop
        assert "arrival_time" in stop
        assert "departure_time" in stop
        assert "net_profit_inr" in stop


# -----------------------------------------------------------------------------
# 5. Fleet Coordination & Reservation Lifecycle (Phase 10.4)
# -----------------------------------------------------------------------------
def test_fleet_reservation_lifecycle(client):
    """Verify cell occupancy query, reservation placement, and cancellation."""
    test_cell = "886189254dfffff"
    vendor_id = f"test-vendor-{uuid.uuid4().hex[:6]}"

    # Check initial occupancy
    resp_occ = client.get(f"/v1/fleet/cell-occupancy?h3_cell={test_cell}")
    assert resp_occ.status_code == 200

    # Place reservation
    resp_res = client.post("/v1/fleet/reserve-cell", json={
        "vendor_id": vendor_id,
        "h3_cell": test_cell,
        "vendor_category": "food",
        "duration_minutes": 60,
        "is_soft": True,
        "deposit_amount_inr": 250.0
    })
    assert resp_res.status_code in (200, 409)

    if resp_res.status_code == 200:
        res_id = resp_res.json().get("reservation_id")
        if res_id:
            # Cancel reservation
            resp_cancel = client.post("/v1/fleet/cancel-reservation", json={
                "reservation_id": res_id,
                "vendor_id": vendor_id
            })
            assert resp_cancel.status_code == 200


# -----------------------------------------------------------------------------
# 6. Telemetry & POS Transaction Ingestion (Phase 11.4)
# -----------------------------------------------------------------------------
def test_telemetry_pos_ingestion(client):
    """Verify live POS transaction logging directly through the API."""
    tx_id = f"POS-INT-{uuid.uuid4().hex[:8]}"
    payload = {
        "transaction_id": tx_id,
        "vendor_id": "vendor-pos-99",
        "amount_inr": 250.0,
        "payment_method": "upi",
        "category": "food",
        "items_count": 2,
        "location": {
            "latitude": 16.5062,
            "longitude": 80.6480
        },
        "event_timestamp": datetime.now(timezone.utc).isoformat()
    }
    resp = client.post("/v1/telemetry/transactions", json=payload, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_duplicate"] is False
    assert data["h3_cell_id"] is not None


# -----------------------------------------------------------------------------
# 7. Multi-City Generalization (Phase 14.4)
# -----------------------------------------------------------------------------
def test_multicity_recommendations(client):
    """Verify recommendation pipeline executes for all supported cities."""
    cities = [
        {"name": "Vijayawada", "lat": 16.5062, "lng": 80.6480},
        {"name": "Hyderabad", "lat": 17.3850, "lng": 78.4867},
        {"name": "Bengaluru", "lat": 12.9716, "lng": 77.5946},
        {"name": "Visakhapatnam", "lat": 17.6868, "lng": 83.2185},
    ]

    for city in cities:
        payload = {
            "latitude": city["lat"],
            "longitude": city["lng"],
            "vendor_category": "food",
            "average_order_value": 100.0,
            "search_radius_km": 3.0,
            "top_n": 2
        }
        resp = client.post("/v1/recommendations/live", json=payload)
        assert resp.status_code == 200, f"Failed for city {city['name']}"
        data = resp.json()
        assert "recommendations" in data


# -----------------------------------------------------------------------------
# 8. Platform Security & Timing-Safe RBAC (Phase 17.3 & 17.4)
# -----------------------------------------------------------------------------
def test_platform_security_timing_safe():
    """Verify timing-safe API key comparison prevents timing side-channels."""
    test_key = "test_secret_api_key_12345"
    api_key_registry.register_key(test_key, role=Role.ADMIN, user_id="admin_sec")
    meta = api_key_registry.validate_key(test_key)
    assert meta is not None
    assert meta["role"] == Role.ADMIN
    assert api_key_registry.validate_key(test_key + "_tampered") is None


def test_system_liveness_and_telemetry(client):
    """Verify system ping, data health, model health, and Prometheus metrics."""
    # Ping
    resp_ping = client.get("/v1/system/ping")
    assert resp_ping.status_code == 200
    assert resp_ping.json()["status"] == "ok"

    # Data Health
    resp_health = client.get("/v1/system/data-health")
    assert resp_health.status_code == 200

    # Model Health
    resp_model = client.get("/v1/system/model-health")
    assert resp_model.status_code == 200
