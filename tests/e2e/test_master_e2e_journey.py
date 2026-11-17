"""
GeoDemand AI — End-to-End Master Vendor Lifecycle Journey Test
Primary Owner: Member 4 (Nithin, Technical Lead & Release Engineering)

Executes the complete commercial workflow:
1. System Health & Public Config Check
2. GPS Location Context Resolution
3. Real-Time Demand Forecast & Quantile Uncertainty
4. Candidate Location Generation & H3 Spatial Filtering
5. Unit Economics Calculation & Relocation Friction
6. Strategic Decision Guidance (STAY / CONSIDER / MOVE)
7. Full-Shift Multi-Stop Route Optimization
8. Fleet Reservation Placement
9. Ground-Truth POS Telemetry Ingestion
10. Model Drift & System Metric Verification
"""

from datetime import datetime, timezone
import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, _load_model


@pytest.fixture
def client():
    _load_model()
    with TestClient(app) as c:
        yield c


def test_complete_vendor_commercial_journey(client):
    # Step 1: System Health Verification
    health_resp = client.get("/v1/system/ping")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    config_resp = client.get("/v1/system/config")
    assert config_resp.status_code == 200

    # Step 2: Location Context Resolution
    lat, lng = 16.5062, 80.6480
    loc_resp = client.post("/v1/location/context", json={"latitude": lat, "longitude": lng})
    assert loc_resp.status_code == 200
    loc_data = loc_resp.json()
    assert "h3_cell" in loc_data

    # Step 3, 4, 5, 6: Live Recommendation Pipeline
    rec_payload = {
        "latitude": lat,
        "longitude": lng,
        "vendor_category": "food",
        "average_order_value": 110.0,
        "search_radius_km": 3.0,
        "top_n": 5
    }
    rec_resp = client.post("/v1/recommendations/live", json=rec_payload)
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()

    assert "recommendations" in rec_data
    assert "decision" in rec_data
    assert rec_data["decision"]["verdict"] in ("STAY_PUT", "CONSIDER_MOVE", "RECOMMENDED_MOVE")

    # Step 7: Multi-Stop Shift Route Optimization
    shift_payload = {
        "latitude": lat,
        "longitude": lng,
        "vendor_category": "food",
        "shift_duration_hours": 8.0,
        "max_relocation_stops": 3,
        "min_dwell_time_minutes": 90,
        "setup_time_minutes": 15,
        "teardown_time_minutes": 15
    }
    shift_resp = client.post("/v1/routes/shift-plan", json=shift_payload)
    assert shift_resp.status_code == 200
    shift_data = shift_resp.json()
    assert shift_data["summary"]["total_net_profit_inr"] > 0
    assert len(shift_data["itinerary"]) > 0

    target_stop_cell = shift_data["itinerary"][0]["h3_cell"]

    # Step 8: Fleet Reservation on the Recommended Stop
    res_payload = {
        "vendor_id": "vendor-e2e-live-01",
        "h3_cell": target_stop_cell,
        "vendor_category": "food",
        "duration_minutes": 60,
        "is_soft": True,
        "deposit_amount_inr": 250.0
    }
    fleet_res = client.post("/v1/fleet/reserve-cell", json=res_payload)
    assert fleet_res.status_code in (200, 409)

    # Step 9: Ingest Real POS Transaction Telemetry
    tx_id = f"POS-E2E-{uuid.uuid4().hex[:8]}"
    pos_payload = {
        "transaction_id": tx_id,
        "vendor_id": "vendor-e2e-live-01",
        "amount_inr": 340.0,
        "payment_method": "upi",
        "category": "food",
        "items_count": 2,
        "location": {
            "latitude": lat,
            "longitude": lng
        },
        "event_timestamp": datetime.now(timezone.utc).isoformat()
    }
    pos_resp = client.post("/v1/telemetry/transactions", json=pos_payload, headers={"X-API-Key": "test-key"})
    assert pos_resp.status_code == 200
    assert pos_resp.json()["h3_cell_id"] is not None

    # Step 10: Model Health & Drift Verification
    drift_resp = client.get("/v1/system/model-health")
    assert drift_resp.status_code == 200
    drift_data = drift_resp.json()
    assert drift_data["status"] in ("HEALTHY", "ok")
