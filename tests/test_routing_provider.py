"""
Unit test suite for OSRM Routing Provider, Cache, Circuit Breaker & Decision Safety.

Validates:
  1. OSRM routing and road distance/travel time calculations
  2. Route response validation (well-formed vs malformed/NoRoute)
  3. Route timeouts and mirror failure fallback
  4. In-memory routing cache with TTL and stats
  5. Circuit breaker behavior (failure threshold and cooldown)
  6. Coordinate validation (bounds, NaNs, Infs)
  7. Traffic proxy integration
  8. Fuel price proxy integration
  9. Safety invariant: Missing routing data cannot generate false relocation decisions
  10. Factory provider selection
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

import pytest
import requests

from api.decision_engine import evaluate_decision
from api.providers.fuel import FuelPriceProvider
from api.providers.routing import (
    OSRMRoutingProvider,
    RoutingProviderFactory,
    validate_coordinates,
    validate_route_response,
)
from api.providers.traffic import TimeOfDayTrafficProxy


@pytest.fixture
def osrm_provider():
    prov = OSRMRoutingProvider()
    prov.clear_cache()
    prov.reset_circuit_breaker()
    return prov


# ---------------------------------------------------------------------------
# 1 & 2. OSRM Road Distance & Travel Time Calculation & Response Validation
# ---------------------------------------------------------------------------

def test_osrm_travel_time_calculation_mock(osrm_provider):
    """Verify road distance and travel duration extracted from valid OSRM response."""
    mock_osrm_json = {
        "code": "Ok",
        "routes": [
            {
                "duration": 480.0,  # 8 minutes
                "distance": 2400.0,  # 2.4 km
                "weight_name": "routability",
                "weight": 480.0,
            }
        ],
        "waypoints": [{"location": [80.6480, 16.5062]}, {"location": [80.6274, 16.5030]}],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_osrm_json

    with patch("requests.get", return_value=mock_resp):
        res = osrm_provider.get_travel_time(16.5062, 80.6480, 16.5030, 80.6274)

        assert res["duration_sec"] == 480
        assert res["duration_min"] == 8.0
        assert res["distance_m"] == 2400
        assert res["distance_km"] == 2.4
        assert res["is_routable"] is True
        assert res["source_type"] == "real_live"
        assert res["source_name"] == "osrm"


def test_identical_origin_dest_zero_cost(osrm_provider):
    """Verify identical origin and destination returns 0 distance and 0 time without network calls."""
    with patch("requests.get") as mock_get:
        res = osrm_provider.get_travel_time(16.5062, 80.6480, 16.5062, 80.6480)
        assert mock_get.call_count == 0
        assert res["distance_km"] == 0.0
        assert res["duration_min"] == 0.0
        assert res["is_routable"] is True


def test_validate_route_response_helper():
    """Verify route response validation rejects malformed, incomplete, or unroutable responses."""
    # Valid response
    valid_data = {"code": "Ok", "routes": [{"duration": 300, "distance": 1500}]}
    assert validate_route_response(valid_data) is True

    # Code not Ok
    assert validate_route_response({"code": "NoRoute", "routes": []}) is False
    assert validate_route_response({"code": "InvalidQuery"}) is False

    # Missing or empty routes
    assert validate_route_response({"code": "Ok", "routes": []}) is False
    assert validate_route_response({"code": "Ok"}) is False

    # Negative duration/distance or NaNs
    assert validate_route_response({"code": "Ok", "routes": [{"duration": -10, "distance": 100}]}) is False
    assert validate_route_response({"code": "Ok", "routes": [{"duration": float("nan"), "distance": 100}]}) is False
    assert validate_route_response(None) is False
    assert validate_route_response("not a dict") is False


# ---------------------------------------------------------------------------
# 3 & 4. Route Timeouts, Provider Failure Fallback & Circuit Breaker
# ---------------------------------------------------------------------------

def test_route_timeout_and_fallback(osrm_provider):
    """Verify route timeout on all mirrors triggers graceful synthetic fallback."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout("OSRM request timed out")):
        res = osrm_provider.get_travel_time(16.5062, 80.6480, 16.5200, 80.6300)

        assert res["source_type"] == "synthetic_fallback"
        assert res["source_name"] == "haversine_estimate"
        assert res["distance_km"] > 0.0
        assert res["duration_min"] > 0.0
        assert res["is_routable"] is True
        # Verify urban road winding factor (1.25x) was applied
        assert "road_winding_factor" in res
        assert res["road_winding_factor"] == 1.25


def test_circuit_breaker_mechanism(osrm_provider):
    """Verify consecutive mirror failures open circuit breaker to protect latency."""
    assert osrm_provider.is_circuit_open() is False

    # Trip the circuit breaker by failing twice
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        osrm_provider.get_travel_time(16.5062, 80.6480, 16.5100, 80.6400)
        osrm_provider.get_travel_time(16.5062, 80.6480, 16.5200, 80.6300)

    assert osrm_provider.is_circuit_open() is True

    # Next call must return immediately from fallback WITHOUT making any network call
    with patch("requests.get") as mock_get:
        res = osrm_provider.get_travel_time(16.5062, 80.6480, 16.5300, 80.6200)
        assert mock_get.call_count == 0  # Zero network attempts!
        assert res["source_type"] == "synthetic_fallback"

    # Reset circuit breaker
    osrm_provider.reset_circuit_breaker()
    assert osrm_provider.is_circuit_open() is False


# ---------------------------------------------------------------------------
# 5 & 6. In-Memory Cache with TTL & Stats & Coordinate Validation
# ---------------------------------------------------------------------------

def test_routing_cache_hits_and_stats(osrm_provider):
    """Verify routing queries are cached in memory and tracked in cache statistics."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "code": "Ok",
        "routes": [{"duration": 360.0, "distance": 1800.0}],
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        # First call: cache miss
        res1 = osrm_provider.get_travel_time(16.5062, 80.6480, 16.5186, 80.6200)
        assert mock_get.call_count == 1

        stats = osrm_provider.get_cache_stats()
        assert stats["size"] == 1
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Second call with same coordinates: cache hit
        res2 = osrm_provider.get_travel_time(16.5062, 80.6480, 16.5186, 80.6200)
        assert mock_get.call_count == 1  # No additional network call!
        assert res1["duration_sec"] == res2["duration_sec"]

        stats2 = osrm_provider.get_cache_stats()
        assert stats2["hits"] == 1

        # Clear cache
        osrm_provider.clear_cache()
        assert osrm_provider.get_cache_stats()["size"] == 0


def test_invalid_coordinates_rejection(osrm_provider):
    """Verify invalid coordinate bounds, NaNs, and Infs raise ValueError."""
    assert validate_coordinates(16.5, 80.6, 16.6, 80.7) is True
    assert validate_coordinates(95.0, 80.6, 16.6, 80.7) is False
    assert validate_coordinates(16.5, 185.0, 16.6, 80.7) is False
    assert validate_coordinates(float("nan"), 80.6, 16.6, 80.7) is False
    assert validate_coordinates(16.5, float("inf"), 16.6, 80.7) is False

    with pytest.raises(ValueError, match="Invalid coordinates"):
        osrm_provider.get_travel_time(95.0, 80.6, 16.5, 80.6)

    with pytest.raises(ValueError, match="Invalid coordinates"):
        osrm_provider.get_travel_time(float("nan"), 80.6, 16.5, 80.6)


# ---------------------------------------------------------------------------
# 7 & 8. Traffic & Fuel Proxy Integration
# ---------------------------------------------------------------------------

def test_traffic_proxy_integration(osrm_provider):
    """Verify route travel duration adjustment via TrafficProvider."""
    traffic_prov = TimeOfDayTrafficProxy()

    with patch.object(osrm_provider, "get_travel_time", return_value={"distance_km": 3.0, "duration_min": 6.0, "is_routable": True}):
        adjusted = osrm_provider.get_traffic_adjusted_travel_time(
            origin_lat=16.5062, origin_lng=80.6480, dest_lat=16.5300, dest_lng=80.6200, traffic_provider=traffic_prov
        )

        assert "duration_min" in adjusted
        assert "base_duration_min" in adjusted
        assert "traffic_delay_min" in adjusted
        assert "congestion_ratio" in adjusted
        assert adjusted["duration_min"] >= adjusted["base_duration_min"]


def test_fuel_proxy_integration(osrm_provider):
    """Verify route fuel cost calculation via FuelPriceProvider with congestion penalty."""
    fuel_prov = FuelPriceProvider()

    # Normal traffic (congestion = 0)
    fuel_normal = osrm_provider.get_route_fuel_cost(distance_km=3.6, fuel_provider=fuel_prov, congestion_ratio=0.0)
    # (3.6 / 18.0) * 95 = 19.00 INR
    assert fuel_normal["distance_km"] == 3.6
    assert fuel_normal["fuel_cost_inr"] == 19.00
    assert fuel_normal["congestion_penalty_multiplier"] == 1.0

    # Heavy traffic (congestion = 1.0 => 1.35x penalty)
    fuel_heavy = osrm_provider.get_route_fuel_cost(distance_km=3.6, fuel_provider=fuel_prov, congestion_ratio=1.0)
    assert fuel_heavy["congestion_penalty_multiplier"] == 1.35
    assert fuel_heavy["fuel_cost_inr"] == round(19.00 * 1.35, 2)


# ---------------------------------------------------------------------------
# 9. Safety Invariant: Missing Routing Data Cannot Generate False Move Decisions
# ---------------------------------------------------------------------------

def test_missing_routing_data_safety_invariant():
    """
    SAFETY INVARIANT TEST:
    If routing fails or candidate is flagged unroutable (is_routable = False),
    the Decision Engine MUST NOT generate a RECOMMENDED_MOVE!
    """
    baseline_curr = {
        "h3_cell": "cell_origin",
        "expected_profit_inr": 1500.0,
        "relocation_adjusted_realized_profit_inr": 1500.0,
        "is_current_cell": True,
    }

    # Case A: Unroutable destination candidate (e.g. into water or unnavigable terrain)
    unroutable_cand = {
        "h3_cell": "cell_unroutable",
        "landmark_name": "Inaccessible Island",
        "expected_profit_inr": 3000.0,
        "candidate_predicted_profit_inr": 3000.0,
        "relocation_adjusted_realized_profit_inr": 2800.0,  # High theoretical profit!
        "distance_km": 1.5,
        "estimated_travel_time_min": 6.0,
        "is_routable": False,  # Flagged unroutable
        "is_current_cell": False,
    }

    dec = evaluate_decision(unroutable_cand, baseline_curr)
    assert dec.decision_verdict == "STAY_PUT"
    assert dec.is_move_recommended is False
    assert "cannot be verified" in dec.decision_reason

    # Case B: Candidate distinct from origin with missing routing data (distance = 0.0)
    zero_dist_cand = {
        "h3_cell": "cell_different",
        "landmark_name": "Ghost Teleport Spot",
        "expected_profit_inr": 2500.0,
        "candidate_predicted_profit_inr": 2500.0,
        "relocation_adjusted_realized_profit_inr": 2500.0,
        "distance_km": 0.0,  # Missing routing data!
        "estimated_travel_time_min": 0.0,
        "is_current_cell": False,
    }

    dec2 = evaluate_decision(zero_dist_cand, baseline_curr)
    # Must enforce fallback distance (1.0 km, 4.0 min) rather than permitting 0.0km teleportation
    assert dec2.distance_km >= 1.0
    assert dec2.travel_time_min >= 4.0


# ---------------------------------------------------------------------------
# 10. Factory Provider Selection
# ---------------------------------------------------------------------------

def test_routing_provider_factory():
    """Verify RoutingProviderFactory instantiates OSRM by default or according to config."""
    prov = RoutingProviderFactory.get_provider("osrm")
    assert isinstance(prov, OSRMRoutingProvider)
    assert prov.source_name == "osrm"


def test_prune_artificial_uturns_initial_loop():
    """Verify OSRMRoutingProvider prunes artificial initial U-turn loop from dual-carriageway medians."""
    mock_osrm_data = {
        "routes": [
            {
                "distance": 1889.0,
                "duration": 193.1,
                "geometry": {
                    "coordinates": [
                        [80.648033, 16.506214],
                        [80.648085, 16.506102],
                        [80.650175, 16.507605],
                        [80.64809, 16.505977],
                        [80.64794, 16.505968],
                    ]
                },
                "legs": [
                    {
                        "steps": [
                            {"maneuver": {"type": "depart", "location": [80.648033, 16.506214]}, "distance": 15.0, "duration": 3.0},
                            {"maneuver": {"type": "turn", "modifier": "left", "location": [80.648085, 16.506102]}, "distance": 300.0, "duration": 30.0},
                            {"maneuver": {"type": "continue", "modifier": "uturn", "location": [80.650175, 16.507605]}, "distance": 300.0, "duration": 30.0},
                            {"maneuver": {"type": "roundabout", "modifier": "slight left", "location": [80.64809, 16.505977]}, "distance": 50.0, "duration": 5.0},
                            {"maneuver": {"type": "exit roundabout", "modifier": "left", "location": [80.64794, 16.505968]}, "distance": 1200.0, "duration": 120.0},
                        ]
                    }
                ],
            }
        ]
    }
    raw_coords = [[pt[1], pt[0]] for pt in mock_osrm_data["routes"][0]["geometry"]["coordinates"]]
    pruned, dist_saved, dur_saved = OSRMRoutingProvider._prune_artificial_uturns(
        mock_osrm_data, 16.5062, 80.6480, 16.505968, 80.64794, raw_coords
    )

    # Must prune the 600m redundant loop
    assert dist_saved >= 600.0
    assert dur_saved >= 60.0
    assert len(pruned) < len(raw_coords)
    # The first coordinate after origin must now jump directly to the roundabout
    assert pruned[0] == [16.5062, 80.6480]
    assert abs(pruned[1][0] - 16.505977) < 1e-4

