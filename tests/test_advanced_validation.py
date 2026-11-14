"""
Test suite for Requirements 5, 6, and 7:
5. Check that water cells and unsuitable locations are excluded.
6. Verify provider failure handling and fallback behavior.
7. Test the API with multiple vendor categories and locations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))
sys.path.insert(0, str(ROOT / "src" / "features"))

import pytest
import h3

from api.spatial_filter import (
    is_water_location,
    is_water_cell,
    is_suitable_vendor_cell,
    KRISHNA_RIVER_POLYGON,
    PRAKASAM_BARRAGE_RESERVOIR_POLYGON,
    BHAVANI_ISLAND_WATER_POLYGON,
)
from api.location import generate_candidates, get_location_context
from api.providers.traffic import HERETrafficProvider, TimeOfDayTrafficProxy, TrafficProviderFactory
from api.providers.routing import OSRMRoutingProvider, RoutingProviderFactory
from api.providers.competition import GooglePlacesCompetitionProvider, OSMCompetitionProvider, CompetitionProviderFactory
from api.providers.events import EventbriteProvider, StochasticEventProvider, EventProviderFactory
from features.weather_client import OpenMeteoClient


# ===========================================================================
# Requirement 5: Water Cells & Unsuitable Locations Exclusion
# ===========================================================================

# Known water coordinates inside Vijayawada water bodies
KNOWN_WATER_COORDS = [
    (16.5080, 80.6150),  # Prakasam Barrage Deep Reservoir
    (16.5040, 80.6200),  # Mid Krishna River Channel
    (16.5200, 80.5800),  # Bhavani Island Backwater Lagoon
    (16.4950, 80.6350),  # Krishna River Krishnalanka riverbed
    (16.5200, 80.6450),  # Canal water basin
]

# Known dry land coordinates (commercial / residential Vijayawada)
KNOWN_LAND_COORDS = [
    (16.5062, 80.6480),  # Governorpet
    (16.4998, 80.6558),  # Benz Circle
    (16.5160, 80.6190),  # Vijayawada Railway Station
    (16.5180, 80.6650),  # Gunadala
]


@pytest.mark.parametrize("lat,lng", KNOWN_WATER_COORDS)
def test_water_location_detection(lat, lng):
    """Coordinates inside river polygons are strictly detected as water."""
    assert is_water_location(lat, lng) is True


@pytest.mark.parametrize("lat,lng", KNOWN_LAND_COORDS)
def test_land_location_not_water(lat, lng):
    """Dry land coordinates are correctly identified as non-water."""
    assert is_water_location(lat, lng) is False


@pytest.mark.parametrize("lat,lng", KNOWN_WATER_COORDS)
def test_water_cell_detection(lat, lng):
    """H3 cells covering water coordinates are detected as water cells."""
    cell = h3.latlng_to_cell(lat, lng, 8)
    assert is_water_cell(cell) is True


def test_candidate_generation_strictly_excludes_water():
    """generate_candidates with origin near Krishna river excludes all water cells from destinations."""
    origin_lat, origin_lng = 16.5062, 80.6180  # Near riverbank
    ctx = get_location_context(origin_lat, origin_lng)
    candidates = generate_candidates(
        origin_lat, origin_lng, ctx.h3_cell,
        search_radius_km=3.0, exclude_water=True,
    )

    # All non-current destination candidates must be non-water
    for c in candidates:
        if not c.is_current_cell:
            assert not is_water_location(c.centroid_lat, c.centroid_lng), f"Candidate {c.h3_cell} centroid is in water!"
            assert not is_water_cell(c.h3_cell), f"Candidate {c.h3_cell} is a water cell!"


def test_unsuitable_vendor_cell_filtering():
    """Unroaded, zero-POI, barren cells are flagged as unsuitable for vendors."""
    barren_features = {
        "road_density": 0.01,
        "population_density": 5.0,
        "building_density": 0.005,
        "total_pois": 0.0,
    }
    suitable_features = {
        "road_density": 1.5,
        "population_density": 2500.0,
        "building_density": 0.4,
        "total_pois": 12.0,
    }

    assert is_suitable_vendor_cell(16.5062, 80.6480, static_features=barren_features) is False
    assert is_suitable_vendor_cell(16.5062, 80.6480, static_features=suitable_features) is True


# ===========================================================================
# Requirement 6: Provider Failure Handling & Fallback Behavior
# ===========================================================================

def test_weather_client_network_failure_fallback():
    """When Open-Meteo raises RequestException, client returns synthetic fallback with proper source tag."""
    client = OpenMeteoClient(cache_dir="data/raw/weather_test")
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Simulated DNS failure")):
        # 1. Current weather fallback
        current = client.get_current_weather(16.5062, 80.6480)
        assert current["source_type"] == "synthetic_fallback"
        assert "synthetic" in current["source_name"]
        assert "temperature" in current
        assert "rainfall" in current

        # 2. Hourly forecast fallback
        forecast = client.get_hourly_forecast(16.5062, 80.6480, target_hour_offset=1)
        assert forecast["source_type"] == "synthetic_fallback"
        assert "synthetic" in forecast["source_name"]
        assert "temperature" in forecast


def test_traffic_provider_failure_fallback():
    """When HERE traffic API raises an error, it falls back to TimeOfDayTrafficProxy."""
    with patch.dict("os.environ", {"HERE_API_KEY": "dummy_key"}):
        with patch("requests.get", side_effect=requests.exceptions.Timeout("Traffic API timed out")):
            here_prov = HERETrafficProvider(api_key="dummy_key")
            result = here_prov.get_traffic(16.5062, 80.6480, 16.5030, 80.6274)
            # Must fall back gracefully without raising an uncaught exception
            assert result["source_type"] == "synthetic_fallback"
            assert result["source_name"] == "time_of_day_proxy"
            assert "traffic_speed_kmh" in result


def test_routing_provider_failure_fallback():
    """When all OSRM mirrors fail, OSRMRoutingProvider falls back to straight-line & grid interpolation."""
    osrm = OSRMRoutingProvider()
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("OSRM servers down")):
        # Travel time fallback
        tt = osrm.get_travel_time(16.5062, 80.6480, 16.5030, 80.6274)
        assert tt["source_type"] == "synthetic_fallback"
        assert tt["source_name"] == "haversine_estimate"
        assert tt["distance_km"] > 0
        assert tt["duration_min"] > 0

        # Geometry fallback
        geom = osrm.get_route_geometry(16.5062, 80.6480, 16.5030, 80.6274)
        assert geom["code"] == "Fallback"
        assert geom["source"] == "grid_road_interpolation"
        assert len(geom["coordinates"]) >= 2


def test_competition_provider_fallback():
    """When Google Places fails or has no key, competition falls back to OSM static POI density."""
    with patch.dict("os.environ", {"GOOGLE_PLACES_API_KEY": "dummy_key"}):
        with (
            patch("requests.post", side_effect=requests.exceptions.HTTPError("Places quota exceeded")),
            patch("requests.get", side_effect=requests.exceptions.HTTPError("Legacy places failed")),
        ):
            places_prov = GooglePlacesCompetitionProvider(api_key="dummy_key")
            comp = places_prov.get_competition_score("88619aa6c7fffff", "food")
            assert comp["source_type"] == "real_periodic"
            assert comp["source_name"] == "openstreetmap"
            assert 0.0 <= comp["competition_score"] <= 1.0


def test_event_provider_fallback():
    """When external event provider has no key or fails, it falls back to StochasticEventProvider."""
    prov = EventProviderFactory.get_provider(preferred="auto")
    assert isinstance(prov, StochasticEventProvider)
    event_data = prov.get_event_importance("88619aa6c7fffff", timestamp=pytest.importorskip("datetime").datetime.now())
    assert event_data["source_type"] == "simulated"
    assert "event_importance" in event_data


# ===========================================================================
# Requirement 7: Multi-Category & Multi-Location Testing
# ===========================================================================

VENDOR_CATEGORIES = [
    "food",
    "fruits",
    "repair",
    "salon",
    "chai",        # non-standard / custom category
    "retail",      # general merchant
]

TEST_LOCATIONS = [
    ("Vijayawada Junction", 16.5160, 80.6190),
    ("Benz Circle Hub", 16.4998, 80.6558),
    ("Governorpet Commercial", 16.5062, 80.6480),
    ("Auto Nagar Cluster", 16.4880, 80.6750),
    ("Gunadala Suburb", 16.5180, 80.6650),
    ("Tadepalle South", 16.4800, 80.6300),
]


@pytest.fixture(scope="module")
def client():
    mock_weather = {
        "temperature": 29.0, "humidity": 65.0, "rainfall": 0.0, "wind_speed": 10.0,
        "pressure": 1010.0, "cloud_cover": 20.0, "visibility": 9.0, "weather_condition": "clear",
        "uv_index": 5.0, "source": "real:open-meteo-current", "source_type": "real_live",
        "source_name": "open-meteo", "observation_timestamp": "2026-08-18T10:00:00+00:00",
        "retrieved_at": "2026-08-18T10:00:00+00:00", "from_cache": False, "cache_age_seconds": 0,
    }
    with (
        patch("features.weather_client.OpenMeteoClient.get_current_weather", return_value=mock_weather),
        patch("features.weather_client.OpenMeteoClient.get_hourly_forecast", return_value={**mock_weather, "source_type": "real_live"}),
    ):
        from fastapi.testclient import TestClient
        from api.app import app
        with TestClient(app) as c:
            yield c


@pytest.mark.parametrize("category", VENDOR_CATEGORIES)
def test_multiple_vendor_categories(client, category):
    """Test live recommendations across multiple standard and custom vendor categories."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": 16.5062,
        "longitude": 80.6480,
        "vendor_category": category,
        "top_n": 3,
        "search_radius_km": 2.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["verdict"] in ("RECOMMENDED_MOVE", "CONSIDER_MOVE", "STAY_PUT")
    assert len(body["recommendations"]) <= 3
    for rec in body["recommendations"]:
        assert rec["expected_customers"] >= 0
        assert rec["expected_profit_inr"] > 0
        assert rec["distance_km"] <= 2.0
        assert not is_water_location(rec["latitude"], rec["longitude"])


@pytest.mark.parametrize("loc_name,lat,lng", TEST_LOCATIONS)
def test_multiple_geographic_locations(client, loc_name, lat, lng):
    """Test live recommendations across diverse commercial, transit, and residential hubs."""
    r = client.post("/v1/recommendations/live", json={
        "latitude": lat,
        "longitude": lng,
        "vendor_category": "food",
        "top_n": 3,
        "search_radius_km": 2.5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["current_location"]["latitude"] == lat
    assert body["current_location"]["longitude"] == lng
    assert body["decision"]["verdict"] in ("RECOMMENDED_MOVE", "CONSIDER_MOVE", "STAY_PUT")
    assert "demand_heatmap" in body
    assert len(body["demand_heatmap"]) > 0

    # Every recommended target must be strictly non-water
    for rec in body["recommendations"]:
        assert not is_water_location(rec["latitude"], rec["longitude"]), f"Water cell recommended near {loc_name}!"
        assert not is_water_cell(rec["h3_cell"]), f"Water cell {rec['h3_cell']} recommended near {loc_name}!"


def test_predict_demand_across_all_categories_and_locations(client):
    """Single location predict endpoint operates consistently across all vendor types."""
    for category in VENDOR_CATEGORIES:
        r = client.post("/v1/predict/demand", json={
            "latitude": 16.4998,
            "longitude": 80.6558,
            "vendor_category": category,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["vendor_category"] == category
        assert body["prediction"]["expected_customers"] >= 0
        assert body["economics"]["expected_profit_inr"] > 0
