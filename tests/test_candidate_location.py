"""
Comprehensive candidate location generation and spatial filter validation test suite.

Validates the full 11-step pipeline for Phase 3.3:
  1. Generate nearby H3 cells via k-ring expansion
  2. Apply configurable search radius
  3. Exclude current location where relocation is being evaluated
  4. Exclude water and unsuitable cells
  5. Remove invalid coordinates
  6. Remove duplicate H3 cells
  7. Resolve candidate POIs
  8. Validate candidate accessibility
  9. Support different starting locations
  10. Return eligible candidate details
  11. Validate candidate generation performance (< 100ms)
"""

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

import h3
import pytest

from api.location import (
    CandidateCell,
    LocationContext,
    estimate_travel_min,
    generate_candidates,
    get_location_context,
    gps_to_h3,
    h3_centroid,
    haversine_km,
)
from api.poi_resolver import POIResolver, resolve_candidate_landmark
from api.spatial_filter import (
    is_accessible_location,
    is_suitable_vendor_cell,
    is_valid_coordinate,
    is_water_cell,
    is_water_location,
    reset_spatial_filter_cache,
)


@pytest.fixture(autouse=True)
def setup_filter_cache():
    """Ensure clean state before each test run."""
    reset_spatial_filter_cache()


# ---------------------------------------------------------------------------
# 1. Generate nearby H3 cells
# ---------------------------------------------------------------------------

def test_nearby_h3_generation_k_ring():
    """Verify H3 cells are generated using valid k-ring expansion at resolution 8."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)

    candidates = generate_candidates(lat, lng, origin_cell, search_radius_km=1.5)
    assert len(candidates) > 1

    for cand in candidates:
        assert isinstance(cand.h3_cell, str)
        assert h3.is_valid_cell(cand.h3_cell)
        assert h3.get_resolution(cand.h3_cell) == 8
        assert cand.ring_distance >= 0
        if cand.is_current_cell:
            assert cand.ring_distance == 0


# ---------------------------------------------------------------------------
# 2. Apply configurable search radius
# ---------------------------------------------------------------------------

def test_configurable_search_radius():
    """Verify configurable search radius controls candidate search bounds."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)

    cands_1km = generate_candidates(lat, lng, origin_cell, search_radius_km=1.0)
    cands_2km = generate_candidates(lat, lng, origin_cell, search_radius_km=2.0)
    cands_3km = generate_candidates(lat, lng, origin_cell, search_radius_km=3.0)

    # Larger search radii encompass equal or greater number of candidates
    assert len(cands_1km) <= len(cands_2km) <= len(cands_3km)

    # All returned candidates must be within the specified radius
    for cand in cands_2km:
        assert cand.distance_km <= 2.0, f"Candidate {cand.h3_cell} exceeded 2.0 km: {cand.distance_km}"

    for cand in cands_3km:
        assert cand.distance_km <= 3.0, f"Candidate {cand.h3_cell} exceeded 3.0 km: {cand.distance_km}"

    # Invalid radius checks
    with pytest.raises(ValueError):
        generate_candidates(lat, lng, origin_cell, search_radius_km=-1.0)

    with pytest.raises(ValueError):
        generate_candidates(lat, lng, origin_cell, search_radius_km=0.0)

    with pytest.raises(ValueError):
        generate_candidates(lat, lng, origin_cell, search_radius_km=float("nan"))


# ---------------------------------------------------------------------------
# 3. Exclude current location where relocation is being evaluated
# ---------------------------------------------------------------------------

def test_exclude_current_location():
    """Verify exclude_current=True removes the origin cell from candidate results."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)

    # Default (exclude_current=False) includes current cell as first element
    cands_with_current = generate_candidates(lat, lng, origin_cell, search_radius_km=2.0, exclude_current=False)
    assert any(c.is_current_cell for c in cands_with_current)
    assert cands_with_current[0].is_current_cell is True
    assert cands_with_current[0].h3_cell == origin_cell
    assert cands_with_current[0].distance_km == 0.0

    # With exclude_current=True, origin cell is excluded
    cands_without_current = generate_candidates(lat, lng, origin_cell, search_radius_km=2.0, exclude_current=True)
    assert not any(c.is_current_cell for c in cands_without_current)
    assert origin_cell not in [c.h3_cell for c in cands_without_current]
    for c in cands_without_current:
        assert c.distance_km > 0.0
        assert c.is_current_cell is False


# ---------------------------------------------------------------------------
# 4. Exclude water and unsuitable cells
# ---------------------------------------------------------------------------

def test_exclude_water_and_unsuitable_cells():
    """Verify water bodies (Krishna River, Prakasam Barrage) and unroutable cells are excluded."""
    # Known Prakasam Barrage water point
    water_lat, water_lng = 16.505, 80.620
    assert is_water_location(water_lat, water_lng) is True

    water_cell = gps_to_h3(water_lat, water_lng)
    assert is_water_cell(water_cell) is True
    assert is_suitable_vendor_cell(water_lat, water_lng, h3_cell=water_cell) is False

    # Candidates generated near river exclude destination water cells
    candidates = generate_candidates(
        water_lat, water_lng, water_cell,
        search_radius_km=3.0,
        exclude_water=True,
        exclude_unsuitable=True,
    )
    destination_candidates = [c for c in candidates if not c.is_current_cell]
    assert len(destination_candidates) > 0

    for cand in destination_candidates:
        assert not cand.is_water, f"Candidate {cand.h3_cell} should not be marked water"
        assert not is_water_location(cand.centroid_lat, cand.centroid_lng)
        assert is_suitable_vendor_cell(cand.centroid_lat, cand.centroid_lng, h3_cell=cand.h3_cell)


# ---------------------------------------------------------------------------
# 5. Remove invalid coordinates
# ---------------------------------------------------------------------------

def test_remove_invalid_coordinates():
    """Verify invalid or non-numeric coordinates are strictly rejected with ValueError."""
    # Out of range coordinates
    with pytest.raises(ValueError, match="Invalid"):
        gps_to_h3(95.0, 80.0)

    with pytest.raises(ValueError, match="Invalid"):
        gps_to_h3(-95.0, 80.0)

    with pytest.raises(ValueError, match="Invalid"):
        gps_to_h3(16.5, 185.0)

    with pytest.raises(ValueError, match="Invalid"):
        gps_to_h3(16.5, -185.0)

    # NaNs and Infs
    with pytest.raises(ValueError, match="Invalid"):
        gps_to_h3(float("nan"), 80.0)

    with pytest.raises(ValueError, match="Invalid"):
        gps_to_h3(16.5, float("inf"))

    with pytest.raises(ValueError, match="Invalid"):
        generate_candidates(float("nan"), 80.0, "88619aa6c7fffff")

    with pytest.raises(ValueError, match="Invalid"):
        get_location_context(91.0, 80.0)

    # Helper coordinate validator
    assert is_valid_coordinate(16.5062, 80.6480) is True
    assert is_valid_coordinate(float("nan"), 80.6480) is False
    assert is_valid_coordinate(16.5062, float("inf")) is False
    assert is_valid_coordinate(None, 80.6480) is False


# ---------------------------------------------------------------------------
# 6. Remove duplicate H3 cells
# ---------------------------------------------------------------------------

def test_zero_duplicate_h3_cells():
    """Verify candidate generation never returns duplicate H3 cells."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)

    for radius in [1.5, 3.0, 5.0]:
        candidates = generate_candidates(lat, lng, origin_cell, search_radius_km=radius)
        cell_ids = [c.h3_cell for c in candidates]
        assert len(cell_ids) == len(set(cell_ids)), f"Duplicate H3 cells found at radius {radius}"


# ---------------------------------------------------------------------------
# 7. Resolve candidate POIs
# ---------------------------------------------------------------------------

def test_resolve_candidate_pois():
    """Verify candidate cells are enriched with resolved OSM landmark and POI details."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)

    candidates = generate_candidates(lat, lng, origin_cell, search_radius_km=2.0, resolve_pois=True)
    assert len(candidates) > 0

    for cand in candidates:
        assert cand.landmark_name is not None
        assert isinstance(cand.landmark_name, str)
        assert len(cand.landmark_name) > 0
        assert cand.landmark_type is not None
        assert isinstance(cand.is_college, bool)
        assert cand.landmark_emoji is not None

    # Directly verify convenience POI resolver function
    poi_info = resolve_candidate_landmark(lat, lng, origin_cell)
    assert "landmark_name" in poi_info
    assert "landmark_type" in poi_info
    assert "emoji" in poi_info


# ---------------------------------------------------------------------------
# 8. Validate candidate accessibility
# ---------------------------------------------------------------------------

def test_validate_candidate_accessibility():
    """Verify candidate accessibility check ensures road navigability."""
    # Accessible urban location in Vijayawada
    urban_lat, urban_lng = 16.5062, 80.6480
    urban_cell = gps_to_h3(urban_lat, urban_lng)
    assert is_accessible_location(urban_lat, urban_lng, h3_cell=urban_cell) is True

    # Inaccessible water location
    water_lat, water_lng = 16.505, 80.620
    water_cell = gps_to_h3(water_lat, water_lng)
    assert is_accessible_location(water_lat, water_lng, h3_cell=water_cell) is False

    # Low road density feature check
    sparse_features = {"road_density": 0.05}
    assert is_accessible_location(urban_lat, urban_lng, static_features=sparse_features) is False

    sufficient_features = {"road_density": 0.35}
    assert is_accessible_location(urban_lat, urban_lng, static_features=sufficient_features) is True


# ---------------------------------------------------------------------------
# 9. Support different starting locations
# ---------------------------------------------------------------------------

def test_support_different_starting_locations():
    """Verify candidate generation operates robustly across multiple distinct starting locations."""
    starting_locations = [
        ("Vijayawada Benz Circle", 16.5062, 80.6480),
        ("Vijayawada Railway Station", 16.5186, 80.6200),
        ("Vijayawada Bhavanipuram", 16.5300, 80.5900),
        ("Undavalli / Amaravati Bank", 16.4950, 80.5850),
        ("Guntur Market", 16.3067, 80.4365),
        ("Hyderabad Hitec City", 17.4435, 78.3772),
    ]

    for name, lat, lng in starting_locations:
        cell = gps_to_h3(lat, lng)
        candidates = generate_candidates(lat, lng, cell, search_radius_km=1.5, resolve_pois=True)
        assert len(candidates) >= 1, f"Failed to generate candidates for {name}"
        assert candidates[0].h3_cell == cell
        assert candidates[0].is_current_cell is True


# ---------------------------------------------------------------------------
# 10. Return eligible candidate details
# ---------------------------------------------------------------------------

def test_return_eligible_candidate_details():
    """Verify CandidateCell data structure contains complete and accurately typed fields."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)

    candidates = generate_candidates(lat, lng, origin_cell, search_radius_km=1.5, resolve_pois=True)
    first_cand = candidates[0]

    # Verify dataclass fields
    assert isinstance(first_cand, CandidateCell)
    assert isinstance(first_cand.h3_cell, str)
    assert isinstance(first_cand.centroid_lat, float)
    assert isinstance(first_cand.centroid_lng, float)
    assert isinstance(first_cand.distance_km, float)
    assert isinstance(first_cand.estimated_travel_min, float)
    assert isinstance(first_cand.is_current_cell, bool)
    assert isinstance(first_cand.is_water, bool)
    assert isinstance(first_cand.ring_distance, int)

    # Verify dictionary export
    cand_dict = first_cand.to_dict()
    assert isinstance(cand_dict, dict)
    expected_keys = {
        "h3_cell", "centroid_lat", "centroid_lng", "distance_km",
        "estimated_travel_min", "is_current_cell", "is_water", "ring_distance",
        "landmark_name", "landmark_type", "is_college", "landmark_emoji", "landmark_display",
    }
    assert expected_keys.issubset(cand_dict.keys())


# ---------------------------------------------------------------------------
# 11. Validate candidate generation performance (< 100ms)
# ---------------------------------------------------------------------------

def test_candidate_generation_performance():
    """Verify candidate generation latency is well under 100ms for standard urban radii."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)

    # Warm up caches
    generate_candidates(lat, lng, origin_cell, search_radius_km=2.0, resolve_pois=True)

    latencies = []
    for _ in range(5):
        start = time.perf_counter()
        cands = generate_candidates(lat, lng, origin_cell, search_radius_km=3.0, resolve_pois=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        assert len(cands) > 5

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 100.0, f"Average candidate generation latency {avg_latency:.2f}ms exceeds 100ms threshold"
