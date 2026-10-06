"""
Unit tests for the Candidate Location Pipeline in GeoDemand AI.

Validates the full step-by-step pipeline:
  Current Cell → Neighbor H3 Cells → Distance Filter → Valid Candidate Cells
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

import h3
try:
    import pytest
except ImportError:
    pytest = None

from api.location import (
    gps_to_h3,
    h3_centroid,
    haversine_km,
    generate_candidates,
    get_location_context,
)
from api.spatial_filter import is_water_location, is_water_cell, is_suitable_vendor_cell


def test_step1_current_cell_validation():
    """Step 1: Current cell must be a valid H3 resolution 8 cell."""
    lat, lng = 16.5062, 80.6480
    cell = gps_to_h3(lat, lng)
    assert h3.is_valid_cell(cell)
    assert h3.get_resolution(cell) == 8

    # Centroid check
    c_lat, c_lng = h3_centroid(cell)
    assert abs(c_lat - lat) < 0.05
    assert abs(c_lng - lng) < 0.05

    # Invalid coordinates check
    with pytest.raises(ValueError):
        gps_to_h3(95.0, 80.0)


def test_step2_neighbor_h3_expansion():
    """Step 2: Neighbor cells are generated using valid H3 grid expansion."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)
    
    candidates = generate_candidates(lat, lng, origin_cell, search_radius_km=1.5)
    assert len(candidates) > 1

    # Ensure all generated cells are valid H3 cells
    for cand in candidates:
        assert h3.is_valid_cell(cand.h3_cell)
        assert h3.get_resolution(cand.h3_cell) == 8


def test_step3_distance_filter():
    """Step 3: Distance filter strictly enforces the search_radius_km limit."""
    lat, lng = 16.5062, 80.6480
    origin_cell = gps_to_h3(lat, lng)
    search_radius = 2.0

    candidates = generate_candidates(lat, lng, origin_cell, search_radius_km=search_radius)
    for cand in candidates:
        if not cand.is_current_cell:
            assert cand.distance_km <= search_radius, f"Candidate {cand.h3_cell} exceeded radius: {cand.distance_km} km"

    # Verify invalid search radius raises ValueError
    with pytest.raises(ValueError):
        generate_candidates(lat, lng, origin_cell, search_radius_km=-1.0)


def test_step4_valid_candidate_cells_water_and_inaccessible_exclusion():
    """Step 4: Exclude water cells and inaccessible/unsuitable terrain."""
    # Prakasam Barrage water zone point
    water_lat, water_lng = 16.505, 80.620
    assert is_water_location(water_lat, water_lng) is True

    water_cell = gps_to_h3(water_lat, water_lng)
    assert is_water_cell(water_cell) is True
    assert is_suitable_vendor_cell(water_lat, water_lng, h3_cell=water_cell) is False

    # Candidates generated near water body exclude destination water cells
    candidates = generate_candidates(water_lat, water_lng, water_cell, search_radius_km=3.0, exclude_water=True)
    destination_candidates = [c for c in candidates if not c.is_current_cell]

    for cand in destination_candidates:
        assert not cand.is_water, f"Destination candidate {cand.h3_cell} is in water"
        assert not is_water_location(cand.centroid_lat, cand.centroid_lng)
        assert is_suitable_vendor_cell(cand.centroid_lat, cand.centroid_lng, h3_cell=cand.h3_cell)


def test_location_context():
    """Location context helper builds full validated provenance context."""
    ctx = get_location_context(16.5062, 80.6480)
    assert ctx.latitude == 16.5062
    assert ctx.longitude == 80.6480
    assert h3.is_valid_cell(ctx.h3_cell)
    assert ctx.h3_resolution == 8
    assert ctx.timezone_name == "Asia/Kolkata"
