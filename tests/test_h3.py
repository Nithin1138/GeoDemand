"""
Comprehensive Test Suite for H3 Spatial Grid & Geospatial Guardrails (Member 2).
"""

import sys
from pathlib import Path
import pytest
import h3
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.simulation.h3_grid import (
    DEFAULT_H3_RESOLUTION,
    VIJAYAWADA_BBOX,
    generate_city_grid,
    is_within_city_bbox,
    is_valid_h3_cell,
    cell_to_centroid,
    get_k_ring_neighbors,
    cell_distance_km,
)
from src.api.spatial_filter import is_water_location, is_suitable_vendor_cell


def test_h3_resolution_strictly_8():
    """All H3 cells across the platform must be Resolution 8 (~0.74 km2)."""
    grid = generate_city_grid(center_lat=16.5062, center_lng=80.6480, radius_km=8.0)
    assert len(grid) > 500, f"Expected >500 H3 cells for Vijayawada 8km radius, got {len(grid)}"
    for cell in grid["h3_cell"]:
        assert is_valid_h3_cell(cell, resolution=8), f"Cell {cell} is not valid res 8"
        assert h3.get_resolution(cell) == DEFAULT_H3_RESOLUTION


def test_h3_latlng_deterministic_conversion():
    """Verify coordinate conversion is deterministic and invertible."""
    lat, lng = 16.5062, 80.6480
    cell = h3.latlng_to_cell(lat, lng, 8)
    assert is_valid_h3_cell(cell, resolution=8)

    c_lat, c_lng = cell_to_centroid(cell)
    # Centroid must be within 0.5km of original coordinate
    dist = cell_distance_km(cell, h3.latlng_to_cell(c_lat, c_lng, 8))
    assert dist == 0.0


def test_vijayawada_bounding_box():
    """All city grid cells must be strictly within the configured Vijayawada bounding box."""
    grid = generate_city_grid(center_lat=16.5062, center_lng=80.6480, radius_km=8.0)
    for _, row in grid.iterrows():
        assert is_within_city_bbox(row["lat"], row["lng"]), (
            f"Cell {row['h3_cell']} at ({row['lat']}, {row['lng']}) is outside Vijayawada bounding box"
        )


def test_water_exclusion_filter():
    """Verify water polygons correctly flag water cells and pass land cells."""
    # Krishna River point
    river_lat, river_lng = 16.506, 80.622
    assert is_water_location(river_lat, river_lng) is True, "Krishna River coordinate must be flagged as water"
    assert is_suitable_vendor_cell(river_lat, river_lng) is False, "Krishna River point must be unsuitable for vendors"

    # Benz Circle land point
    benz_lat, benz_lng = 16.5012, 80.6540
    assert is_water_location(benz_lat, benz_lng) is False, "Benz Circle commercial zone is NOT water"
    assert is_suitable_vendor_cell(benz_lat, benz_lng) is True, "Benz Circle must be suitable for vendors"


def test_k_ring_neighbors_expansion():
    """Hexagonal neighbor expansion formulas: k=0 -> 1, k=1 -> 7, k=2 -> 19 cells."""
    origin_cell = h3.latlng_to_cell(16.5062, 80.6480, 8)

    k0 = get_k_ring_neighbors(origin_cell, k=0)
    assert len(k0) == 1
    assert k0[0] == origin_cell

    k1 = get_k_ring_neighbors(origin_cell, k=1)
    assert len(k1) == 7
    assert origin_cell in k1

    k2 = get_k_ring_neighbors(origin_cell, k=2)
    assert len(k2) == 19
    assert set(k1).issubset(set(k2))


def test_distance_accuracy_and_monotonicity():
    """Distance between cells is symmetric, non-negative, and physically realistic."""
    cell_a = h3.latlng_to_cell(16.5062, 80.6480, 8)
    cell_b = h3.latlng_to_cell(16.5012, 80.6540, 8)

    # Self-distance is 0
    assert cell_distance_km(cell_a, cell_a) == 0.0

    # Symmetry
    d_ab = cell_distance_km(cell_a, cell_b)
    d_ba = cell_distance_km(cell_b, cell_a)
    assert abs(d_ab - d_ba) < 1e-6
    assert 0.1 < d_ab < 2.0  # Approx 0.8 km
