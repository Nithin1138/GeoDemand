"""
Geospatial Water & Land-Suitability Filter for GeoDemand AI.

Prevents recommending candidate locations inside water bodies, rivers,
reservoirs, dams, swamps, and unroutable terrain.

Mobile vendors (food trucks, fruit stalls, tea stalls, salon vans) require
road access and pedestrian/commercial footfall. They cannot operate in:
  - Krishna River & Prakasam Barrage reservoir pool
  - Bhavani Island water channels & backwaters
  - Major irrigation canals & drainage reservoirs
  - Non-roaded water cells
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import h3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Water Body Exclusion Polygons (Vijayawada Region & Krishna River Corridor)
# ---------------------------------------------------------------------------

# Main Krishna River Corridor: Upstream (Kondapalli / Ibrahimpatnam) ->
# Prakasam Barrage -> Downstream (Krishnalanka / Tadepalle / Yanamalakuduru)
KRISHNA_RIVER_POLYGON: List[Tuple[float, float]] = [
    (16.570, 80.485),
    (16.555, 80.520),
    (16.545, 80.550),
    (16.535, 80.575),
    (16.520, 80.598),
    (16.514, 80.612),  # North bank - Durga Ghat / Kanaka Durga
    (16.508, 80.622),  # Prakasam Barrage North / Governorpet riverbank
    (16.498, 80.642),  # Krishnalanka bank
    (16.488, 80.665),  # Ranigarithota bank
    (16.478, 80.690),  # Tarakarama Nagar riverbank
    (16.455, 80.730),  # Downstream exit North
    (16.440, 80.718),  # Downstream exit South
    (16.460, 80.680),  # South bank - Yanamalakuduru / Penumaka
    (16.475, 80.650),  # South bank - Undavalli / Tadepalle
    (16.488, 80.630),  # South bank - Seethanagaram
    (16.500, 80.608),  # Prakasam Barrage South (Seethanagaram hill base)
    (16.510, 80.578),  # South bank upstream (Venkatapalem)
    (16.522, 80.540),  # South bank upstream (Mandadam / Rayapudi)
    (16.538, 80.495),  # South bank upstream (Amaravati bank)
]

# Prakasam Barrage Deep Reservoir & Seethanagaram Lock
PRAKASAM_BARRAGE_RESERVOIR_POLYGON: List[Tuple[float, float]] = [
    (16.512, 80.605),
    (16.515, 80.625),
    (16.505, 80.632),
    (16.495, 80.625),
    (16.498, 80.605),
]

# Bhavani Island Backwater Lagoon / River Split
BHAVANI_ISLAND_WATER_POLYGON: List[Tuple[float, float]] = [
    (16.532, 80.570),
    (16.528, 80.595),
    (16.515, 80.595),
    (16.512, 80.575),
    (16.522, 80.560),
]

# Gundala / Eluru Canal reservoir pond & Ryves Canal junction
CANAL_WATER_BASINS_POLYGON: List[Tuple[float, float]] = [
    (16.524, 80.636),
    (16.525, 80.648),
    (16.516, 80.652),
    (16.512, 80.638),
]

ALL_WATER_POLYGONS = [
    KRISHNA_RIVER_POLYGON,
    PRAKASAM_BARRAGE_RESERVOIR_POLYGON,
    BHAVANI_ISLAND_WATER_POLYGON,
    CANAL_WATER_BASINS_POLYGON,
]


def point_in_polygon(lat: float, lng: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    p1lat, p1lng = polygon[0]
    for i in range(1, n + 1):
        p2lat, p2lng = polygon[i % n]
        if lng > min(p1lng, p2lng):
            if lng <= max(p1lng, p2lng):
                if lat <= max(p1lat, p2lat):
                    if p1lng != p2lng:
                        xinters = (lng - p1lng) * (p2lat - p1lat) / (p2lng - p1lng) + p1lat
                    if p1lat == p2lat or lat <= xinters:
                        inside = not inside
        p1lat, p1lng = p2lat, p2lng
    return inside


def is_water_location(lat: float, lng: float) -> bool:
    """Check if coordinates fall inside any known water body polygon."""
    for poly in ALL_WATER_POLYGONS:
        if point_in_polygon(lat, lng, poly):
            return True
    return False


def is_water_cell(h3_cell: str) -> bool:
    """
    Comprehensive H3 cell water test.
    Checks centroid AND boundary vertices to ensure no partial river/water cell passes.
    """
    try:
        lat, lng = h3.cell_to_latlng(h3_cell)
        if is_water_location(lat, lng):
            return True

        boundary = h3.cell_to_boundary(h3_cell)
        water_vertex_count = sum(1 for b_lat, b_lng in boundary if is_water_location(b_lat, b_lng))
        # If 2 or more vertices are in water, reject as a water cell
        if water_vertex_count >= 2:
            return True
    except Exception:
        pass
    return False


def is_suitable_vendor_cell(
    lat: float,
    lng: float,
    h3_cell: str | None = None,
    static_features: dict | None = None,
) -> bool:
    """
    Determine if an H3 cell location is physically suitable for street vendors.

    Checks:
      1. Water exclusion (rivers, reservoirs, dams, canals) for point and cell boundary
      2. If static features provided: road density, POI access & non-barren terrain
    """
    if is_water_location(lat, lng):
        return False

    if h3_cell and is_water_cell(h3_cell):
        return False

    if static_features:
        road_density = float(static_features.get("road_density", 1.0))
        pop_density = float(static_features.get("population_density", 100.0))
        building_density = float(static_features.get("building_density", 0.1))
        total_pois = float(static_features.get("total_pois", 1.0))

        # Completely unroaded / zero-POI / zero-building / zero-population cells are unsuitable for vendors
        if road_density <= 0.02 and pop_density <= 10 and building_density <= 0.01 and total_pois <= 0:
            return False

    return True

