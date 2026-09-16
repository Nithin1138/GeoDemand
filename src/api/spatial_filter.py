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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Water Body Exclusion Polygons (Vijayawada Region & Krishna River Corridor)
# ---------------------------------------------------------------------------

# Krishna River Corridor: Upstream (Kondapalli / Ibrahimpatnam) ->
# Prakasam Barrage -> Downstream (Krishnalanka / Tadepalle / Yanamalakuduru)
KRISHNA_RIVER_POLYGON: List[Tuple[float, float]] = [
    (16.560, 80.495),
    (16.550, 80.525),
    (16.540, 80.555),
    (16.530, 80.580),
    (16.518, 80.600),
    (16.512, 80.614),  # North bank - Durga Ghat / Kanaka Durga
    (16.506, 80.622),  # Prakasam Barrage North / Governorpet riverbank
    (16.495, 80.642),  # Krishnalanka bank
    (16.485, 80.665),  # Ranigarithota bank
    (16.475, 80.690),  # Tarakarama Nagar riverbank
    (16.455, 80.725),  # Downstream exit North
    (16.445, 80.715),  # Downstream exit South
    (16.465, 80.678),  # South bank - Yanamalakuduru / Penumaka
    (16.478, 80.648),  # South bank - Undavalli / Tadepalle
    (16.490, 80.628),  # South bank - Seethanagaram
    (16.502, 80.608),  # Prakasam Barrage South (Seethanagaram hill base)
    (16.512, 80.580),  # South bank upstream (Venkatapalem)
    (16.525, 80.545),  # South bank upstream (Mandadam / Rayapudi)
    (16.540, 80.505),  # South bank upstream (Amaravati bank)
]

# Bhavani Island Backwater Lagoon / River Split
BHAVANI_ISLAND_WATER_POLYGON: List[Tuple[float, float]] = [
    (16.528, 80.575),
    (16.525, 80.590),
    (16.518, 80.592),
    (16.515, 80.580),
    (16.520, 80.568),
]

# Gundala / Eluru Canal reservoir pond & water basin
ELURU_CANAL_BASIN: List[Tuple[float, float]] = [
    (16.520, 80.640),
    (16.522, 80.645),
    (16.518, 80.648),
    (16.515, 80.642),
]

ALL_WATER_POLYGONS = [
    KRISHNA_RIVER_POLYGON,
    BHAVANI_ISLAND_WATER_POLYGON,
    ELURU_CANAL_BASIN,
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


def is_suitable_vendor_cell(
    lat: float,
    lng: float,
    static_features: dict | None = None,
) -> bool:
    """
    Determine if an H3 cell location is physically suitable for street vendors.

    Checks:
      1. Water exclusion (rivers, reservoirs, dams, canals)
      2. If static features provided: road density & POI access
    """
    if is_water_location(lat, lng):
        return False

    if static_features:
        road_density = float(static_features.get("road_density", 1.0))
        pop_density = float(static_features.get("population_density", 100.0))
        building_density = float(static_features.get("building_density", 0.1))

        # Completely unroaded / 0-building / 0-population cells are unsuitable
        if road_density <= 0.01 and pop_density <= 10 and building_density <= 0.01:
            return False

    return True
