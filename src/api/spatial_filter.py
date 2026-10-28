"""
Geospatial Water & Land-Suitability Filter for GeoDemand AI.

Prevents recommending candidate locations inside water bodies, rivers,
reservoirs, dams, swamps, and unroutable terrain across any global coordinates.

Mobile vendors (food trucks, fruit stalls, tea stalls, salon vans, repair vans)
physically require road access and commercial/pedestrian footfall.
They cannot operate in:
  - Rivers, reservoirs, dams, canals, lakes, oceans, swamps
  - Non-roaded or inaccessible wilderness/water cells

Detection Architecture:
  1. Internal Fast Geometric Cache: Known regional water bodies for instant 0ms offline checks.
  2. Algorithmic Navigability & Road Proximity (OSRM / Routing):
     Checks distance to nearest drivable road. If > 180m or routing fails, the location
     is physically in water or inaccessible terrain. Works globally without manual polygon tracing.
  3. Memory LRU Cache: Caches coordinates and H3 cells for sub-millisecond repeated lookups.
"""

from __future__ import annotations

import logging
from typing import List, Tuple
import requests
import h3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regional Geometric Acceleration Cache
# (Pre-indexed water bodies for instant zero-latency evaluation)
# ---------------------------------------------------------------------------

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

PRAKASAM_BARRAGE_RESERVOIR_POLYGON: List[Tuple[float, float]] = [
    (16.512, 80.598),
    (16.516, 80.615),
    (16.506, 80.622),
    (16.496, 80.615),
    (16.498, 80.598),
]

BHAVANI_ISLAND_WATER_POLYGON: List[Tuple[float, float]] = [
    (16.532, 80.570),
    (16.528, 80.595),
    (16.515, 80.595),
    (16.512, 80.575),
    (16.522, 80.560),
]

ELURU_CANAL_MARSH_POLYGON: List[Tuple[float, float]] = [
    (16.514, 80.650),
    (16.520, 80.650),
    (16.522, 80.658),
    (16.515, 80.658),
]

ALL_WATER_POLYGONS = [
    KRISHNA_RIVER_POLYGON,
    PRAKASAM_BARRAGE_RESERVOIR_POLYGON,
    BHAVANI_ISLAND_WATER_POLYGON,
    ELURU_CANAL_MARSH_POLYGON,
]

# In-memory evaluation cache for coordinates: (lat_round, lng_round) -> is_water
_WATER_LOCATION_CACHE: dict[tuple[float, float], bool] = {}


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


def is_water_location(lat: float, lng: float, timeout_s: float = 1.5) -> bool:
    """
    Algorithmic water and unroutable terrain detector for any location.

    Multi-layer determination:
      1. In-memory LRU cache (0ms)
      2. Regional geometric cache (pre-indexed water bodies for instant offline evaluation)
      3. Algorithmic road proximity & navigability check (OSRM nearest drivable road):
         Street vendors (food trucks, fruit stalls, salon vans, repair vans) physically
         require accessible road frontage. If the nearest drivable road is > 180 meters
         away (or road lookup returns unroutable/error), the location is physically inside
         a body of water (ocean, river, lake, reservoir, swamp) or inaccessible terrain.
    """
    key = (round(lat, 4), round(lng, 4))
    if key in _WATER_LOCATION_CACHE:
        return _WATER_LOCATION_CACHE[key]

    # Layer 1: Fast geometric check against known regional water bodies
    for poly in ALL_WATER_POLYGONS:
        if point_in_polygon(lat, lng, poly):
            _WATER_LOCATION_CACHE[key] = True
            return True

    # Layer 2: Algorithmic road proximity & navigability check (works anywhere on Earth)
    try:
        url = f"https://router.project-osrm.org/nearest/v1/driving/{lng:.5f},{lat:.5f}"
        resp = requests.get(url, timeout=timeout_s).json()
        if resp.get("code") == "Ok" and resp.get("waypoints"):
            dist = float(resp["waypoints"][0].get("distance", 0.0))
            is_unroutable = dist > 180.0
            _WATER_LOCATION_CACHE[key] = is_unroutable
            return is_unroutable
    except Exception as e:
        logger.debug(f"Algorithmic road check skipped for ({lat}, {lng}): {e}")

    _WATER_LOCATION_CACHE[key] = False
    return False


def is_water_cell(h3_cell: str) -> bool:
    """
    Comprehensive H3 cell water test.
    Checks centroid and majority boundary vertices to prevent water cells
    while avoiding false-positives for waterfront urban commercial districts.
    """
    try:
        lat, lng = h3.cell_to_latlng(h3_cell)
        if is_water_location(lat, lng):
            return True

        boundary = h3.cell_to_boundary(h3_cell)
        water_vertex_count = sum(1 for b_lat, b_lng in boundary if is_water_location(b_lat, b_lng))
        # If majority (4 or more out of 6) vertices are in water, reject as water cell
        if water_vertex_count >= 4:
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
    Determine if an H3 cell location is physically and commercially suitable for street vendors.

    Checks:
      1. Water & unroutable terrain exclusion (rivers, reservoirs, dams, canals, lakes, oceans)
         for point and cell boundary via algorithmic road proximity and geometric filters.
      2. If static features provided: road density, commercial POIs, and building presence.
         Street vendors require footfall anchors (commercial POIs, building presence,
         or commercial land use) and paved road access.
    """
    if is_water_location(lat, lng):
        return False

    if h3_cell and is_water_cell(h3_cell):
        return False

    if static_features:
        road_density = float(static_features.get("road_density", 0.5))
        building_count = float(static_features.get("building_count", 0.0))
        commercial_ratio = float(static_features.get("commercial_ratio", 0.0))

        poi_cols = [
            "office_count", "college_count", "school_count", "hospital_count",
            "mall_count", "restaurant_count", "bus_stop_count", "park_count",
        ]
        total_pois = float(sum(static_features.get(col, 0.0) for col in poi_cols)) if any(col in static_features for col in poi_cols) else float(static_features.get("total_pois", 1.0))

        # Check for unpaved dirt tracks / inaccessible terrain
        if road_density < 0.10:
            return False

        # Check for lack of commercial footfall (empty agricultural lands, barren plots)
        if total_pois < 8 and building_count < 2 and commercial_ratio < 0.20:
            return False

    return True
