"""
Location context layer — GPS → H3, candidate generation, and land suitability filtering.

Responsibility:
  - Convert GPS (lat, lng) → H3 cell at resolution 8
  - Generate nearby H3 candidate cells within a configurable radius
  - Filter out water bodies (Krishna River, Prakasam Barrage water zone, canals)
  - Calculate centroid, distance, and travel time for each candidate
  - Return provenance-tagged location context
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import h3

from api.spatial_filter import is_water_location, is_suitable_vendor_cell


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LocationContext:
    """GPS → H3 context, returned by POST /v1/location/context."""
    latitude: float
    longitude: float
    h3_cell: str
    h3_resolution: int
    timestamp: str
    timezone_name: str
    is_water: bool = False
    source_type: str = "real_live"
    source_name: str = "device_gps"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class CandidateCell:
    """One H3 candidate cell for demand prediction."""
    h3_cell: str
    centroid_lat: float
    centroid_lng: float
    distance_km: float
    estimated_travel_min: float
    is_current_cell: bool = False
    is_water: bool = False
    ring_distance: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

H3_RESOLUTION = 8


def gps_to_h3(lat: float, lng: float, resolution: int = H3_RESOLUTION) -> str:
    """Convert GPS coordinates to H3 cell index."""
    return h3.latlng_to_cell(lat, lng, resolution)


def h3_centroid(h3_cell: str) -> tuple[float, float]:
    """Get (lat, lng) centroid of an H3 cell."""
    lat, lng = h3.cell_to_latlng(h3_cell)
    return lat, lng


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_travel_min(distance_km: float, avg_speed_kmh: float = 20.0) -> float:
    """Estimate travel time in minutes assuming urban driving speed."""
    return round((distance_km / avg_speed_kmh) * 60, 1)


def generate_candidates(
    origin_lat: float,
    origin_lng: float,
    h3_cell: str,
    search_radius_km: float = 3.0,
    resolution: int = H3_RESOLUTION,
    avg_speed_kmh: float = 20.0,
    exclude_water: bool = True,
) -> list[CandidateCell]:
    """
    Generate nearby H3 cells within search_radius_km of the origin.

    Filters out water bodies (rivers, barrage pools, reservoirs) so mobile
    vendors are never recommended destinations in water.

    Args:
        origin_lat, origin_lng: Vendor's current GPS position
        h3_cell: Current H3 cell (anchor)
        search_radius_km: Maximum search radius (default: 3.0 km)
        resolution: H3 resolution (default: 8)
        avg_speed_kmh: Assumed urban travel speed for time estimate
        exclude_water: When True, filters out cells falling into water polygons

    Returns:
        List[CandidateCell] sorted by distance ascending (current cell first)
    """
    candidates: dict[str, CandidateCell] = {}

    # Current cell anchor
    cur_lat, cur_lng = h3_centroid(h3_cell)
    cur_is_water = is_water_location(cur_lat, cur_lng)
    candidates[h3_cell] = CandidateCell(
        h3_cell=h3_cell,
        centroid_lat=cur_lat,
        centroid_lng=cur_lng,
        distance_km=haversine_km(origin_lat, origin_lng, cur_lat, cur_lng),
        estimated_travel_min=0.0,
        is_current_cell=True,
        is_water=cur_is_water,
        ring_distance=0,
    )

    # Expand rings until the closest cell in the ring exceeds search_radius_km
    for k in range(1, 25):
        ring_cells = h3.grid_ring(h3_cell, k)
        if not ring_cells:
            break
        
        min_ring_dist = float("inf")
        for cell in ring_cells:
            c_lat, c_lng = h3_centroid(cell)
            dist = haversine_km(origin_lat, origin_lng, c_lat, c_lng)
            if dist < min_ring_dist:
                min_ring_dist = dist

            if dist <= search_radius_km and cell not in candidates:
                cell_in_water = is_water_location(c_lat, c_lng)

                # Skip water bodies for destination recommendations
                if exclude_water and cell_in_water:
                    continue

                candidates[cell] = CandidateCell(
                    h3_cell=cell,
                    centroid_lat=round(c_lat, 6),
                    centroid_lng=round(c_lng, 6),
                    distance_km=round(dist, 3),
                    estimated_travel_min=estimate_travel_min(dist, avg_speed_kmh),
                    is_current_cell=False,
                    is_water=cell_in_water,
                    ring_distance=k,
                )

        # Stop expanding only when all cells in the ring are strictly beyond the search radius
        if min_ring_dist > search_radius_km:
            break

    # Strictly filter all destination candidates by search_radius_km
    valid_candidates = [
        c for c in candidates.values()
        if c.is_current_cell or c.distance_km <= search_radius_km
    ]
    return sorted(valid_candidates, key=lambda c: c.distance_km)


def get_location_context(lat: float, lng: float, resolution: int = H3_RESOLUTION) -> LocationContext:
    """
    Convert GPS coordinates to a full location context object.
    This is the response payload for POST /v1/location/context.
    """
    h3_cell = gps_to_h3(lat, lng, resolution)
    c_lat, c_lng = h3_centroid(h3_cell)
    now = datetime.now(timezone.utc)
    return LocationContext(
        latitude=round(lat, 6),
        longitude=round(lng, 6),
        h3_cell=h3_cell,
        h3_resolution=resolution,
        timestamp=now.isoformat(),
        timezone_name="Asia/Kolkata",
        is_water=is_water_location(c_lat, c_lng),
        source_type="real_live",
        source_name="device_gps",
    )


if __name__ == "__main__":
    lat, lng = 16.5062, 80.6480
    ctx = get_location_context(lat, lng)
    print("Location Context:", ctx.to_dict())
    cands = generate_candidates(lat, lng, ctx.h3_cell, search_radius_km=3.0)
    print(f"{len(cands)} land candidates generated (water filtered)")
