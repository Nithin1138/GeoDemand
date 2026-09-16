"""
H3 grid setup for GeoDemand AI.

Generates the set of H3 cells covering the simulated city area, and provides
helpers for assigning static (rarely-changing) spatial attributes to each cell.
This is the spatial backbone every other module (feature store, simulation,
model, dashboard) keys off of.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import h3
except ImportError as e:
    raise ImportError(
        "h3 is required. Install with: pip install h3 --break-system-packages"
    ) from e


# Resolution 8 ≈ ~460m average hexagon edge, ~0.7 km² area.
# This is a reasonable "walkable neighborhood" size for a mobile vendor.
DEFAULT_H3_RESOLUTION = 8


def generate_city_grid(
    center_lat: float,
    center_lng: float,
    radius_km: float = 8.0,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> pd.DataFrame:
    """
    Generate all H3 cells within `radius_km` of a center point.

    Returns a DataFrame with one row per H3 cell: h3_cell, lat, lng (centroid).
    """
    center_cell = h3.latlng_to_cell(center_lat, center_lng, resolution)

    # Approximate number of "rings" needed to cover radius_km.
    # Edge length at res 8 is ~0.46km; each ring adds roughly one edge length.
    edge_len_km = h3.average_hexagon_edge_length(resolution, unit="km")
    k = max(1, int(np.ceil(radius_km / edge_len_km)))

    cells = h3.grid_disk(center_cell, k)

    rows = []
    for cell in cells:
        lat, lng = h3.cell_to_latlng(cell)
        rows.append({"h3_cell": cell, "lat": lat, "lng": lng})

    df = pd.DataFrame(rows)
    df["h3_resolution"] = resolution
    return df


def cell_distance_km(cell_a: str, cell_b: str) -> float:
    """Great-circle-ish distance between two H3 cell centroids, in km."""
    lat1, lng1 = h3.cell_to_latlng(cell_a)
    lat2, lng2 = h3.cell_to_latlng(cell_b)
    return _haversine_km(lat1, lng1, lat2, lng2)


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


if __name__ == "__main__":
    # Quick smoke test — Vijayawada city center as the simulated launch zone
    grid = generate_city_grid(center_lat=16.5062, center_lng=80.6480, radius_km=8.0)
    print(f"Generated {len(grid)} H3 cells at resolution {DEFAULT_H3_RESOLUTION}")
    print(grid.head())
