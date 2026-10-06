"""
Dataset 1 — H3 Spatial Grid (h3_cells.parquet)

Single responsibility: defines every geographical cell used everywhere
else in the pipeline. Every other dataset joins to this one on h3_cell_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h3
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from h3_grid import generate_city_grid, DEFAULT_H3_RESOLUTION  # noqa: E402
try:
    from spatial_filter import is_water_location  # noqa: E402
except ImportError:
    from src.api.spatial_filter import is_water_location  # noqa: E402


def build_h3_cells(
    center_lat: float = 16.5062,
    center_lng: float = 80.6480,
    radius_km: float = 8.0,
    resolution: int = DEFAULT_H3_RESOLUTION,
    city: str = "Vijayawada",
    state: str = "Andhra Pradesh",
    country: str = "India",
) -> pd.DataFrame:
    grid = generate_city_grid(center_lat, center_lng, radius_km, resolution)
    grid = grid.rename(columns={"h3_cell": "h3_cell_id", "h3_resolution": "resolution"})
    grid["latitude"] = grid["lat"]
    grid["longitude"] = grid["lng"]
    grid["city"] = city
    grid["state"] = state
    grid["country"] = country
    grid["boundary_area_km2"] = grid["h3_cell_id"].apply(
        lambda c: h3.cell_area(c, unit="km^2")
    )
    grid["is_water"] = grid.apply(
        lambda r: is_water_location(r["latitude"], r["longitude"]), axis=1
    )
    grid["source"] = "real:spatial_indexing"
    return grid[[
        "h3_cell_id", "latitude", "longitude", "resolution",
        "city", "state", "country", "boundary_area_km2", "is_water", "source",
    ]]


if __name__ == "__main__":
    df = build_h3_cells()
    out = Path(__file__).parent.parent.parent / "data" / "processed" / "h3_cells.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"h3_cells.parquet: {len(df)} cells -> {out}")
    print(df.head())
