"""
Static feature layer for GeoDemand AI.

Assigns rarely-changing spatial attributes to each H3 cell: population,
POI counts (offices, colleges, malls, hospitals, bus stops, restaurants,
parks), road density.

Two modes:
  - "real": pull from OpenStreetMap via Overpass API (src/features/osm_client.py)
  - "synthetic": generate spatially-correlated synthetic values for cells
    where real data is missing/sparse, so the simulation has full coverage.

For local dev / repeated runs we default to synthetic generation seeded by
cell geography, so results are deterministic and don't hammer the Overpass
API. The real OSM client (Phase 1b) can overwrite these with actual data
for the launch zone, and any remaining gaps stay synthetic — this hybrid
is the "Hybrid Data Collection" layer in the architecture, and every row
carries a `source` tag so it's always clear which is which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


STATIC_FEATURE_COLUMNS = [
    "population_density",
    "office_count",
    "college_count",
    "school_count",
    "hospital_count",
    "mall_count",
    "restaurant_count",
    "park_count",
    "bus_stop_count",
    "road_density",
]


def generate_synthetic_static_features(
    grid: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate spatially-correlated synthetic static features for each H3 cell.

    Uses smooth 2D noise fields (sum of sinusoids at different frequencies,
    per feature) rather than i.i.d. random draws per cell, so that
    neighboring cells have correlated values — real cities aren't random
    noise, POI density is spatially clustered (downtown cores, college
    zones, residential belts).
    """
    rng = np.random.default_rng(seed)
    df = grid.copy()

    lat = df["lat"].to_numpy()
    lng = df["lng"].to_numpy()
    lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9)
    lng_n = (lng - lng.min()) / (lng.max() - lng.min() + 1e-9)

    def smooth_field(freq_a, freq_b, phase_a, phase_b, weight=1.0):
        field = (
            np.sin(2 * np.pi * freq_a * lat_n + phase_a)
            * np.cos(2 * np.pi * freq_b * lng_n + phase_b)
        )
        # normalize to [0, 1]
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        return field * weight

    # Each feature gets its own smooth field with a distinct random phase/freq,
    # so "office density" and "college density" are correlated with geography
    # but NOT identical to each other or to population — avoids the trivial
    # linear-combination trap called out in the review.
    params = {
        "population_density": (2.0, 1.5),
        "office_count": (3.0, 2.0),
        "college_count": (1.0, 4.0),
        "school_count": (2.5, 1.0),
        "hospital_count": (1.5, 1.5),
        "mall_count": (4.0, 3.0),
        "restaurant_count": (2.0, 3.5),
        "park_count": (1.0, 2.0),
        "bus_stop_count": (3.5, 1.0),
        "road_density": (2.0, 2.0),
    }

    for feat, (fa, fb) in params.items():
        phase_a, phase_b = rng.uniform(0, 2 * np.pi, size=2)
        base = smooth_field(fa, fb, phase_a, phase_b)
        noise = rng.normal(0, 0.05, size=len(df))
        val = np.clip(base + noise, 0, 1)
        df[feat] = val

    # Rescale [0,1] fields to plausible integer/real counts per H3 cell
    # (res-8 cell ≈ 0.7 km²)
    df["population_density"] = (df["population_density"] * 25000).round().astype(int)  # people/km2-ish
    df["office_count"] = (df["office_count"] * 40).round().astype(int)
    df["college_count"] = (df["college_count"] * 3).round().astype(int)
    df["school_count"] = (df["school_count"] * 6).round().astype(int)
    df["hospital_count"] = (df["hospital_count"] * 4).round().astype(int)
    df["mall_count"] = (df["mall_count"] * 3).round().astype(int)
    df["restaurant_count"] = (df["restaurant_count"] * 25).round().astype(int)
    df["park_count"] = (df["park_count"] * 5).round().astype(int)
    df["bus_stop_count"] = (df["bus_stop_count"] * 12).round().astype(int)
    df["road_density"] = df["road_density"].round(3)  # keep as normalized 0-1 index

    df["source"] = "synthetic"
    return df


if __name__ == "__main__":
    from h3_grid import generate_city_grid

    grid = generate_city_grid(center_lat=16.5062, center_lng=80.6480, radius_km=8.0)
    feats = generate_synthetic_static_features(grid)
    print(feats[["h3_cell"] + STATIC_FEATURE_COLUMNS].describe())
