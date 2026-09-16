"""
Dataset 2 — Static Spatial Features (static_features.parquet)

Single responsibility: rarely-changing per-cell spatial attributes.
Pipeline Order:
  1. Attempt real OpenStreetMap Overpass ingestion (src/data/osm_ingestion.py)
  2. If OSM succeeds: mark POIs as real_periodic, population_density as real_proxy,
     and land-use ratios as derived_from_real_osm.
  3. If OSM network unavailable: fall back to synthetic spatial generator,
     marked explicitly as synthetic_fallback.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from static_features_synth import generate_synthetic_static_features as _base_synth


def _smooth_field(lat_n, lng_n, freq_a, freq_b, phase_a, phase_b):
    field = np.sin(2 * np.pi * freq_a * lat_n + phase_a) * np.cos(2 * np.pi * freq_b * lng_n + phase_b)
    return (field - field.min()) / (field.max() - field.min() + 1e-9)


def build_static_features(h3_cells: pd.DataFrame, seed: int = 42, use_real_osm: bool = True) -> pd.DataFrame:
    """
    Builds static_features DataFrame for the given H3 cells.
    Tries real OpenStreetMap Overpass ingestion first; falls back to synthetic if offline.
    """
    if use_real_osm:
        try:
            from osm_ingestion import fetch_osm_city_pois, map_and_aggregate_pois
            logger.info("Attempting real OpenStreetMap Overpass POI ingestion...")
            elements = fetch_osm_city_pois()
            if elements and len(elements) > 100:
                osm_df = map_and_aggregate_pois(elements, h3_cells)
                osm_df["source"] = "real:openstreetmap_overpass"
                osm_df["source_type"] = "real_periodic"
                osm_df["source_name"] = "openstreetmap_overpass"
                osm_df["population_density_provenance"] = "real_proxy"
                osm_df["land_use_provenance"] = "derived_from_real_osm"
                logger.info(f"Real OpenStreetMap features successfully generated for {len(osm_df)} cells.")
                return osm_df
        except Exception as e:
            logger.warning(f"Real OSM ingestion failed ({e}). Falling back to synthetic spatial generator.")

    # Synthetic fallback path
    logger.info("Generating synthetic static spatial features fallback...")
    grid = h3_cells.rename(columns={"h3_cell_id": "h3_cell", "latitude": "lat", "longitude": "lng"})
    base = _base_synth(grid, seed=seed)
    base = base.rename(columns={"h3_cell": "h3_cell_id"})

    rng = np.random.default_rng(seed + 1)
    lat_n = (grid["lat"] - grid["lat"].min()) / (grid["lat"].max() - grid["lat"].min() + 1e-9)
    lng_n = (grid["lng"] - grid["lng"].min()) / (grid["lng"].max() - grid["lng"].min() + 1e-9)

    railway = _smooth_field(lat_n, lng_n, 0.8, 0.5, *rng.uniform(0, 2 * np.pi, 2))
    metro = _smooth_field(lat_n, lng_n, 1.2, 0.9, *rng.uniform(0, 2 * np.pi, 2))
    parking = _smooth_field(lat_n, lng_n, 2.2, 2.8, *rng.uniform(0, 2 * np.pi, 2))
    building = _smooth_field(lat_n, lng_n, 3.0, 2.5, *rng.uniform(0, 2 * np.pi, 2))

    base["railway_station_count"] = (railway * 2).round().astype(int)
    base["metro_station_count"] = (metro * 1.5).round().astype(int)
    base["parking_count"] = (parking * 10).round().astype(int)
    base["building_density"] = building.round(3)

    office_n = base["office_count"] / (base["office_count"].max() + 1e-9)
    pop_n = base["population_density"] / (base["population_density"].max() + 1e-9)
    commercial_ratio = np.clip(0.5 * office_n + 0.3 * building + rng.normal(0, 0.05, len(base)), 0, 1)
    residential_ratio = np.clip(0.6 * pop_n + rng.normal(0, 0.05, len(base)), 0, 1)
    industrial_ratio = np.clip(1 - commercial_ratio - residential_ratio, 0, None)
    total = commercial_ratio + residential_ratio + industrial_ratio
    total = np.where(total == 0, 1, total)
    base["commercial_ratio"] = (commercial_ratio / total).round(3)
    base["residential_ratio"] = (residential_ratio / total).round(3)
    base["industrial_ratio"] = (industrial_ratio / total).round(3)

    dominant_idx = base[["residential_ratio", "commercial_ratio", "industrial_ratio"]].to_numpy().argmax(axis=1)
    base["land_use_type"] = np.array(["residential", "commercial", "industrial"])[dominant_idx]
    base["source"] = "real-proxy:osm_synthetic_fallback"
    base["source_type"] = "synthetic_fallback"
    base["source_name"] = "osm_synthetic_fallback"
    base["population_density_provenance"] = "synthetic_fallback"
    base["land_use_provenance"] = "synthetic_fallback"

    return base


if __name__ == "__main__":
    from h3_cells import build_h3_cells

    cells = build_h3_cells()
    df = build_static_features(cells)
    out = ROOT / "data" / "processed" / "static_features.parquet"
    df.to_parquet(out, index=False)
    print(f"static_features.parquet: {len(df)} cells, {len(df.columns)} columns -> {out}")
    print(df[["h3_cell_id", "office_count", "restaurant_count", "population_density", "source_type"]].head())
