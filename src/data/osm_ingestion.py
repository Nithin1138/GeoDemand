"""
Real OpenStreetMap Ingestion Pipeline for GeoDemand AI.

Fetches real POIs, transit stations, commercial hubs, and amenities across
Vijayawada from OpenStreetMap (via Overpass API), maps them to H3 cells (res 8),
and writes real aggregated spatial features to static_features.parquet.

Every record is tagged with source provenance:
  source_type: "real_periodic"
  source_name: "openstreetmap_overpass"
  ingested_at: ISO8601 timestamp
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import h3
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
RAW_OSM_DIR = ROOT / "data" / "raw" / "osm"
PROCESSED_DIR = ROOT / "data" / "processed"

# Vijayawada study bounding box: south, west, north, east
VIJAYAWADA_BBOX = "16.38,80.50,16.63,80.79"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_osm_city_pois(bbox: str = VIJAYAWADA_BBOX) -> list[dict]:
    """Fetch all relevant POI nodes and ways for the bounding box in a single query."""
    query = f"""
    [out:json][timeout:90];
    (
      node["amenity"]({bbox});
      way["amenity"]({bbox});
      node["shop"]({bbox});
      way["shop"]({bbox});
      node["office"]({bbox});
      way["office"]({bbox});
      node["leisure"]({bbox});
      way["leisure"]({bbox});
      node["highway"="bus_stop"]({bbox});
      node["amenity"="bus_station"]({bbox});
      node["railway"="station"]({bbox});
      node["railway"="halt"]({bbox});
      node["amenity"="parking"]({bbox});
      way["amenity"="parking"]({bbox});
      node["building"]({bbox});
      way["highway"]({bbox});
    );
    out center;
    """

    RAW_OSM_DIR.mkdir(parents=True, exist_ok=True)
    raw_cache = RAW_OSM_DIR / "osm_vijayawada_pois.json"

    logger.info(f"Querying OpenStreetMap Overpass API for Vijayawada bbox [{bbox}]...")
    headers = {"User-Agent": "GeoDemandAI-RealOSM-Ingestion/2.1 (Academic/Portfolio Research)"}
    
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=100)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])
        logger.info(f"Successfully downloaded {len(elements)} real OSM features.")
        
        # Save raw snapshot with metadata
        snapshot = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source_type": "real_periodic",
            "source_name": "openstreetmap_overpass",
            "bbox": bbox,
            "element_count": len(elements),
            "elements": elements,
        }
        with open(raw_cache, "w") as f:
            json.dump(snapshot, f)
        return elements

    except Exception as e:
        logger.warning(f"Overpass API fetch error: {e}")
        if raw_cache.exists():
            logger.info(f"Loading previous raw OSM snapshot from {raw_cache}")
            with open(raw_cache) as f:
                return json.load(f).get("elements", [])
        raise


def map_and_aggregate_pois(elements: list[dict], h3_cells_df: pd.DataFrame) -> pd.DataFrame:
    """Map OSM elements to H3 resolution 8 cells and compute category counts."""
    valid_cells = set(h3_cells_df["h3_cell_id"].unique())
    
    # Category counters per H3 cell
    counts: Dict[str, Dict[str, int]] = {
        cell: {
            "office_count": 0,
            "college_count": 0,
            "school_count": 0,
            "hospital_count": 0,
            "mall_count": 0,
            "restaurant_count": 0,
            "park_count": 0,
            "bus_stop_count": 0,
            "railway_station_count": 0,
            "metro_station_count": 0,
            "parking_count": 0,
            "building_count": 0,
            "road_segment_count": 0,
        }
        for cell in valid_cells
    }

    now_iso = datetime.now(timezone.utc).isoformat()

    for elem in elements:
        lat = elem.get("lat") or elem.get("center", {}).get("lat")
        lon = elem.get("lon") or elem.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue

        try:
            cell = h3.latlng_to_cell(lat, lon, 8)
        except Exception:
            continue

        if cell not in counts:
            continue

        tags = elem.get("tags", {})
        amenity = tags.get("amenity", "").lower()
        shop = tags.get("shop", "").lower()
        office = tags.get("office", "").lower()
        leisure = tags.get("leisure", "").lower()
        highway = tags.get("highway", "").lower()
        railway = tags.get("railway", "").lower()
        building = tags.get("building", "").lower()

        # Offices / Commercial
        if office or amenity in ("bank", "post_office", "courthouse", "townhall", "police"):
            counts[cell]["office_count"] += 1

        # Higher Education
        if amenity in ("college", "university"):
            counts[cell]["college_count"] += 1

        # Schools
        if amenity in ("school", "kindergarten"):
            counts[cell]["school_count"] += 1

        # Healthcare
        if amenity in ("hospital", "clinic", "pharmacy", "doctors", "dentist"):
            counts[cell]["hospital_count"] += 1

        # Shopping & Malls
        if shop in ("mall", "department_store", "supermarket", "clothes", "electronics") or amenity == "marketplace":
            counts[cell]["mall_count"] += 1

        # Food & Beverage
        if amenity in ("restaurant", "fast_food", "cafe", "food_court", "bar", "ice_cream") or shop in ("bakery", "beverages", "tea", "coffee"):
            counts[cell]["restaurant_count"] += 1

        # Parks & Recreation
        if leisure in ("park", "garden", "playground", "pitch", "sports_centre"):
            counts[cell]["park_count"] += 1

        # Public Transit Bus Stops
        if highway == "bus_stop" or amenity == "bus_station":
            counts[cell]["bus_stop_count"] += 1

        # Rail Stations
        if railway in ("station", "halt"):
            counts[cell]["railway_station_count"] += 1

        # Parking
        if amenity == "parking":
            counts[cell]["parking_count"] += 1

        # Buildings
        if building:
            counts[cell]["building_count"] += 1

        # Roads
        if highway:
            counts[cell]["road_segment_count"] += 1

    # Convert to DataFrame
    rows = []
    for cell, c in counts.items():
        row = {"h3_cell_id": cell, **c}
        rows.append(row)

    df_counts = pd.DataFrame(rows)

    # Compute normalized indices & land use ratios
    max_road = df_counts["road_segment_count"].max() or 1
    max_bld = df_counts["building_count"].max() or 1
    max_off = df_counts["office_count"].max() or 1
    max_rest = df_counts["restaurant_count"].max() or 1

    df_counts["road_density"] = (df_counts["road_segment_count"] / max_road).round(3)
    df_counts["building_density"] = (df_counts["building_count"] / max_bld).round(3)

    # Estimate population density based on building count and residential proximity
    # res-8 cell area ≈ 0.737 km²
    df_counts["population_density"] = (
        (df_counts["building_count"] * 180 + df_counts["road_segment_count"] * 120 + 2500)
        .clip(500, 35000)
        .astype(int)
    )

    # Land use ratios
    comm_score = df_counts["office_count"] / max_off + df_counts["restaurant_count"] / max_rest + df_counts["mall_count"]
    total_poi = df_counts["office_count"] + df_counts["restaurant_count"] + df_counts["school_count"] + df_counts["hospital_count"] + 1

    commercial_ratio = np.clip((comm_score / (comm_score + 2)).round(3), 0.05, 0.90)
    residential_ratio = np.clip((1.0 - commercial_ratio - 0.10).round(3), 0.10, 0.85)
    industrial_ratio = np.clip((1.0 - commercial_ratio - residential_ratio).round(3), 0.0, 0.35)

    df_counts["commercial_ratio"] = commercial_ratio
    df_counts["residential_ratio"] = residential_ratio
    df_counts["industrial_ratio"] = industrial_ratio

    dominant = np.where(
        commercial_ratio > residential_ratio, "commercial", "residential"
    )
    df_counts["land_use_type"] = dominant
    df_counts["source"] = "real:openstreetmap_overpass"
    df_counts["source_type"] = "real_periodic"
    df_counts["source_name"] = "openstreetmap_overpass"
    df_counts["ingested_at"] = now_iso

    # Merge with original lat/lng coordinates from h3_cells
    merged = pd.merge(h3_cells_df[["h3_cell_id", "latitude", "longitude"]], df_counts, on="h3_cell_id")

    return merged


def run_osm_ingestion():
    """Main entrypoint to run real OSM ingestion and save static_features.parquet."""
    h3_path = PROCESSED_DIR / "h3_cells.parquet"
    if not h3_path.exists():
        from data.h3_cells import build_h3_cells
        h3_cells_df = build_h3_cells()
    else:
        h3_cells_df = pd.read_parquet(h3_path)

    elements = fetch_osm_city_pois(VIJAYAWADA_BBOX)
    static_df = map_and_aggregate_pois(elements, h3_cells_df)

    out_path = PROCESSED_DIR / "static_features.parquet"
    static_df.to_parquet(out_path, index=False)
    logger.info(f"Real OpenStreetMap features saved -> {out_path} ({len(static_df)} cells, {len(static_df.columns)} cols)")
    logger.info(f"Sample:\n{static_df[['h3_cell_id', 'office_count', 'restaurant_count', 'hospital_count', 'road_density', 'source_type']].head(5)}")
    return static_df


if __name__ == "__main__":
    run_osm_ingestion()
