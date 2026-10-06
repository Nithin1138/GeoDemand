"""
Real POI data client — OpenStreetMap via the Overpass API.

NO API KEY REQUIRED. Overpass is a free, community-run query service over
OSM data. Rate-limit yourself (this client sleeps between requests) and set
a descriptive User-Agent — Overpass will block anonymous/abusive traffic.

If you want higher rate limits or a more polished API, Geoapify is the
paid alternative — see `GeoapifyClient` at the bottom for where to paste
that key.

USAGE:
    client = OverpassClient(cache_dir="data/raw/osm")
    poi_df = client.get_poi_counts_for_grid(grid_df, radius_m=400)
    # returns one row per h3_cell with real counts per POI category
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tag mapping -> our static feature columns
POI_CATEGORY_TAGS = {
    "office_count": [("office", "*")],
    "college_count": [("amenity", "college"), ("amenity", "university")],
    "school_count": [("amenity", "school")],
    "hospital_count": [("amenity", "hospital"), ("amenity", "clinic")],
    "mall_count": [("shop", "mall"), ("shop", "department_store")],
    "restaurant_count": [("amenity", "restaurant"), ("amenity", "fast_food"), ("amenity", "cafe")],
    "park_count": [("leisure", "park")],
    "bus_stop_count": [("highway", "bus_stop")],
}


class OverpassClient:
    def __init__(self, cache_dir: str | Path = "data/raw/osm", request_delay_s: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay_s = request_delay_s
        self.headers = {"User-Agent": "GeoDemandAI-PortfolioProject/1.0 (educational use)"}

    def _query_count(self, lat: float, lng: float, radius_m: int, tag_key: str, tag_value: str) -> int:
        """Count OSM nodes/ways of a given tag within radius_m of (lat, lng)."""
        tag_filter = f'["{tag_key}"="{tag_value}"]' if tag_value != "*" else f'["{tag_key}"]'
        query = f"""
        [out:json][timeout:25];
        (
          node(around:{radius_m},{lat},{lng}){tag_filter};
          way(around:{radius_m},{lat},{lng}){tag_filter};
        );
        out count;
        """
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Overpass 'out count' returns a single element with tags.total
        elements = data.get("elements", [])
        if elements and "tags" in elements[0]:
            return int(elements[0]["tags"].get("total", 0))
        return 0

    def get_poi_counts_for_cell(self, lat: float, lng: float, radius_m: int = 400) -> dict:
        """Fetch real POI counts for one location across all categories. Rate-limited."""
        counts = {}
        for feature_col, tag_pairs in POI_CATEGORY_TAGS.items():
            total = 0
            for tag_key, tag_value in tag_pairs:
                total += self._query_count(lat, lng, radius_m, tag_key, tag_value)
                time.sleep(self.request_delay_s)  # be polite to the free public instance
            counts[feature_col] = total
        counts["source"] = "real:osm-overpass"
        return counts

    def get_poi_counts_for_grid(
        self, grid_df: pd.DataFrame, radius_m: int = 400, cache_name: str = "poi_counts.json"
    ) -> pd.DataFrame:
        """
        Fetch real POI counts for every cell in an H3 grid DataFrame
        (expects columns: h3_cell, lat, lng). Cached — this is slow
        (one Overpass query per category per cell) so re-runs reuse cache.

        WARNING: for a full-city grid (hundreds of cells x 8 categories),
        this can take a long time and may hit Overpass rate limits. For
        development, run it on a subset first (grid_df.sample(20)), or
        batch multiple categories into one Overpass query (left as an
        exercise — the per-cell method above is the simple, readable version).
        """
        cache_path = self.cache_dir / cache_name
        if cache_path.exists():
            logger.info(f"Loading cached POI counts from {cache_path}")
            return pd.read_json(cache_path)

        rows = []
        for i, row in grid_df.iterrows():
            logger.info(f"Fetching POIs for cell {row['h3_cell']} ({i+1}/{len(grid_df)})")
            try:
                counts = self.get_poi_counts_for_cell(row["lat"], row["lng"], radius_m)
                counts["h3_cell"] = row["h3_cell"]
                rows.append(counts)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Overpass request failed for {row['h3_cell']}: {e}")
                continue

        if not rows:
            logger.warning(
                "No real POI data fetched (network unavailable?). Returning empty "
                "DataFrame — caller should fall back to static_features.py's "
                "synthetic generator for coverage."
            )
            return pd.DataFrame(columns=["h3_cell", "source"] + list(POI_CATEGORY_TAGS.keys()))

        df = pd.DataFrame(rows)
        df.to_json(cache_path, orient="records")
        logger.info(f"Fetched real POI data for {len(df)} cells, cached to {cache_path}")
        return df

    def get_poi_features_for_h3_cell(
        self,
        h3_cell_id: str,
        static_features_path: str | Path | None = None,
    ) -> dict:
        """
        Fast lookup of static POI features for a specific H3 cell from pre-extracted datasets.
        Guarantees zero-network latency for live recommendation pipeline.
        """
        if static_features_path is None:
            static_features_path = (
                Path(__file__).parent.parent.parent / "data" / "processed" / "static_features.parquet"
            )
        static_path = Path(static_features_path)

        if static_path.exists():
            try:
                df = pd.read_parquet(static_path)
                match = df[df["h3_cell_id"] == h3_cell_id]
                if not match.empty:
                    row = match.iloc[0].to_dict()
                    return {
                        "h3_cell_id": h3_cell_id,
                        "hospital_count": int(row.get("hospital_count", 0)),
                        "school_count": int(row.get("school_count", 0)),
                        "college_count": int(row.get("college_count", 0)),
                        "office_count": int(row.get("office_count", 0)),
                        "restaurant_count": int(row.get("restaurant_count", 0)),
                        "mall_count": int(row.get("mall_count", 0)),
                        "park_count": int(row.get("park_count", 0)),
                        "bus_stop_count": int(row.get("bus_stop_count", 0)),
                        "source": "real:osm-overpass-preprocessed",
                        "source_type": "real_periodic",
                        "source_name": "osm-overpass",
                    }
            except Exception as e:
                logger.warning(f"Failed reading static features for {h3_cell_id}: {e}")

        # Safe fallback
        return {
            "h3_cell_id": h3_cell_id,
            "hospital_count": 0,
            "school_count": 0,
            "college_count": 0,
            "office_count": 0,
            "restaurant_count": 0,
            "mall_count": 0,
            "park_count": 0,
            "bus_stop_count": 0,
            "source": "synthetic:osm-fallback",
            "source_type": "synthetic_fallback",
            "source_name": "osm-synthetic",
        }


class GeoapifyClient:
    """
    Alternative/supplementary POI source with a more reliable free tier
    and cleaner category taxonomy than raw Overpass.

    >>> REPLACE THIS <<<
    Get a free API key at https://www.geoapify.com/ (3,000 requests/day free)
    and set it as an environment variable:

        export GEOAPIFY_API_KEY="your_key_here"

    Useful if Overpass rate limits become a bottleneck during real data
    collection for the full grid.
    """

    def __init__(self, api_key: str | None = None):
        import os
        self.api_key = api_key or os.environ.get("GEOAPIFY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Geoapify API key found. Set GEOAPIFY_API_KEY env var, "
                "or pass api_key= explicitly. Get a free key at https://www.geoapify.com/"
            )

    def get_places(self, lat: float, lng: float, radius_m: int, category: str) -> list[dict]:
        url = "https://api.geoapify.com/v2/places"
        params = {
            "categories": category,
            "filter": f"circle:{lng},{lat},{radius_m}",
            "limit": 100,
            "apiKey": self.api_key,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("features", [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OverpassClient()
    # single-cell smoke test
    counts = client.get_poi_counts_for_cell(16.5062, 80.6480, radius_m=400)
    print(counts)
