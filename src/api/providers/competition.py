"""
Competition provider abstraction layer.

Provides category-aware competition scoring:
  - Tea / Coffee vendor: counts tea stalls, cafes, coffee shops, bakeries
  - Food vendor: counts restaurants, fast food, dhabas, food courts
  - Fruits / Juice vendor: counts fruit shops, juice stalls, bakeries
  - Salon vendor: counts beauty salons, hair dressers, spas
  - Repair vendor: counts automobile garages, repair shops, mechanic workshops

Provider hierarchy:
  1. GooglePlacesCompetitionProvider — real-time category search (Places API New with legacy fallback)
  2. OSMCompetitionProvider          — real OSM static features POI density (real_periodic)
  3. SimulatedCompetitionProvider    — fallback based on simulation dataset

Configured via COMPETITION_PROVIDER in .env:
  - "auto"          -> uses Google Places if key present, else OSM
  - "google_places" -> forces Google Places provider
  - "osm"           -> forces OSM POI density provider
  - "synthetic"     -> forces simulated competition provider
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent


def _score_to_level(score: float) -> str:
    if score < 0.33:
        return "low"
    elif score < 0.67:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

class CompetitionProvider(ABC):
    """Returns competition score for a given H3 cell and vendor category."""

    @abstractmethod
    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        """
        Returns:
          competition_score: float [0,1]
          competition_level: 'low' | 'medium' | 'high'
          source_type: str
          source_name: str
          retrieved_at: str (ISO)
        """

    @property
    @abstractmethod
    def source_type(self) -> str:
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class SimulatedCompetitionProvider(CompetitionProvider):
    """Reads from pre-generated competition.parquet."""
    source_type = "simulated"
    source_name = "market_simulation_engine"

    def __init__(self, competition_path: Optional[Path] = None):
        path = competition_path or (ROOT / "data" / "processed" / "competition.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            self._lookup = df.set_index("h3_cell_id")["competition_score"].to_dict()
        else:
            self._lookup = {}

    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        score = self._lookup.get(h3_cell, 0.5)
        if isinstance(score, pd.Series):
            score = float(score.iloc[0])
        return {
            "competition_score": round(float(score), 4),
            "competition_level": _score_to_level(float(score)),
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": "Competition data is simulated — based on synthetic vendor profiles.",
        }


class OSMCompetitionProvider(CompetitionProvider):
    """
    Derives fine-grained category-aware competition from real OpenStreetMap static features.
    source_type = real_periodic (OSM data refreshed periodically)
    """
    source_type = "real_periodic"
    source_name = "openstreetmap"

    CATEGORY_POI_MAP = {
        "tea": "restaurant_count",
        "coffee": "restaurant_count",
        "food": "restaurant_count",
        "fruits": "restaurant_count",
        "salon": "mall_count",
        "repair": "office_count",
    }

    def __init__(self, static_features_path: Optional[Path] = None):
        path = static_features_path or (ROOT / "data" / "processed" / "static_features.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            self._static = df.set_index("h3_cell_id")
        else:
            self._static = pd.DataFrame()

    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        if self._static.empty or h3_cell not in self._static.index:
            return SimulatedCompetitionProvider().get_competition_score(h3_cell, vendor_category)

        poi_col = self.CATEGORY_POI_MAP.get(vendor_category.lower(), "restaurant_count")
        raw_count = float(self._static.loc[h3_cell, poi_col]) if poi_col in self._static.columns else 2.0

        scale = 15.0 if vendor_category in ("tea", "coffee", "fruits") else 25.0
        score = min(1.0, raw_count / scale)

        return {
            "competition_score": round(score, 4),
            "competition_level": _score_to_level(score),
            "raw_competitor_count": int(raw_count),
            "poi_column_used": poi_col,
            "vendor_category": vendor_category,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": f"Category-aware competition derived from real OSM {poi_col} counts (real_periodic).",
        }


class GooglePlacesCompetitionProvider(CompetitionProvider):
    """
    Live category-aware competition via Google Places API (New) with legacy endpoint fallback.
    Set GOOGLE_PLACES_API_KEY env var.
    """
    source_type = "real_live"
    source_name = "google_places_api"

    CATEGORY_INCLUDED_TYPES = {
        "tea": ["cafe", "coffee_shop", "bakery"],
        "coffee": ["cafe", "coffee_shop"],
        "food": ["restaurant", "fast_food_restaurant", "meal_takeaway"],
        "fruits": ["grocery_store", "supermarket"],
        "salon": ["beauty_salon", "hair_care", "spa"],
        "repair": ["car_repair", "auto_parts_store"],
    }

    CATEGORY_KEYWORDS = {
        "tea": "tea stall cafe chai",
        "coffee": "coffee shop cafe",
        "food": "restaurant fast food meals dhaba",
        "fruits": "fruit shop juice stall",
        "salon": "beauty salon hair salon barber",
        "repair": "automobile repair mechanic tyre shop garage",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_PLACES_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GOOGLE_PLACES_API_KEY not set.")
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 1800  # 30 min

    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        import h3
        lat, lng = h3.cell_to_latlng(h3_cell)
        cache_key = f"{h3_cell}_{vendor_category}"
        now_ts = time.time()

        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if now_ts - entry["ts"] < self._cache_ttl:
                return {**entry["data"], "from_cache": True, "cache_age_seconds": round(now_ts - entry["ts"])}

        cat_lower = vendor_category.lower()

        # 1. Attempt Google Places API (New) searchNearby
        try:
            url_new = "https://places.googleapis.com/v1/places:searchNearby"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "places.displayName,places.primaryType",
            }
            body = {
                "includedTypes": self.CATEGORY_INCLUDED_TYPES.get(cat_lower, ["restaurant"]),
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": 500.0,
                    }
                },
            }
            resp = requests.post(url_new, json=body, headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                places = data.get("places", [])
                count = len(places)
                score = min(1.0, count / 15.0)
                res = {
                    "competition_score": round(score, 4),
                    "competition_level": _score_to_level(score),
                    "raw_competitor_count": count,
                    "vendor_category": vendor_category,
                    "source_type": self.source_type,
                    "source_name": "google_places_api_new",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "note": f"Live Google Places API (New): {count} competitors found.",
                }
                self._cache[cache_key] = {"data": res, "ts": now_ts}
                return res
        except Exception as e:
            logger.info(f"Google Places API (New) attempt: {e}. Trying legacy endpoint fallback.")

        # 2. Legacy Nearby Search Fallback
        try:
            keyword = self.CATEGORY_KEYWORDS.get(cat_lower, vendor_category)
            url_legacy = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{lat},{lng}",
                "radius": 500,
                "keyword": keyword,
                "key": self.api_key,
            }
            resp = requests.get(url_legacy, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            count = len(results)
            score = min(1.0, count / 15.0)
            res = {
                "competition_score": round(score, 4),
                "competition_level": _score_to_level(score),
                "raw_competitor_count": count,
                "vendor_category": vendor_category,
                "source_type": self.source_type,
                "source_name": "google_places_api_legacy",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "note": f"Live Google Places API (Legacy): {count} competitors found.",
            }
            self._cache[cache_key] = {"data": res, "ts": now_ts}
            return res
        except Exception as e:
            logger.warning(f"Google Places API failed ({e}). Falling back to OSM.")
            return OSMCompetitionProvider().get_competition_score(h3_cell, vendor_category)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class CompetitionProviderFactory:
    @staticmethod
    def create(preferred: str = "auto") -> CompetitionProvider:
        pref = preferred.lower() if preferred != "auto" else os.getenv("COMPETITION_PROVIDER", "auto").lower()

        if pref == "synthetic":
            return SimulatedCompetitionProvider()

        if pref in ("google_places", "real", "live") or (pref == "auto" and os.environ.get("GOOGLE_PLACES_API_KEY")):
            try:
                return GooglePlacesCompetitionProvider()
            except Exception as e:
                logger.info(f"Google Places unavailable ({e}) — using OSM competition provider.")

        return OSMCompetitionProvider()

    get_provider = create
