"""
Competition Provider Abstraction Layer & Intelligence Subsystem.

Provides category-aware competitor extraction, operating-hours parsing,
H3 spatial aggregation, multi-provider integration (Google Places, OSM Overpass,
local OSM snapshot), rate-limit resilience, caching, and scheduled refresh.

Provider Hierarchy:
  1. GooglePlacesCompetitionProvider — real-time Places API (New & Legacy Nearby Search)
  2. OSMOverpassCompetitionProvider   — live OpenStreetMap Overpass QL queries with rate-limit circuit breaking
  3. OSMCompetitionProvider          — local high-speed OSM POI snapshot (real_periodic, 0ms offline)
  4. SimulatedCompetitionProvider    — synthetic competition baseline from competition.parquet

Configured via COMPETITION_PROVIDER in .env:
  - "auto"          -> Google Places if API key present, else OSM Overpass / OSM Snapshot
  - "google_places" -> forces Google Places API
  - "osm_overpass"  -> forces live OSM Overpass API queries
  - "osm"           -> forces local OpenStreetMap POI snapshot
  - "synthetic"     -> forces simulated competition engine
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import requests

try:
    import h3
except ImportError:
    h3 = None

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent
LOCAL_OSM_POI_PATH = ROOT / "data" / "raw" / "osm" / "osm_vijayawada_pois.json"
STATIC_FEATURES_PATH = ROOT / "data" / "processed" / "static_features.parquet"
COMPETITION_PARQUET_PATH = ROOT / "data" / "processed" / "competition.parquet"

# Core monitored deployment H3 cells for scheduled background refresh
DEFAULT_MONITORED_H3_CELLS = [
    "88619aa6c7fffff",  # Governorpet Commercial
    "88619aa6c1fffff",  # Benz Circle Hub
    "88619aa6c5fffff",  # Vijayawada Railway Station / Junction
    "88619aa6c3fffff",  # Auto Nagar Industrial / Commercial
    "88619aa6cbfffff",  # Gunadala Commercial Suburb
    "88619aa6d7fffff",  # Bhavanipuram / West Terminal
    "88619aa6d3fffff",  # Krishnalanka Riverside
    "88619aa455fffff",  # Tadepalle South Gateway
]


# ---------------------------------------------------------------------------
# Category Mappings & Normalization
# ---------------------------------------------------------------------------

AMENITY_TO_CATEGORY: Dict[str, str] = {
    # Food & meals
    "restaurant": "food",
    "fast_food": "food",
    "food_court": "food",
    "dhaba": "food",
    "tiffin": "food",
    "meal_takeaway": "food",
    # Beverages & tea/coffee
    "cafe": "beverages",
    "coffee_shop": "beverages",
    "tea": "beverages",
    "tea_stall": "beverages",
    "juice_bar": "beverages",
    "bar": "beverages",
    "pub": "beverages",
    # Fresh fruits & produce
    "greengrocer": "fruits",
    "fruit_vendor": "fruits",
    "fruit_shop": "fruits",
    # Vegetables & markets
    "vegetables": "vegetables",
    "marketplace": "vegetables",
    "market": "vegetables",
    # Snacks & confectionery
    "bakery": "snacks",
    "ice_cream": "snacks",
    "confectionery": "snacks",
    "sweet_shop": "snacks",
    # Personal grooming & salon
    "hairdresser": "salon",
    "beauty_salon": "salon",
    "barber": "salon",
    "spa": "salon",
    # Vehicle and equipment repairs
    "car_repair": "repair",
    "auto_repair": "repair",
    "bicycle_repair": "repair",
    "motorcycle_repair": "repair",
    "electronics_repair": "repair",
    "tyres": "repair",
    "tyre_shop": "repair",
    # General retail
    "supermarket": "general",
    "convenience": "general",
    "department_store": "general",
    "general_store": "general",
    "kiosk": "general",
}

GOOGLE_TYPE_TO_CATEGORY: Dict[str, str] = {
    "restaurant": "food",
    "fast_food_restaurant": "food",
    "meal_takeaway": "food",
    "cafe": "beverages",
    "coffee_shop": "beverages",
    "bar": "beverages",
    "bakery": "snacks",
    "grocery_store": "fruits",
    "supermarket": "fruits",
    "beauty_salon": "salon",
    "hair_care": "salon",
    "spa": "salon",
    "barber_shop": "salon",
    "car_repair": "repair",
    "auto_parts_store": "repair",
}

CATEGORY_SYNONYMS: Dict[str, str] = {
    "chai_samosa": "beverages",
    "tea": "beverages",
    "coffee": "beverages",
    "juice": "beverages",
    "tiffin_breakfast": "food",
    "biryani_rice": "food",
    "fast_food": "food",
    "fresh_fruits": "fruits",
    "produce": "fruits",
    "greens": "vegetables",
    "mechanic": "repair",
    "garage": "repair",
    "barber": "salon",
}


def normalize_vendor_category(category: str) -> str:
    """Normalizes vendor category or synonym into canonical category."""
    cat_clean = category.lower().strip()
    return CATEGORY_SYNONYMS.get(cat_clean, cat_clean)


def _score_to_level(score: float) -> str:
    """Maps continuous competition score [0.0, 1.0] to discrete ordinal level."""
    if score < 0.33:
        return "low"
    elif score < 0.67:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Operating Hours Parser
# ---------------------------------------------------------------------------

def parse_operating_hours_detailed(
    hours_str: Optional[str],
    category: str = "food",
    reference_dt: Optional[datetime] = None,
    now: Optional[datetime] = None,
    current_hour: Optional[int] = None,
) -> Tuple[Optional[bool], bool, bool, str]:
    """
    Parses OpenStreetMap opening_hours or Google Places weekday descriptions.
    
    Returns:
      (is_open_now, is_daytime_active, is_nighttime_active, operating_hours_source)
      where operating_hours_source is 'provider_explicit', 'category_default', or 'unknown'.
    """
    eval_now = now or reference_dt or datetime.now(timezone.utc)
    cur_hour = current_hour if current_hour is not None else eval_now.hour
    cur_min = eval_now.minute
    cur_total_min = cur_hour * 60 + cur_min

    # Case 1: Missing or empty operating hours -> apply category domain heuristics
    if not hours_str or not isinstance(hours_str, str) or not hours_str.strip():
        cat = normalize_vendor_category(category)
        if cat in ("food", "beverages"):
            # Street food & tea vendors operate across lunch and dinner shifts
            is_day = True
            is_night = True
        elif cat in ("fruits", "vegetables"):
            # Morning to early evening market shifts
            is_day = True
            is_night = False
        elif cat in ("salon", "repair"):
            # Standard daytime / commercial business hours (09:00 - 20:00)
            is_day = True
            is_night = False
        else:
            is_day = True
            is_night = False

        # Open estimate: 06:00 to 22:00 for active shifts
        is_open_now = (6 <= cur_hour < 22) if (is_day or is_night) else False
        return (is_open_now, is_day, is_night, "category_default")

    normalized = hours_str.strip().lower()

    # Case 2: 24/7 continuous operation
    if "24/7" in normalized:
        return (True, True, True, "provider_explicit")

    # Case 3: Explicitly closed
    if "closed" in normalized or "off" in normalized:
        return (False, False, False, "provider_explicit")

    # Case 4: Standard hour range syntax (e.g., "08:00-22:00", "09:00 - 21:00", "18:00-02:00")
    matches = re.findall(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", normalized)
    if not matches:
        is_day = "day" in normalized or "morning" in normalized or not ("night" in normalized or "evening" in normalized)
        is_night = "night" in normalized or "evening" in normalized
        is_open_now = (6 <= cur_hour < 18) if is_day else ((cur_hour >= 18 or cur_hour < 6) if is_night else None)
        return (is_open_now, is_day, is_night, "provider_explicit")

    is_day = False
    is_night = False
    is_open_now = False

    for sh, sm, eh, em in matches:
        s_min = int(sh) * 60 + int(sm)
        e_min = int(eh) * 60 + int(em)

        # Daytime check: overlaps with 06:00–18:00 (360–1080 min)
        if s_min < 1080 and (e_min > 360 or e_min < s_min):
            is_day = True

        # Nighttime check: overlaps with 18:00–06:00 (>=1080 or <=360)
        if e_min > 1080 or e_min <= 360 or s_min >= 1080 or e_min < s_min:
            is_night = True

        # Open right now evaluation
        if s_min <= e_min:
            if s_min <= cur_total_min <= e_min:
                is_open_now = True
        else:
            # Shift spans across midnight (e.g. 18:00 to 02:00)
            if cur_total_min >= s_min or cur_total_min <= e_min:
                is_open_now = True

    return (is_open_now, is_day, is_night, "provider_explicit")


# ---------------------------------------------------------------------------
# Normalized Competitor Entity
# ---------------------------------------------------------------------------

@dataclass
class CompetitorEntity:
    """Normalized structured competitor business entity."""
    competitor_id: str
    name: str
    category: str
    raw_type: str
    latitude: float
    longitude: float
    h3_cell: str
    opening_hours: Optional[str] = None
    is_open_now: Optional[bool] = None
    is_daytime_active: bool = True
    is_nighttime_active: bool = False
    operating_hours_source: str = "unknown"
    source: str = "openstreetmap"
    rating: Optional[float] = None
    address: Optional[str] = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competitor_id": self.competitor_id,
            "name": self.name,
            "category": self.category,
            "raw_type": self.raw_type,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "h3_cell": self.h3_cell,
            "opening_hours": self.opening_hours,
            "is_open_now": self.is_open_now,
            "is_daytime_active": self.is_daytime_active,
            "is_nighttime_active": self.is_nighttime_active,
            "operating_hours_source": self.operating_hours_source,
            "source": self.source,
            "rating": self.rating,
            "address": self.address,
            "retrieved_at": self.retrieved_at,
        }


def aggregate_competitors_for_cell(
    competitors: List[CompetitorEntity],
    h3_cell: str,
    vendor_category: str,
) -> Dict[str, Any]:
    """
    Aggregates a list of competitor entities into H3 cell-level competition metrics.
    """
    target_cat = normalize_vendor_category(vendor_category)
    total_count = len(competitors)
    same_cat_count = 0
    active_now_count = 0
    cat_counts: Dict[str, int] = {}

    for comp in competitors:
        c_cat = normalize_vendor_category(comp.category)
        cat_counts[c_cat] = cat_counts.get(c_cat, 0) + 1
        if c_cat == target_cat:
            same_cat_count += 1
        if comp.is_open_now is True:
            active_now_count += 1

    scale = 15.0 if target_cat in ("beverages", "tea", "coffee", "fruits", "chai_samosa") else 25.0
    score = min(1.0, round(same_cat_count / scale, 4))
    level = _score_to_level(score)

    return {
        "h3_cell": h3_cell,
        "vendor_category": vendor_category,
        "total_competitors": total_count,
        "same_category_competitors": same_cat_count,
        "active_competitors_now": active_now_count,
        "category_counts": cat_counts,
        "competition_score": score,
        "competition_level": level,
        "raw_competitor_count": same_cat_count,
    }


# ---------------------------------------------------------------------------
# Common Competitor Provider Interface
# ---------------------------------------------------------------------------

class CompetitionProvider(ABC):
    """
    Common competitor provider interface.
    Provides category-aware competition scoring, competitor entity retrieval,
    H3 spatial aggregation, caching, and scheduled refresh.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """'real_live', 'real_periodic', or 'synthetic_fallback'."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Provider identification string."""
        ...

    @abstractmethod
    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        """
        Calculates normalized competition score [0.0, 1.0] and metadata.
        """
        ...

    @abstractmethod
    def get_competitors(
        self,
        h3_cell: str,
        vendor_category: Optional[str] = None,
        radius_m: float = 600.0,
    ) -> List[CompetitorEntity]:
        """
        Retrieves concrete competitor entities located in or around the H3 cell.
        """
        ...

    @abstractmethod
    def get_cell_competition(
        self,
        h3_cell: str,
        vendor_category: str,
    ) -> Dict[str, Any]:
        """
        Aggregates competitors by H3 cell and category, returning count breakdown,
        operating hours status, and freshness metadata.
        """
        ...

    def refresh(
        self,
        h3_cells: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes proactive competitor refresh for specified or monitored H3 cells.
        """
        cells = h3_cells or DEFAULT_MONITORED_H3_CELLS
        refreshed = 0
        total_comps = 0
        for cell in cells:
            comps = self.get_competitors(cell)
            total_comps += len(comps)
            refreshed += 1

        return {
            "status": "COMPLETED",
            "refreshed_cells_count": refreshed,
            "total_competitors_found": total_comps,
            "provider_used": self.source_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Implementation 1: Simulated Competition Provider
# ---------------------------------------------------------------------------

class SimulatedCompetitionProvider(CompetitionProvider):
    """Reads from pre-generated competition.parquet with synthetic fallback generation."""
    source_type = "simulated"
    source_name = "market_simulation_engine"

    def __init__(self, competition_path: Optional[Path] = None):
        path = competition_path or COMPETITION_PARQUET_PATH
        if path.exists():
            df = pd.read_parquet(path)
            self._lookup = df.set_index("h3_cell_id")["competition_score"].to_dict()
        else:
            self._lookup = {}

    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        score = self._lookup.get(h3_cell, 0.5)
        if isinstance(score, pd.Series):
            score = float(score.iloc[0])
        score_f = round(float(score), 4)
        return {
            "competition_score": score_f,
            "competition_level": _score_to_level(score_f),
            "raw_competitor_count": int(round(score_f * 20)),
            "vendor_category": vendor_category,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": "Competition data is simulated — based on synthetic vendor profiles.",
        }

    def get_competitors(
        self,
        h3_cell: str,
        vendor_category: Optional[str] = None,
        radius_m: float = 600.0,
    ) -> List[CompetitorEntity]:
        score_data = self.get_competition_score(h3_cell, vendor_category or "food")
        count = max(1, score_data.get("raw_competitor_count", 5))
        lat, lng = (16.5062, 80.6480)
        if h3 is not None:
            try:
                lat, lng = h3.cell_to_latlng(h3_cell)
            except Exception:
                pass

        cat = vendor_category or "food"
        results: List[CompetitorEntity] = []
        for i in range(min(count, 8)):
            c_lat = lat + (0.001 * (i % 3 - 1))
            c_lng = lng + (0.001 * ((i // 3) % 3 - 1))
            results.append(CompetitorEntity(
                competitor_id=f"sim_{h3_cell[:6]}_{i}",
                name=f"Simulated {cat.capitalize()} Vendor #{i+1}",
                category=cat,
                raw_type="street_stall",
                latitude=c_lat,
                longitude=c_lng,
                h3_cell=h3_cell,
                opening_hours="09:00-21:00",
                is_open_now=True,
                is_daytime_active=True,
                is_nighttime_active=(i % 2 == 0),
                operating_hours_source="category_default",
                source=self.source_name,
            ))
        return results

    def get_cell_competition(self, h3_cell: str, vendor_category: str) -> Dict[str, Any]:
        competitors = self.get_competitors(h3_cell, vendor_category)
        agg = aggregate_competitors_for_cell(competitors, h3_cell, vendor_category)
        agg["competitors"] = [c.to_dict() for c in competitors]
        agg["freshness"] = {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "from_cache": True,
            "cache_age_seconds": 0.0,
        }
        return agg


# ---------------------------------------------------------------------------
# Implementation 2: OSM Competition Provider (Local High-Speed Snapshot)
# ---------------------------------------------------------------------------

class OSMCompetitionProvider(CompetitionProvider):
    """
    Derives fine-grained category-aware competition from real OpenStreetMap static POI features.
    source_type = real_periodic (OSM data refreshed periodically, executes in 0ms offline)
    """
    source_type = "real_periodic"
    source_name = "openstreetmap"

    CATEGORY_POI_MAP = {
        "chai_samosa": "restaurant_count",
        "tea": "restaurant_count",
        "coffee": "restaurant_count",
        "tiffin_breakfast": "restaurant_count",
        "food": "restaurant_count",
        "biryani_rice": "restaurant_count",
        "fast_food": "restaurant_count",
        "fruits": "restaurant_count",
        "vegetables": "restaurant_count",
        "salon": "mall_count",
        "repair": "office_count",
    }

    def __init__(
        self,
        static_features_path: Optional[Path] = None,
        poi_json_path: Optional[Path] = None,
    ):
        self._static_path = static_features_path or STATIC_FEATURES_PATH
        self._poi_path = poi_json_path or LOCAL_OSM_POI_PATH
        self._static: pd.DataFrame = pd.DataFrame()
        self._cell_poi_index: Dict[str, List[CompetitorEntity]] = {}
        self._lock = threading.RLock()
        self._loaded_pois = False

        if self._static_path.exists():
            try:
                df = pd.read_parquet(self._static_path)
                self._static = df.set_index("h3_cell_id")
            except Exception as e:
                logger.warning(f"Could not load static features: {e}")

    def _ensure_local_pois_indexed(self):
        with self._lock:
            if self._loaded_pois:
                return
            if not self._poi_path.exists():
                self._loaded_pois = True
                return

            try:
                with open(self._poi_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                elements = data.get("elements", [])

                for elem in elements:
                    tags = elem.get("tags", {})
                    lat = elem.get("lat") or elem.get("center", {}).get("lat")
                    lng = elem.get("lon") or elem.get("center", {}).get("lon")
                    if lat is None or lng is None:
                        continue

                    lat_f, lng_f = float(lat), float(lng)
                    amenity = tags.get("amenity") or tags.get("shop") or tags.get("craft") or tags.get("commercial")
                    if not amenity or amenity not in AMENITY_TO_CATEGORY:
                        continue

                    category = AMENITY_TO_CATEGORY[amenity]
                    elem_id = str(elem.get("id") or f"osm_{lat_f:.4f}_{lng_f:.4f}")
                    name = (
                        tags.get("name")
                        or tags.get("name:en")
                        or tags.get("official_name")
                        or f"{category.capitalize()} Outlet"
                    )

                    h3_c = "88619aa6c7fffff"
                    if h3 is not None:
                        try:
                            h3_c = h3.latlng_to_cell(lat_f, lng_f, 8)
                        except Exception:
                            pass

                    hours_str = tags.get("opening_hours")
                    is_open, is_day, is_night, h_src = parse_operating_hours_detailed(hours_str, category)

                    comp = CompetitorEntity(
                        competitor_id=elem_id,
                        name=name,
                        category=category,
                        raw_type=amenity,
                        latitude=lat_f,
                        longitude=lng_f,
                        h3_cell=h3_c,
                        opening_hours=hours_str,
                        is_open_now=is_open,
                        is_daytime_active=is_day,
                        is_nighttime_active=is_night,
                        operating_hours_source=h_src,
                        source=self.source_name,
                        address=tags.get("addr:street") or tags.get("address"),
                    )

                    if h3_c not in self._cell_poi_index:
                        self._cell_poi_index[h3_c] = []
                    self._cell_poi_index[h3_c].append(comp)

                logger.info(f"Indexed {sum(len(v) for v in self._cell_poi_index.values())} local OSM competitors across {len(self._cell_poi_index)} H3 cells.")
            except Exception as e:
                logger.warning(f"Failed indexing local OSM POIs: {e}")
            finally:
                self._loaded_pois = True

    def get_competitors(
        self,
        h3_cell: str,
        vendor_category: Optional[str] = None,
        radius_m: float = 600.0,
    ) -> List[CompetitorEntity]:
        self._ensure_local_pois_indexed()
        comps = self._cell_poi_index.get(h3_cell, [])
        if vendor_category:
            norm_cat = normalize_vendor_category(vendor_category)
            return [c for c in comps if normalize_vendor_category(c.category) == norm_cat]
        return comps

    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        if self._static.empty or h3_cell not in self._static.index:
            return SimulatedCompetitionProvider().get_competition_score(h3_cell, vendor_category)

        poi_col = self.CATEGORY_POI_MAP.get(vendor_category.lower(), "restaurant_count")
        raw_count = float(self._static.loc[h3_cell, poi_col]) if poi_col in self._static.columns else 0.0

        if raw_count <= 0.0 and "building_count" in self._static.columns and "commercial_ratio" in self._static.columns:
            bldg_cnt = float(self._static.loc[h3_cell, "building_count"])
            comm_ratio = float(self._static.loc[h3_cell, "commercial_ratio"])
            raw_count = max(0.5, bldg_cnt * 0.4 + comm_ratio * 3.0)

        scale = 15.0 if vendor_category.lower() in ("chai_samosa", "tea", "coffee", "fruits") else 25.0
        score = min(1.0, raw_count / scale)

        return {
            "competition_score": round(score, 4),
            "competition_level": _score_to_level(score),
            "raw_competitor_count": int(round(raw_count)),
            "poi_column_used": poi_col,
            "vendor_category": vendor_category,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": f"Category-aware competition derived from real OSM {poi_col} counts (real_periodic).",
        }

    def get_cell_competition(self, h3_cell: str, vendor_category: str) -> Dict[str, Any]:
        competitors = self.get_competitors(h3_cell)
        if not competitors:
            # Fallback to static POI counts if cell has no indexed points
            score_data = self.get_competition_score(h3_cell, vendor_category)
            return {
                "h3_cell": h3_cell,
                "vendor_category": vendor_category,
                "total_competitors": score_data.get("raw_competitor_count", 0),
                "same_category_competitors": score_data.get("raw_competitor_count", 0),
                "active_competitors_now": score_data.get("raw_competitor_count", 0),
                "category_counts": {normalize_vendor_category(vendor_category): score_data.get("raw_competitor_count", 0)},
                "competition_score": score_data.get("competition_score", 0.3),
                "competition_level": score_data.get("competition_level", "low"),
                "competitors": [],
                "freshness": {
                    "source_type": self.source_type,
                    "source_name": self.source_name,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "from_cache": True,
                    "cache_age_seconds": 0.0,
                },
            }

        agg = aggregate_competitors_for_cell(competitors, h3_cell, vendor_category)
        agg["competitors"] = [c.to_dict() for c in competitors]
        agg["freshness"] = {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "from_cache": True,
            "cache_age_seconds": 0.0,
        }
        return agg


# ---------------------------------------------------------------------------
# Implementation 3: OSM Overpass API Provider (Live Queries with Fallback)
# ---------------------------------------------------------------------------

class OSMOverpassCompetitionProvider(CompetitionProvider):
    """
    Live OpenStreetMap Overpass API competitor provider.
    Executes live Overpass QL queries within radius of H3 cell,
    resolving competitor category, location, and operating hours.
    Maintains LRU cache and rate-limit circuit breaking with automatic
    fallback to OSMCompetitionProvider (local snapshot).
    """
    source_type = "real_live"
    source_name = "osm_overpass"

    OVERPASS_ENDPOINTS = [
        os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter"),
        "https://overpass.kumi.systems/api/interpreter",
    ]

    def __init__(
        self,
        cache_ttl_sec: float = 3600.0,
        timeout_sec: float = 3.5,
    ):
        self.cache_ttl_sec = cache_ttl_sec
        self.timeout_sec = timeout_sec
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._disabled_until: float = 0.0
        self._fallback = OSMCompetitionProvider()
        self._lock = threading.RLock()

    def is_rate_limited(self) -> bool:
        return time.time() < self._disabled_until

    def get_competitors(
        self,
        h3_cell: str,
        vendor_category: Optional[str] = None,
        radius_m: float = 600.0,
    ) -> List[CompetitorEntity]:
        now_ts = time.time()
        cache_key = f"{h3_cell}_{radius_m:.0f}"

        # 1. Check in-memory cache
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if now_ts - entry["ts"] < self.cache_ttl_sec:
                    comps: List[CompetitorEntity] = entry["competitors"]
                    if vendor_category:
                        norm_cat = normalize_vendor_category(vendor_category)
                        return [c for c in comps if normalize_vendor_category(c.category) == norm_cat]
                    return comps

        # 2. Check circuit breaker (Rate limit cooldown)
        if self.is_rate_limited():
            logger.debug(f"Overpass circuit breaker active ({self._disabled_until - now_ts:.0f}s left); using OSM snapshot fallback.")
            return self._fallback.get_competitors(h3_cell, vendor_category, radius_m)

        # 3. Resolve centroid coordinates
        lat, lng = 16.5062, 80.6480
        if h3 is not None:
            try:
                lat, lng = h3.cell_to_latlng(h3_cell)
            except Exception:
                pass

        # 4. Construct Overpass QL Query
        query = f"""
        [out:json][timeout:{int(self.timeout_sec)}];
        (
          node["amenity"~"restaurant|fast_food|cafe|bar|ice_cream|marketplace|food_court|pub"](around:{radius_m},{lat:.5f},{lng:.5f});
          node["shop"~"greengrocer|supermarket|bakery|hairdresser|convenience|kiosk|tyres"](around:{radius_m},{lat:.5f},{lng:.5f});
          node["craft"~"car_repair|motorcycle_repair|electronics_repair"](around:{radius_m},{lat:.5f},{lng:.5f});
        );
        out body;
        """

        elements = []
        for endpoint in self.OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={"User-Agent": "GeoDemandAI/2.0"},
                    timeout=self.timeout_sec,
                )
                if resp.status_code == 429:
                    logger.warning(f"Overpass endpoint {endpoint} rate-limited (HTTP 429). Tripping circuit breaker for 3m.")
                    self._disabled_until = now_ts + 180.0
                    break
                elif resp.status_code == 200:
                    data = resp.json()
                    elements = data.get("elements", [])
                    break
            except Exception as e:
                logger.info(f"Overpass endpoint {endpoint} request failed: {e}")

        # If live query produced no elements or failed, use snapshot fallback
        if not elements:
            if not self.is_rate_limited():
                # Minor failure: pause live attempts for 60s
                self._disabled_until = now_ts + 60.0
            return self._fallback.get_competitors(h3_cell, vendor_category, radius_m)

        # 5. Parse Elements into CompetitorEntity
        discovered: List[CompetitorEntity] = []
        for elem in elements:
            tags = elem.get("tags", {})
            e_lat = elem.get("lat")
            e_lng = elem.get("lon")
            if e_lat is None or e_lng is None:
                continue

            amenity = tags.get("amenity") or tags.get("shop") or tags.get("craft")
            if not amenity or amenity not in AMENITY_TO_CATEGORY:
                continue

            cat = AMENITY_TO_CATEGORY[amenity]
            elem_id = str(elem.get("id") or f"ovp_{e_lat:.4f}_{e_lng:.4f}")
            name = tags.get("name") or tags.get("name:en") or f"{cat.capitalize()} Venue"
            hours_str = tags.get("opening_hours")
            is_open, is_day, is_night, h_src = parse_operating_hours_detailed(hours_str, cat)

            comp_h3 = h3_cell
            if h3 is not None:
                try:
                    comp_h3 = h3.latlng_to_cell(float(e_lat), float(e_lng), 8)
                except Exception:
                    pass

            discovered.append(CompetitorEntity(
                competitor_id=elem_id,
                name=name,
                category=cat,
                raw_type=amenity,
                latitude=float(e_lat),
                longitude=float(e_lng),
                h3_cell=comp_h3,
                opening_hours=hours_str,
                is_open_now=is_open,
                is_daytime_active=is_day,
                is_nighttime_active=is_night,
                operating_hours_source=h_src,
                source=self.source_name,
                address=tags.get("addr:street"),
            ))

        with self._lock:
            self._cache[cache_key] = {"competitors": discovered, "ts": now_ts}

        if vendor_category:
            norm_cat = normalize_vendor_category(vendor_category)
            return [c for c in discovered if normalize_vendor_category(c.category) == norm_cat]
        return discovered

    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        competitors = self.get_competitors(h3_cell, vendor_category)
        same_cat = len(competitors)
        norm_cat = normalize_vendor_category(vendor_category)
        scale = 15.0 if norm_cat in ("beverages", "tea", "coffee", "fruits", "chai_samosa") else 25.0
        score = min(1.0, round(same_cat / scale, 4))
        return {
            "competition_score": score,
            "competition_level": _score_to_level(score),
            "raw_competitor_count": same_cat,
            "vendor_category": vendor_category,
            "source_type": self.source_type if not self.is_rate_limited() else "real_periodic",
            "source_name": self.source_name if not self.is_rate_limited() else "openstreetmap",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": f"Live OSM Overpass competition: {same_cat} competitors in cell.",
        }

    def get_cell_competition(self, h3_cell: str, vendor_category: str) -> Dict[str, Any]:
        competitors = self.get_competitors(h3_cell)
        agg = aggregate_competitors_for_cell(competitors, h3_cell, vendor_category)
        agg["competitors"] = [c.to_dict() for c in competitors]
        agg["freshness"] = {
            "source_type": self.source_type if not self.is_rate_limited() else "real_periodic",
            "source_name": self.source_name if not self.is_rate_limited() else "openstreetmap",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "from_cache": not self.is_rate_limited(),
            "rate_limit_active": self.is_rate_limited(),
        }
        return agg


# ---------------------------------------------------------------------------
# Implementation 4: Google Places Competition Provider
# ---------------------------------------------------------------------------

class GooglePlacesCompetitionProvider(CompetitionProvider):
    """
    Live category-aware competition via Google Places API (New & Legacy).
    Requires GOOGLE_PLACES_API_KEY environment variable.
    Handles rate limits (HTTP 429), quotas, and missing operating hours gracefully.
    """
    source_type = "real_live"
    source_name = "google_places_api"

    CATEGORY_INCLUDED_TYPES = {
        "tea": ["cafe", "coffee_shop", "bakery"],
        "coffee": ["cafe", "coffee_shop"],
        "beverages": ["cafe", "coffee_shop", "bar"],
        "food": ["restaurant", "fast_food_restaurant", "meal_takeaway"],
        "fruits": ["grocery_store", "supermarket"],
        "vegetables": ["grocery_store", "supermarket"],
        "salon": ["beauty_salon", "hair_care", "spa"],
        "repair": ["car_repair", "auto_parts_store"],
    }

    CATEGORY_KEYWORDS = {
        "tea": "tea stall cafe chai",
        "coffee": "coffee shop cafe",
        "beverages": "juice stall cafe tea coffee",
        "food": "restaurant fast food meals dhaba",
        "fruits": "fruit shop juice stall",
        "vegetables": "vegetable market greengrocer",
        "salon": "beauty salon hair salon barber",
        "repair": "automobile repair mechanic tyre shop garage",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_PLACES_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GOOGLE_PLACES_API_KEY not set.")
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 1800  # 30 minutes
        self._disabled_until: float = 0.0
        self._fallback = OSMCompetitionProvider()
        self._lock = threading.RLock()

    def is_rate_limited(self) -> bool:
        return time.time() < self._disabled_until

    def get_competitors(
        self,
        h3_cell: str,
        vendor_category: Optional[str] = None,
        radius_m: float = 600.0,
    ) -> List[CompetitorEntity]:
        now_ts = time.time()
        cache_key = f"{h3_cell}_{vendor_category or 'all'}"

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if now_ts - entry["ts"] < self._cache_ttl:
                    return entry["competitors"]

        if self.is_rate_limited():
            logger.debug(f"Google Places circuit breaker active; falling back to OSM snapshot.")
            return self._fallback.get_competitors(h3_cell, vendor_category, radius_m)

        lat, lng = 16.5062, 80.6480
        if h3 is not None:
            try:
                lat, lng = h3.cell_to_latlng(h3_cell)
            except Exception:
                pass

        cat_clean = normalize_vendor_category(vendor_category or "food")
        discovered: List[CompetitorEntity] = []

        # 1. Attempt Google Places API (New) searchNearby
        try:
            url_new = "https://places.googleapis.com/v1/places:searchNearby"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": (
                    "places.id,places.displayName,places.primaryType,places.location,"
                    "places.currentOpeningHours,places.regularOpeningHours,places.rating,places.formattedAddress"
                ),
            }
            body = {
                "includedTypes": self.CATEGORY_INCLUDED_TYPES.get(cat_clean, ["restaurant"]),
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": radius_m,
                    }
                },
            }
            resp = requests.post(url_new, json=body, headers=headers, timeout=3.0)
            if resp.status_code == 429:
                logger.warning("Google Places API (New) rate-limited (HTTP 429). Tripping circuit breaker for 3m.")
                self._disabled_until = now_ts + 180.0
                return self._fallback.get_competitors(h3_cell, vendor_category, radius_m)

            if resp.status_code == 200:
                data = resp.json()
                places = data.get("places", [])
                for p in places:
                    p_loc = p.get("location", {})
                    p_lat = p_loc.get("latitude", lat)
                    p_lng = p_loc.get("longitude", lng)
                    p_name = p.get("displayName", {}).get("text", "Competitor Business")
                    p_type = p.get("primaryType", "restaurant")
                    mapped_cat = GOOGLE_TYPE_TO_CATEGORY.get(p_type, cat_clean)

                    # Operating hours handling
                    cur_hours = p.get("currentOpeningHours", {})
                    reg_hours = p.get("regularOpeningHours", {})
                    open_now = cur_hours.get("openNow") if cur_hours.get("openNow") is not None else reg_hours.get("openNow")
                    weekday_desc = "; ".join(reg_hours.get("weekdayDescriptions", [])) or None

                    h_source = "provider_explicit" if open_now is not None or weekday_desc else "category_default"
                    is_day = True
                    is_night = cat_clean in ("food", "beverages")

                    comp_h3 = h3_cell
                    if h3 is not None:
                        try:
                            comp_h3 = h3.latlng_to_cell(p_lat, p_lng, 8)
                        except Exception:
                            pass

                    discovered.append(CompetitorEntity(
                        competitor_id=p.get("id") or f"gp_{p_lat:.4f}_{p_lng:.4f}",
                        name=p_name,
                        category=mapped_cat,
                        raw_type=p_type,
                        latitude=p_lat,
                        longitude=p_lng,
                        h3_cell=comp_h3,
                        opening_hours=weekday_desc,
                        is_open_now=open_now,
                        is_daytime_active=is_day,
                        is_nighttime_active=is_night,
                        operating_hours_source=h_source,
                        source=self.source_name,
                        rating=p.get("rating"),
                        address=p.get("formattedAddress"),
                    ))

                with self._lock:
                    self._cache[cache_key] = {"competitors": discovered, "ts": now_ts}
                return discovered
        except Exception as e:
            logger.info(f"Google Places API (New) attempt: {e}. Trying legacy endpoint fallback.")

        # 2. Legacy Nearby Search Fallback
        try:
            keyword = self.CATEGORY_KEYWORDS.get(cat_clean, vendor_category or "food")
            url_legacy = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                "location": f"{lat},{lng}",
                "radius": int(radius_m),
                "keyword": keyword,
                "key": self.api_key,
            }
            resp = requests.get(url_legacy, params=params, timeout=3.0)
            if resp.status_code == 429:
                self._disabled_until = now_ts + 180.0
                return self._fallback.get_competitors(h3_cell, vendor_category, radius_m)

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            for r in results:
                geo = r.get("geometry", {}).get("location", {})
                r_lat = geo.get("lat", lat)
                r_lng = geo.get("lng", lng)
                r_name = r.get("name", "Competitor Business")
                types = r.get("types", [])
                p_type = types[0] if types else "restaurant"
                mapped_cat = GOOGLE_TYPE_TO_CATEGORY.get(p_type, cat_clean)

                opening = r.get("opening_hours", {})
                open_now = opening.get("open_now")

                comp_h3 = h3_cell
                if h3 is not None:
                    try:
                        comp_h3 = h3.latlng_to_cell(r_lat, r_lng, 8)
                    except Exception:
                        pass

                discovered.append(CompetitorEntity(
                    competitor_id=r.get("place_id") or f"gpl_{r_lat:.4f}_{r_lng:.4f}",
                    name=r_name,
                    category=mapped_cat,
                    raw_type=p_type,
                    latitude=r_lat,
                    longitude=r_lng,
                    h3_cell=comp_h3,
                    opening_hours="Open Now" if open_now else None,
                    is_open_now=open_now,
                    is_daytime_active=True,
                    is_nighttime_active=cat_clean in ("food", "beverages"),
                    operating_hours_source="provider_explicit" if open_now is not None else "category_default",
                    source=self.source_name,
                    rating=r.get("rating"),
                    address=r.get("vicinity"),
                ))

            with self._lock:
                self._cache[cache_key] = {"competitors": discovered, "ts": now_ts}
            return discovered
        except Exception as e:
            logger.warning(f"Google Places API failed ({e}). Tripping circuit breaker for 3m, falling back to OSM.")
            self._disabled_until = now_ts + 180.0
            return self._fallback.get_competitors(h3_cell, vendor_category, radius_m)

    def get_competition_score(self, h3_cell: str, vendor_category: str) -> dict:
        now_ts = time.time()
        if self.is_rate_limited():
            return self._fallback.get_competition_score(h3_cell, vendor_category)

        competitors = self.get_competitors(h3_cell, vendor_category)
        if self.is_rate_limited():
            return self._fallback.get_competition_score(h3_cell, vendor_category)

        count = len(competitors)
        scale = 15.0 if normalize_vendor_category(vendor_category) in ("beverages", "tea", "coffee", "fruits") else 25.0
        score = min(1.0, round(count / scale, 4))

        return {
            "competition_score": score,
            "competition_level": _score_to_level(score),
            "raw_competitor_count": count,
            "vendor_category": vendor_category,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": f"Live Google Places API: {count} competitors discovered in zone.",
        }

    def get_cell_competition(self, h3_cell: str, vendor_category: str) -> Dict[str, Any]:
        competitors = self.get_competitors(h3_cell)
        if self.is_rate_limited():
            return self._fallback.get_cell_competition(h3_cell, vendor_category)

        agg = aggregate_competitors_for_cell(competitors, h3_cell, vendor_category)
        agg["competitors"] = [c.to_dict() for c in competitors]
        agg["freshness"] = {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "from_cache": True,
            "rate_limit_active": self.is_rate_limited(),
        }
        return agg


# ---------------------------------------------------------------------------
# Scheduled Competitor Refresh Worker
# ---------------------------------------------------------------------------

class ScheduledCompetitorRefresh:
    """
    Background scheduler for periodic competitor intelligence refresh.
    Refreshes competitor densities and operating hours across monitored H3 deployment cells.
    """

    def __init__(
        self,
        provider: Optional[CompetitionProvider] = None,
        refresh_interval_sec: float = 14400.0,  # 4 hours
        monitored_cells: Optional[List[str]] = None,
    ):
        self.provider = provider or OSMCompetitionProvider()
        self.refresh_interval_sec = refresh_interval_sec
        self.monitored_cells = list(monitored_cells or DEFAULT_MONITORED_H3_CELLS)
        self.last_refresh_time: Optional[datetime] = None
        self.next_refresh_time: Optional[datetime] = None
        self.refreshed_cells_count: int = 0
        self.total_competitors_indexed: int = 0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def start(self):
        """Starts background refresh worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="CompetitorRefreshWorker")
            self._thread.start()
            logger.info("Scheduled Competitor Refresh worker started.")

    def stop(self):
        """Stops background refresh worker gracefully."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            logger.info("Scheduled Competitor Refresh worker stopped.")

    def _worker_loop(self):
        while self._running:
            # Sleep in small increments for responsive shutdown before refresh cycle
            slept = 0.0
            while self._running and slept < self.refresh_interval_sec:
                time.sleep(1.0)
                slept += 1.0

            if not self._running:
                break

            try:
                self.refresh_cells(self.monitored_cells, force=False)
            except Exception as e:
                logger.warning(f"Error in scheduled competitor refresh cycle: {e}")

    def refresh_cells(
        self,
        h3_cells: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Executes refresh across targeted cells."""
        cells = h3_cells or self.monitored_cells
        now = datetime.now(timezone.utc)
        count = 0
        total_comps = 0

        for cell in cells:
            try:
                comps = self.provider.get_competitors(cell)
                total_comps += len(comps)
                count += 1
            except Exception as e:
                logger.debug(f"Refresh failed for cell {cell}: {e}")

        with self._lock:
            self.last_refresh_time = now
            self.next_refresh_time = now + timedelta(seconds=self.refresh_interval_sec)
            self.refreshed_cells_count = count
            self.total_competitors_indexed = total_comps

        return {
            "status": "COMPLETED",
            "refreshed_cells_count": count,
            "total_competitors_found": total_comps,
            "provider_used": self.provider.source_name,
            "timestamp": now.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "provider_used": self.provider.source_name,
                "monitored_cells_count": len(self.monitored_cells),
                "last_refresh_time": self.last_refresh_time.isoformat() if self.last_refresh_time else None,
                "next_refresh_time": self.next_refresh_time.isoformat() if self.next_refresh_time else None,
                "refreshed_cells_count": self.refreshed_cells_count,
                "total_competitors_indexed": self.total_competitors_indexed,
                "refresh_interval_hours": round(self.refresh_interval_sec / 3600.0, 1),
            }


# Singleton competitor refresh scheduler instance
competitor_refresh_scheduler = ScheduledCompetitorRefresh()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class CompetitionProviderFactory:
    """Factory creating and caching configured competition provider."""

    _cached_provider: Optional[CompetitionProvider] = None
    _cached_pref: Optional[str] = None
    _lock = threading.Lock()

    @classmethod
    def create(cls, preferred: str = "auto") -> CompetitionProvider:
        pref = preferred.lower() if preferred != "auto" else os.getenv("COMPETITION_PROVIDER", "auto").lower()

        with cls._lock:
            if cls._cached_provider is not None and cls._cached_pref == pref:
                return cls._cached_provider

            provider: CompetitionProvider

            if pref == "synthetic":
                provider = SimulatedCompetitionProvider()
            elif pref in ("google_places", "real", "live") or (pref == "auto" and os.environ.get("GOOGLE_PLACES_API_KEY")):
                try:
                    provider = GooglePlacesCompetitionProvider()
                except Exception as e:
                    logger.info(f"Google Places unavailable ({e}) — trying OSM Overpass / snapshot.")
                    provider = OSMOverpassCompetitionProvider()
            elif pref in ("osm_overpass", "overpass"):
                provider = OSMOverpassCompetitionProvider()
            else:
                provider = OSMCompetitionProvider()

            cls._cached_provider = provider
            cls._cached_pref = pref
            return provider

    get_provider = create
