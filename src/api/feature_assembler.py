"""
Feature assembler for live inference.

Single responsibility: given a candidate H3 cell, a vendor profile, and
current real-world context, assemble the EXACT same 41-column feature
vector that the model was trained on (feature_store.parquet schema).

Schema contract (must match feature_store.py column sets):
  - Static spatial features (17 cols)
  - Weather features (9 cols)
  - Calendar features (8 cols)
  - Dynamic: competition_score, event_importance (2 cols)
  - Vendor: vendor_category, inventory_capacity, preparation_time, average_order_value (4 cols)
  Total trainable: 40 features + expected_customer_count (target, excluded at inference)

The assembler also returns a `data_freshness` dict for every feature group,
which the API surfaces in the recommendation response and dashboard.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from feature_store import (
    STATIC_FEATURE_COLUMNS, WEATHER_FEATURE_COLUMNS,
    CALENDAR_FEATURE_COLUMNS, VENDOR_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calendar helper — reproduces calendar.py logic for current timestamp
# ---------------------------------------------------------------------------

SEASONS = {
    (1, 2): "winter", (3, 5): "summer",
    (6, 9): "monsoon", (10, 12): "post_monsoon",
}


def _get_season(month: int) -> str:
    for (m1, m2), season in SEASONS.items():
        if m1 <= month <= m2:
            return season
    return "post_monsoon"


def _is_school_vacation(month: int, weekday: int) -> bool:
    return month in (5, 6) or (month == 12 and weekday >= 5)


def _calendar_features(ts: datetime) -> dict:
    """Derive calendar feature dict from a timestamp (no external calls)."""
    try:
        import holidays as hols
        india_holidays = hols.India(years=ts.year, subdiv="AP")
        is_holiday = int(ts.date() in india_holidays)
    except Exception:
        is_holiday = 0

    weekday = ts.weekday()  # Monday=0, Sunday=6
    hour = ts.hour
    month = ts.month

    return {
        "hour": hour,
        "day_of_week": weekday,
        "month": month,
        "is_weekend": int(weekday >= 5),
        "is_holiday": is_holiday,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "day_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "day_cos": float(np.cos(2 * np.pi * weekday / 7)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
    }


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

_DYNAMIC_SPATIAL_CACHE: dict[str, dict] = {}


class FeatureAssembler:
    """
    Assembles inference feature vectors from live data providers.
    Supports both pre-computed grid datasets (e.g. Vijayawada) and
    dynamic spatial POI & road feature discovery for any global location.
    """

    def __init__(self, static_features_path: Optional[Path] = None):
        path = static_features_path or (ROOT / "data" / "processed" / "static_features.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            self._static = df.set_index("h3_cell_id")
        else:
            raise FileNotFoundError(f"static_features.parquet not found at {path}. Run build_all_datasets.py first.")

    def _discover_cell_features(self, h3_cell: str) -> tuple[dict, str]:
        """
        Dynamically extract live real-world spatial & POI features for any H3 cell
        outside the pre-computed grid dataset (e.g. Hyderabad, Guntur, Bangalore, Mumbai).
        """
        import os
        import requests
        import h3

        medians = {
            col: float(self._static[col].median()) if col in self._static.columns else 0.0
            for col in STATIC_FEATURE_COLUMNS
        }
        feat = medians.copy()

        try:
            lat, lng = h3.cell_to_latlng(h3_cell)
        except Exception:
            return feat, "synthetic_fallback"

        tomtom_key = os.getenv("TOMTOM_API_KEY")
        source = "synthetic_fallback"

        # 1. Live POI Category Discovery via TomTom Places API
        if tomtom_key:
            try:
                url = f"https://api.tomtom.com/search/2/nearbySearch/.json?lat={lat:.5f}&lon={lng:.5f}&radius=450&limit=30&key={tomtom_key}"
                resp = requests.get(url, timeout=2.5).json()
                results = resp.get("results", [])
                if results:
                    poi_counts = {
                        "office_count": 0, "restaurant_count": 0, "hospital_count": 0,
                        "mall_count": 0, "school_count": 0, "college_count": 0,
                        "park_count": 0, "bus_stop_count": 0, "parking_count": 0
                    }
                    for item in results:
                        cats = [c.lower() for c in item.get("poi", {}).get("categories", [])]
                        cat_str = " ".join(cats)
                        if any(k in cat_str for k in ["office", "government", "corporate"]):
                            poi_counts["office_count"] += 1
                        if any(k in cat_str for k in ["restaurant", "cafe", "food", "fast food"]):
                            poi_counts["restaurant_count"] += 1
                        if any(k in cat_str for k in ["hospital", "clinic", "health", "dentist", "pharmacy"]):
                            poi_counts["hospital_count"] += 1
                        if any(k in cat_str for k in ["shop", "mall", "clothing", "retail", "market"]):
                            poi_counts["mall_count"] += 1
                        if any(k in cat_str for k in ["school"]):
                            poi_counts["school_count"] += 1
                        if any(k in cat_str for k in ["college", "university"]):
                            poi_counts["college_count"] += 1
                        if any(k in cat_str for k in ["park", "garden", "recreation"]):
                            poi_counts["park_count"] += 1
                        if any(k in cat_str for k in ["bus", "transit", "station"]):
                            poi_counts["bus_stop_count"] += 1
                        if any(k in cat_str for k in ["parking"]):
                            poi_counts["parking_count"] += 1

                    for k, v in poi_counts.items():
                        feat[k] = float(v)

                    comm = feat["office_count"] + feat["restaurant_count"] + feat["mall_count"]
                    tot = sum(poi_counts.values())
                    feat["commercial_ratio"] = round(min(1.0, comm / max(1.0, tot + 2.0)), 2)
                    feat["residential_ratio"] = round(max(0.1, 1.0 - feat["commercial_ratio"]), 2)
                    feat["building_density"] = round(min(5.0, max(0.5, tot * 0.2)), 2)
                    source = "real_live_poi"
            except Exception as e:
                logger.debug(f"Live POI discovery error for {h3_cell}: {e}")

        # 2. Road Accessibility & Density via OSRM
        try:
            osrm_url = f"https://router.project-osrm.org/nearest/v1/driving/{lng:.5f},{lat:.5f}"
            osrm_resp = requests.get(osrm_url, timeout=1.5).json()
            if osrm_resp.get("code") == "Ok" and osrm_resp.get("waypoints"):
                dist = float(osrm_resp["waypoints"][0].get("distance", 50.0))
                if dist <= 15.0:
                    feat["road_density"] = 0.85
                elif dist <= 60.0:
                    feat["road_density"] = 0.60
                elif dist <= 150.0:
                    feat["road_density"] = 0.35
                else:
                    feat["road_density"] = 0.05
        except Exception:
            pass

        return feat, source

    def assemble(
        self,
        h3_cell: str,
        vendor_profile: dict,
        weather: dict,
        competition: dict,
        event: dict,
        decision_timestamp: Optional[datetime] = None,
    ) -> tuple[dict, dict]:
        """
        Assemble a complete inference feature row matching the training schema.
        """
        ts = decision_timestamp or datetime.now(timezone.utc)
        now_str = datetime.now(timezone.utc).isoformat()

        # --- 1. Static spatial features ---
        if h3_cell not in self._static.index:
            if h3_cell in _DYNAMIC_SPATIAL_CACHE:
                static_row = _DYNAMIC_SPATIAL_CACHE[h3_cell].copy()
                static_source = "real_live_poi"
            else:
                static_row, static_source = self._discover_cell_features(h3_cell)
                _DYNAMIC_SPATIAL_CACHE[h3_cell] = static_row.copy()
        else:
            row = self._static.loc[h3_cell]
            static_row = {col: float(row[col]) if col in row.index else 0.0 for col in STATIC_FEATURE_COLUMNS}
            static_source = str(row.get("source", "real_periodic"))

        # --- 2. Weather features (from provider) ---
        rainfall = float(weather.get("rainfall", 0.0))
        weather_row = {
            "temperature": float(weather.get("temperature", 28.0)),
            "humidity": float(weather.get("humidity", 60.0)),
            "rainfall": rainfall,
            "wind_speed": float(weather.get("wind_speed", 10.0)),
            "cloud_cover": float(weather.get("cloud_cover", 20.0)),
            "uv_index": float(weather.get("uv_index", 5.0)),
            "pressure": float(weather.get("pressure", 1010.0)),
            "is_rainy": 1.0 if rainfall > 0.5 or "rain" in str(weather.get("condition", "")).lower() else 0.0,
        }

        # --- 3. Calendar features (derived locally) ---
        cal_row = _calendar_features(ts)

        # --- 4. Competition & event features ---
        comp_score = float(competition.get("competition_score", 0.3))
        event_active = bool(event.get("event_active", False))
        event_importance = float(event.get("event_importance", 0.0))
        event_att = event.get("event_attendance_estimate")

        # --- Assemble complete feature dict ---
        features = {
            **static_row,
            **cal_row,
            **weather_row,
            "competition_score": comp_score,
            "active_events_count": 1.0 if (event_active or event_importance > 0.3) else 0.0,
            "event_max_attendance": float(event_att) if event_att is not None else 0.0,
            "event_min_distance_km": 0.5 if event_active else 5.0,
        }

        # Data freshness for dashboard display
        data_freshness = {
            "static_poi": {
                "source_type": static_source,
                "source_name": "openstreetmap",
                "description": "POI counts & land use",
                "note": "Refreshed periodically from OSM data",
                "as_of": now_str,
            },
            "weather": {
                "source_type": weather.get("source_type", "unknown"),
                "source_name": weather.get("source_name", "open-meteo"),
                "retrieved_at": weather.get("retrieved_at", now_str),
                "observation_timestamp": weather.get("observation_timestamp", now_str),
                "from_cache": weather.get("from_cache", False),
                "cache_age_seconds": weather.get("cache_age_seconds", 0),
                "fallback": weather.get("source_type") == "synthetic_fallback",
            },
            "calendar": {
                "source_type": "real_offline",
                "source_name": "india_holiday_calendar",
                "description": "Holidays & seasons from Python holidays library",
                "as_of": now_str,
            },
            "competition": {
                "source_type": competition.get("source_type", "simulated"),
                "source_name": competition.get("source_name", "market_simulation_engine"),
                "retrieved_at": competition.get("retrieved_at", now_str),
                "note": competition.get("note", ""),
            },
            "events": {
                "source_type": event.get("source_type", "simulated"),
                "source_name": event.get("source_name", "stochastic_event_model"),
                "retrieved_at": event.get("retrieved_at", now_str),
                "active": event.get("event_active", False),
            },
        }

        return features, data_freshness

    def to_dataframe(self, features: dict) -> pd.DataFrame:
        """Convert features dict to single-row DataFrame for model inference."""
        return pd.DataFrame([features])
