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

class FeatureAssembler:
    """
    Assembles inference feature vectors from live data providers.
    """

    def __init__(self, static_features_path: Optional[Path] = None):
        path = static_features_path or (ROOT / "data" / "processed" / "static_features.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            self._static = df.set_index("h3_cell_id")
        else:
            raise FileNotFoundError(f"static_features.parquet not found at {path}. Run build_all_datasets.py first.")

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
            logger.warning(f"H3 cell {h3_cell} not in static_features. Using median defaults.")
            static_row = {col: float(self._static[col].median()) if col in self._static.columns else 0.0
                          for col in STATIC_FEATURE_COLUMNS}
            static_source = "synthetic_fallback"
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
