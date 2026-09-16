"""
Event provider abstraction layer.

Provider hierarchy:
  1. EventbriteProvider       — real live public events (when EVENTBRITE_TOKEN is set)
  2. StochasticEventProvider   — reads from events.parquet (simulated stochastic event model)

Configured via EVENT_PROVIDER in .env:
  - "auto"       -> uses Eventbrite if key present, else stochastic
  - "eventbrite" -> forces Eventbrite provider
  - "simulated"  -> forces stochastic simulation provider
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


class EventProvider(ABC):
    @abstractmethod
    def get_event_importance(self, h3_cell: str, timestamp: datetime) -> dict:
        """Returns event context for a cell at a given time."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...


class EventbriteProvider(EventProvider):
    """
    Live event provider using Eventbrite API / Public Event feeds.
    Set EVENTBRITE_TOKEN env var.
    """
    source_type = "real_live"
    source_name = "eventbrite_api"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("EVENTBRITE_TOKEN")
        if not self.token:
            raise EnvironmentError("EVENTBRITE_TOKEN not set.")
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 900  # 15 minutes

    def get_event_importance(self, h3_cell: str, timestamp: datetime) -> dict:
        import h3
        lat, lng = h3.cell_to_latlng(h3_cell)
        cache_key = f"{h3_cell}_{timestamp.strftime('%Y%m%d%H')}"
        now_ts = time.time()

        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if now_ts - entry["ts"] < self._cache_ttl:
                return {**entry["data"], "from_cache": True}

        url = "https://www.eventbriteapi.com/v3/events/search/"
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {
            "location.latitude": lat,
            "location.longitude": lng,
            "location.within": "2km",
            "start_date.range_start": timestamp.isoformat(),
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                events = data.get("events", [])
                if events:
                    ev = events[0]
                    res = {
                        "event_active": True,
                        "event_importance": 0.85,
                        "event_type": ev.get("name", {}).get("text", "Local Event"),
                        "event_attendance_estimate": ev.get("capacity"),
                        "source_type": self.source_type,
                        "source_name": self.source_name,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "note": f"Live Eventbrite event: {ev.get('name', {}).get('text', '')}",
                    }
                    self._cache[cache_key] = {"data": res, "ts": now_ts}
                    return res
        except Exception as e:
            logger.warning(f"Eventbrite API request failed: {e}. Falling back to simulation.")

        return StochasticEventProvider().get_event_importance(h3_cell, timestamp)


class StochasticEventProvider(EventProvider):
    """Reads from events.parquet — sparse stochastic events."""
    source_type = "simulated"
    source_name = "stochastic_event_model"

    def __init__(self, events_path: Optional[Path] = None):
        path = events_path or (ROOT / "data" / "processed" / "events.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["timestamp"]).dt.floor("D")
            self._events = df
        else:
            self._events = pd.DataFrame(columns=["h3_cell_id", "date", "event_importance"])

    def get_event_importance(self, h3_cell: str, timestamp: datetime) -> dict:
        if self._events.empty:
            return self._null_result()
        date = pd.Timestamp(timestamp).floor("D")
        matches = self._events[
            (self._events["h3_cell_id"] == h3_cell) & (self._events["date"] == date)
        ]
        if matches.empty:
            return self._null_result()
        best = matches.nlargest(1, "event_importance").iloc[0]
        return {
            "event_active": True,
            "event_importance": round(float(best["event_importance"]), 3),
            "event_type": best.get("event_type", "local_gathering"),
            "event_attendance_estimate": int(best["expected_attendance"]) if "expected_attendance" in best else None,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": "Event data is simulated (sparse stochastic model).",
        }

    def _null_result(self) -> dict:
        return {
            "event_active": False,
            "event_importance": 0.0,
            "event_type": None,
            "event_attendance_estimate": None,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }


class EventProviderFactory:
    """Selects event provider according to EVENT_PROVIDER configuration."""

    @staticmethod
    def get_provider(preferred: str = "auto") -> EventProvider:
        pref = preferred.lower() if preferred != "auto" else os.getenv("EVENT_PROVIDER", "auto").lower()

        if pref in ("eventbrite", "real", "live") or (pref == "auto" and os.getenv("EVENTBRITE_TOKEN")):
            try:
                return EventbriteProvider()
            except Exception as e:
                logger.info(f"Eventbrite provider unavailable ({e}) — using stochastic event model.")

        return StochasticEventProvider()

    create = get_provider
