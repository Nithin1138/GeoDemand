"""
Event provider abstraction layer.

Provider hierarchy:
  1. PredictHQProvider         — real live events with attendance estimates (PREDICTHQ_API_KEY)
  2. TicketmasterProvider      — large-scale events with venue data (TICKETMASTER_API_KEY)
  3. EventbriteProvider        — real live public events (EVENTBRITE_TOKEN)
  4. StochasticEventProvider   — reads from events.parquet (simulated stochastic event model)

Configured via EVENT_PROVIDER in .env:
  - "auto"          -> tries PredictHQ → Ticketmaster → Eventbrite → stochastic
  - "predicthq"     -> forces PredictHQ provider
  - "ticketmaster"  -> forces Ticketmaster provider
  - "eventbrite"    -> forces Eventbrite provider
  - "simulated"     -> forces stochastic simulation provider
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


# ---------------------------------------------------------------------------
# PredictHQ Provider (Premium — best attendance estimates)
# ---------------------------------------------------------------------------

class PredictHQProvider(EventProvider):
    """
    PredictHQ Events API — real events with actual attendance ML forecasts.
    The gold standard for event-driven demand signals.

    Set PREDICTHQ_API_KEY env var.
    Get free key at: https://www.predicthq.com/
    Free tier: 1,000 events/month (Control Center plan).

    Steps to get free API key:
      1. Go to https://www.predicthq.com/
      2. Click "Get Started Free"
      3. Register with email
      4. Go to API Credentials → Create Token
      5. Set PREDICTHQ_API_KEY=<your_token>
    """
    source_type = "real_live"
    source_name = "predicthq_api"

    BASE_URL = "https://api.predicthq.com/v1/events/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("PREDICTHQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError("PREDICTHQ_API_KEY not set.")
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 900  # 15 minutes

    def get_event_importance(self, h3_cell: str, timestamp: datetime) -> dict:
        import h3 as h3lib
        lat, lng = h3lib.cell_to_latlng(h3_cell)
        cache_key = f"phq_{h3_cell}_{timestamp.strftime('%Y%m%d%H')}"
        now_ts = time.time()

        if cache_key in self._cache and now_ts - self._cache[cache_key]["ts"] < self._cache_ttl:
            return {**self._cache[cache_key]["data"], "from_cache": True}

        try:
            resp = requests.get(
                self.BASE_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={
                    "within": f"2km@{lat},{lng}",
                    "active.gte": timestamp.strftime("%Y-%m-%d"),
                    "active.lte": timestamp.strftime("%Y-%m-%d"),
                    "sort": "rank",
                    "limit": 5,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                top = results[0]
                phq_rank = top.get("phq_attendance", top.get("rank", 50))
                importance = min(1.0, phq_rank / 100.0)
                res = {
                    "event_active": True,
                    "event_importance": round(importance, 3),
                    "event_type": top.get("category", "local_event"),
                    "event_attendance_estimate": top.get("phq_attendance"),
                    "event_title": top.get("title"),
                    "source_type": self.source_type,
                    "source_name": self.source_name,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "note": f"PredictHQ live event: {top.get('title', '')}",
                }
                self._cache[cache_key] = {"data": res, "ts": now_ts}
                return res
        except Exception as e:
            logger.warning(f"PredictHQ API failed: {e}. Falling back.")

        return StochasticEventProvider().get_event_importance(h3_cell, timestamp)


# ---------------------------------------------------------------------------
# Ticketmaster Provider (Free — large venue events)
# ---------------------------------------------------------------------------

class TicketmasterProvider(EventProvider):
    """
    Ticketmaster Discovery API — concerts, sports, and large venue events.

    Set TICKETMASTER_API_KEY env var.
    Get free key at: https://developer.ticketmaster.com/
    Free tier: 5,000 API calls/day — completely free.

    Steps to get free API key:
      1. Go to https://developer.ticketmaster.com/
      2. Click "Get Your API Key"
      3. Register / sign in with a Ticketmaster account
      4. Create an app → copy the Consumer Key
      5. Set TICKETMASTER_API_KEY=<consumer_key>
    """
    source_type = "real_live"
    source_name = "ticketmaster_api"

    BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TICKETMASTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError("TICKETMASTER_API_KEY not set.")
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 900

    def get_event_importance(self, h3_cell: str, timestamp: datetime) -> dict:
        import h3 as h3lib
        lat, lng = h3lib.cell_to_latlng(h3_cell)
        cache_key = f"tm_{h3_cell}_{timestamp.strftime('%Y%m%d%H')}"
        now_ts = time.time()

        if cache_key in self._cache and now_ts - self._cache[cache_key]["ts"] < self._cache_ttl:
            return {**self._cache[cache_key]["data"], "from_cache": True}

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "apikey": self.api_key,
                    "latlong": f"{lat},{lng}",
                    "radius": "2",
                    "unit": "km",
                    "startDateTime": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "size": 5,
                    "sort": "relevance,desc",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            events = data.get("_embedded", {}).get("events", [])
            if events:
                ev = events[0]
                venue = ev.get("_embedded", {}).get("venues", [{}])[0]
                capacity = venue.get("generalInfo", {}).get("generalRule")
                res = {
                    "event_active": True,
                    "event_importance": 0.80,
                    "event_type": ev.get("classifications", [{}])[0].get("segment", {}).get("name", "entertainment"),
                    "event_attendance_estimate": None,  # Ticketmaster doesn't expose capacity directly
                    "event_title": ev.get("name"),
                    "event_venue": venue.get("name"),
                    "source_type": self.source_type,
                    "source_name": self.source_name,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "note": f"Ticketmaster event: {ev.get('name', '')}",
                }
                self._cache[cache_key] = {"data": res, "ts": now_ts}
                return res
        except Exception as e:
            logger.warning(f"Ticketmaster API failed: {e}. Falling back.")

        return StochasticEventProvider().get_event_importance(h3_cell, timestamp)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class EventProviderFactory:
    """Selects event provider according to EVENT_PROVIDER configuration.
    Auto priority: PredictHQ → Ticketmaster → Eventbrite → Stochastic
    """

    @staticmethod
    def get_provider(preferred: str = "auto") -> EventProvider:
        pref = preferred.lower() if preferred != "auto" else os.getenv("EVENT_PROVIDER", "auto").lower()

        if pref == "predicthq" or (pref == "auto" and os.getenv("PREDICTHQ_API_KEY")):
            try:
                return PredictHQProvider()
            except Exception as e:
                logger.info(f"PredictHQ provider unavailable ({e}).")

        if pref == "ticketmaster" or (pref == "auto" and os.getenv("TICKETMASTER_API_KEY")):
            try:
                return TicketmasterProvider()
            except Exception as e:
                logger.info(f"Ticketmaster provider unavailable ({e}).")

        if pref in ("eventbrite", "real", "live") or (pref == "auto" and os.getenv("EVENTBRITE_TOKEN")):
            try:
                return EventbriteProvider()
            except Exception as e:
                logger.info(f"Eventbrite provider unavailable ({e}) — using stochastic event model.")

        return StochasticEventProvider()

    create = get_provider

