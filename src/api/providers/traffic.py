"""
Traffic provider abstraction layer.

Provider hierarchy:
  1. HERETrafficProvider     — real-time (requires HERE_API_KEY env var)
  2. TomTomTrafficProvider   — real-time (requires TOMTOM_API_KEY env var)
  3. TimeOfDayTrafficProxy   — synthetic_fallback (time + road_density heuristic)

Every response is labeled with source_type so the dashboard can show
  "Traffic: LIVE" vs "Traffic: Proxy (time-of-day)"
and never pretend synthetic traffic is real.
"""

from __future__ import annotations

import logging
import math
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass-style return value
# ---------------------------------------------------------------------------

def _traffic_result(
    speed_kmh: float,
    free_flow_kmh: float,
    congestion_ratio: float,
    jam_factor: Optional[float],
    source_type: str,
    source_name: str,
    retrieved_at: str,
    note: str = "",
) -> dict:
    delay_min = 0.0
    if speed_kmh > 0 and free_flow_kmh > 0:
        delay_ratio = max(0.0, (free_flow_kmh - speed_kmh) / free_flow_kmh)
        delay_min = round(delay_ratio * 10, 2)  # approximate minutes of delay per 10 min journey
    level = "low" if congestion_ratio < 0.3 else ("medium" if congestion_ratio < 0.65 else "high")
    return {
        "traffic_speed_kmh": round(speed_kmh, 1),
        "free_flow_speed_kmh": round(free_flow_kmh, 1),
        "congestion_ratio": round(congestion_ratio, 3),
        "jam_factor": round(jam_factor, 2) if jam_factor is not None else None,
        "traffic_delay_min": delay_min,
        "traffic_level": level,
        "source_type": source_type,
        "source_name": source_name,
        "retrieved_at": retrieved_at,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

class TrafficProvider(ABC):
    @abstractmethod
    def get_traffic(self, origin_lat: float, origin_lng: float,
                    dest_lat: float, dest_lng: float) -> dict:
        """Returns traffic data for the route between two points."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Time-of-Day Proxy (always available, no API key needed)
# ---------------------------------------------------------------------------

class TimeOfDayTrafficProxy(TrafficProvider):
    """
    Heuristic traffic proxy: urban speed varies by time-of-day.
    source_type = synthetic_fallback — always clearly labeled.
    """
    source_type = "synthetic_fallback"
    source_name = "time_of_day_proxy"

    # Typical Vijayawada urban speed by hour (km/h)
    HOUR_SPEED = {
        0: 35, 1: 38, 2: 40, 3: 40, 4: 38, 5: 35,
        6: 28, 7: 20, 8: 18, 9: 22, 10: 28, 11: 30,
        12: 26, 13: 28, 14: 30, 15: 28, 16: 22, 17: 18,
        18: 16, 19: 20, 20: 24, 21: 28, 22: 32, 23: 35,
    }
    FREE_FLOW_SPEED = 40.0  # urban free-flow baseline

    def get_traffic(self, origin_lat: float, origin_lng: float,
                    dest_lat: float, dest_lng: float) -> dict:
        hour = datetime.now(timezone.utc).hour
        # Adjust for IST (+5:30)
        ist_hour = (hour + 5) % 24
        speed = self.HOUR_SPEED.get(ist_hour, 25.0)
        congestion = max(0.0, (self.FREE_FLOW_SPEED - speed) / self.FREE_FLOW_SPEED)
        return _traffic_result(
            speed_kmh=speed,
            free_flow_kmh=self.FREE_FLOW_SPEED,
            congestion_ratio=congestion,
            jam_factor=round(congestion * 10, 1),
            source_type=self.source_type,
            source_name=self.source_name,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            note="Synthetic time-of-day traffic proxy. Set HERE_API_KEY for live traffic.",
        )


# ---------------------------------------------------------------------------
# HERE Traffic (real-time — optional)
# ---------------------------------------------------------------------------

class HERETrafficProvider(TrafficProvider):
    """
    HERE Traffic API — real-time flow data.
    Set HERE_API_KEY env var. Get key at: https://developer.here.com/
    Free tier: 250,000 transactions/month.
    """
    source_type = "real_live"
    source_name = "here_traffic"

    FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("HERE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("HERE_API_KEY not set.")
        self._cache: dict = {}
        self._cache_ttl = 120  # 2 minutes

    def get_traffic(self, origin_lat: float, origin_lng: float,
                    dest_lat: float, dest_lng: float) -> dict:
        import requests
        # Use midpoint of route as query center
        mid_lat = (origin_lat + dest_lat) / 2
        mid_lng = (origin_lng + dest_lng) / 2
        cache_key = f"{mid_lat:.3f},{mid_lng:.3f}"
        now = time.time()
        if cache_key in self._cache and now - self._cache[cache_key]["_ts"] < self._cache_ttl:
            return self._cache[cache_key]

        try:
            resp = requests.get(
                self.FLOW_URL,
                params={
                    "in": f"circle:{mid_lat},{mid_lng};r=500",
                    "locationReferencing": "shape",
                    "apiKey": self.api_key,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                raise ValueError("No HERE flow results for this location")

            # Aggregate across all road segments in the response
            speeds = [r["currentFlow"]["speed"] for r in results if "currentFlow" in r]
            ff_speeds = [r["currentFlow"]["freeFlow"] for r in results if "currentFlow" in r]

            avg_speed = sum(speeds) / len(speeds) if speeds else 25.0
            avg_ff = sum(ff_speeds) / len(ff_speeds) if ff_speeds else 40.0
            congestion = max(0.0, (avg_ff - avg_speed) / avg_ff)
            jam_factors = [r["currentFlow"].get("jamFactor", 0) for r in results if "currentFlow" in r]
            jam = sum(jam_factors) / len(jam_factors) if jam_factors else None

            result = _traffic_result(
                speed_kmh=avg_speed * 3.6,  # m/s → km/h
                free_flow_kmh=avg_ff * 3.6,
                congestion_ratio=congestion,
                jam_factor=jam,
                source_type=self.source_type,
                source_name=self.source_name,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            )
            result["_ts"] = now
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"HERE traffic API failed ({e}); falling back to time-of-day proxy.")
            return TimeOfDayTrafficProxy().get_traffic(origin_lat, origin_lng, dest_lat, dest_lng)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TrafficProviderFactory:
    """Selects traffic provider according to TRAFFIC_PROVIDER configuration."""

    @staticmethod
    def get_provider(preferred: str = "auto") -> TrafficProvider:
        pref = preferred.lower() if preferred != "auto" else os.getenv("TRAFFIC_PROVIDER", "auto").lower()

        if pref == "time_of_day" or pref == "synthetic":
            return TimeOfDayTrafficProxy()

        if pref in ("here", "auto") and os.environ.get("HERE_API_KEY"):
            try:
                return HERETrafficProvider()
            except Exception as e:
                logger.info(f"HERE traffic provider initialization failed ({e}) — falling back to time-of-day proxy.")

        logger.info("No live traffic API configured — using time-of-day proxy (synthetic_fallback).")
        return TimeOfDayTrafficProxy()

    create = get_provider

