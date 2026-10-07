"""
Weather provider abstraction layer.

Provider hierarchy:
  1. OpenWeatherMapProvider  — real-time (requires OPENWEATHER_API_KEY env var)
                               Adds: rain alerts, UV index, feels-like as demand signals
  2. OpenMeteoProvider       — free, no key required (current + 1h forecast)

Configured via WEATHER_PROVIDER in .env:
  - "auto"            -> uses OpenWeatherMap if key present, else Open-Meteo
  - "openweathermap"  -> forces OpenWeatherMap
  - "open_meteo"      -> forces Open-Meteo (free, no key)
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

class WeatherProvider(ABC):
    @abstractmethod
    def get_weather(self, lat: float, lng: float) -> dict:
        """Returns current weather context for a coordinate."""

    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @property
    @abstractmethod
    def source_name(self) -> str: ...


# ---------------------------------------------------------------------------
# Open-Meteo (Free, no key)
# ---------------------------------------------------------------------------

class OpenMeteoProvider(WeatherProvider):
    """
    Open-Meteo free weather API — no API key required.
    Returns current conditions + next-hour precipitation forecast.
    """
    source_type = "real_live"
    source_name = "open_meteo"

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = int(os.environ.get("WEATHER_CACHE_TTL_SEC", 300))

    def get_weather(self, lat: float, lng: float) -> dict:
        cache_key = f"{lat:.3f},{lng:.3f}"
        now = time.time()
        if cache_key in self._cache and now - self._cache[cache_key]["_ts"] < self._cache_ttl:
            return self._cache[cache_key]

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "latitude": lat,
                    "longitude": lng,
                    "current": "temperature_2m,relative_humidity_2m,precipitation,weathercode,windspeed_10m",
                    "hourly": "precipitation_probability",
                    "forecast_hours": 2,
                    "timezone": "Asia/Kolkata",
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            cur = data.get("current", {})
            hourly = data.get("hourly", {})
            rain_prob = (hourly.get("precipitation_probability") or [0, 0])[1]

            result = {
                "temperature_c": cur.get("temperature_2m"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "precipitation_mm": cur.get("precipitation", 0.0),
                "wind_speed_kmh": cur.get("windspeed_10m"),
                "weather_code": cur.get("weathercode"),
                "next_hour_rain_probability_pct": rain_prob,
                "uv_index": None,
                "feels_like_c": None,
                "rain_alert": (rain_prob or 0) >= 60,
                "source_type": self.source_type,
                "source_name": self.source_name,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "_ts": now,
            }
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"Open-Meteo request failed: {e}")
            return self._null_result(now)

    def _null_result(self, ts: float) -> dict:
        return {
            "temperature_c": 28.0,
            "humidity_pct": 65.0,
            "precipitation_mm": 0.0,
            "wind_speed_kmh": 10.0,
            "weather_code": 0,
            "next_hour_rain_probability_pct": 0,
            "uv_index": None,
            "feels_like_c": None,
            "rain_alert": False,
            "source_type": "synthetic_fallback",
            "source_name": "weather_defaults",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "_ts": ts,
        }


# ---------------------------------------------------------------------------
# OpenWeatherMap (Real-time — optional, richer signals)
# ---------------------------------------------------------------------------

class OpenWeatherMapProvider(WeatherProvider):
    """
    OpenWeatherMap Current Weather API — real-time weather with UV index,
    feels-like temperature, and precipitation alerts.

    Set OPENWEATHER_API_KEY env var.
    Get free key at: https://home.openweathermap.org/api_keys
    Free tier: 60 calls/min, 1,000,000 calls/month (Current Weather API).
    UV index + hourly rain probability requires One Call API 3.0 (paid plan).
    """
    source_type = "real_live"
    source_name = "openweathermap"

    CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
    ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENWEATHER_API_KEY not set.")
        self._cache: dict = {}
        self._cache_ttl = int(os.environ.get("WEATHER_CACHE_TTL_SEC", 300))

    def get_weather(self, lat: float, lng: float) -> dict:
        cache_key = f"{lat:.3f},{lng:.3f}"
        now = time.time()
        if cache_key in self._cache and now - self._cache[cache_key]["_ts"] < self._cache_ttl:
            return self._cache[cache_key]

        try:
            resp = requests.get(
                self.CURRENT_URL,
                params={
                    "lat": lat,
                    "lon": lng,
                    "appid": self.api_key,
                    "units": "metric",
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()

            main = data.get("main", {})
            wind = data.get("wind", {})
            rain = data.get("rain", {})
            weather_desc = data.get("weather", [{}])[0]

            # Try One Call API for UV index + hourly rain probability (requires paid plan)
            uv_index = None
            rain_prob_next_hour = None
            try:
                oc_resp = requests.get(
                    self.ONECALL_URL,
                    params={
                        "lat": lat,
                        "lon": lng,
                        "appid": self.api_key,
                        "units": "metric",
                        "exclude": "minutely,daily,alerts",
                    },
                    timeout=8,
                )
                if oc_resp.status_code == 200:
                    oc = oc_resp.json()
                    uv_index = oc.get("current", {}).get("uvi")
                    hourly = oc.get("hourly", [{}])
                    rain_prob_next_hour = int((hourly[1].get("pop", 0) if len(hourly) > 1 else 0) * 100)
            except Exception:
                pass  # One Call is optional, gracefully skip

            rain_1h = rain.get("1h", 0.0)
            result = {
                "temperature_c": main.get("temp"),
                "humidity_pct": main.get("humidity"),
                "precipitation_mm": rain_1h,
                "wind_speed_kmh": round((wind.get("speed", 0)) * 3.6, 1),  # m/s -> km/h
                "weather_code": weather_desc.get("id"),
                "next_hour_rain_probability_pct": rain_prob_next_hour,
                "uv_index": uv_index,
                "feels_like_c": main.get("feels_like"),
                "weather_description": weather_desc.get("description"),
                "rain_alert": rain_1h > 1.0 or (rain_prob_next_hour or 0) >= 60,
                "source_type": self.source_type,
                "source_name": self.source_name,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "_ts": now,
            }
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"OpenWeatherMap API failed ({e}); falling back to Open-Meteo.")
            return OpenMeteoProvider().get_weather(lat, lng)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class WeatherProviderFactory:
    """Selects weather provider according to WEATHER_PROVIDER configuration."""

    @staticmethod
    def get_provider(preferred: str = "auto") -> WeatherProvider:
        pref = preferred.lower() if preferred != "auto" else os.getenv("WEATHER_PROVIDER", "auto").lower()

        if pref in ("openweathermap", "owm") or (pref == "auto" and os.environ.get("OPENWEATHER_API_KEY")):
            try:
                return OpenWeatherMapProvider()
            except Exception as e:
                logger.info(f"OpenWeatherMap provider unavailable ({e}) — using Open-Meteo.")

        logger.info("Using Open-Meteo weather provider (free, no key required).")
        return OpenMeteoProvider()

    create = get_provider
