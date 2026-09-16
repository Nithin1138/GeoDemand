"""
Weather data client — Open-Meteo (https://open-meteo.com).

NO API KEY REQUIRED. Open-Meteo's historical archive AND current/forecast
endpoints are all free and unlimited for non-commercial/prototype use.

Three modes:
  1. get_historical_weather()  — historical archive for training data
  2. get_current_weather()     — current observation for live inference
  3. get_hourly_forecast()     — 1-hour-ahead forecast for next-hour prediction

All methods return rows tagged with source_type and source_name so callers
can never mistake synthetic fallback data for real data.

TEMPORAL CONTRACT (enforced by callers):
  For a prediction made at t=16:00, only use:
    - Current observed weather at 16:00 (get_current_weather)
    - Forecast for 17:00 issued at 16:00 (get_hourly_forecast target_hour=17)
  NEVER use actual weather observed AFTER the prediction time (future leakage).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather code → categorical label mapping (subset relevant to demand)
WMO_TO_CONDITION = {
    range(0, 2): "clear",
    range(2, 4): "cloudy",
    range(51, 68): "rain",
    range(71, 78): "cloudy",   # snow → treated as cloudy for tropical context
    range(80, 87): "rain",
    range(95, 100): "rain",
}


def _wmo_to_condition(code: int) -> str:
    for r, label in WMO_TO_CONDITION.items():
        if code in r:
            return label
    if code < 2:
        return "clear"
    if code < 51:
        return "cloudy"
    return "rain"


class OpenMeteoClient:
    def __init__(self, cache_dir: str | Path = "data/raw/weather", cache_ttl_sec: int = 300):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_sec = cache_ttl_sec  # for live data: 5 min default

    # ------------------------------------------------------------------
    # 1. HISTORICAL ARCHIVE (for training data generation)
    # ------------------------------------------------------------------

    def get_historical_weather(
        self, lat: float, lng: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Hourly weather for a date range — used by build_all_datasets.py to
        populate weather_hourly.parquet for model training.

        Returns one row per hour with full feature set + source tag.
        Falls back to deterministic synthetic series if offline.
        """
        cache_key = f"hist_weather_{lat}_{lng}_{start_date}_{end_date}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            logger.info(f"Loading cached historical weather from {cache_path}")
            return pd.read_parquet(cache_path)

        hourly_vars = [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "rain", "weather_code", "cloud_cover", "wind_speed_10m",
            "surface_pressure", "visibility", "uv_index",
        ]
        params = {
            "latitude": lat, "longitude": lng,
            "start_date": start_date, "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": "auto",
        }

        try:
            resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()["hourly"]

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["time"]),
                "temperature": [round(v, 1) if v is not None else 27.0 for v in data["temperature_2m"]],
                "humidity": [round(v, 1) if v is not None else 65.0 for v in data["relative_humidity_2m"]],
                "rainfall": [round(v, 2) if v is not None else 0.0 for v in data["precipitation"]],
                "wind_speed": [round(v, 1) if v is not None else 5.0 for v in data["wind_speed_10m"]],
                "pressure": [round(v, 1) if v is not None else 1010.0 for v in data["surface_pressure"]],
                "cloud_cover": [round(v, 1) if v is not None else 20.0 for v in data["cloud_cover"]],
                "visibility": [round(min(v / 1000.0, 12.0), 2) if v is not None else 8.0 for v in data["visibility"]],
                "weather_condition": [_wmo_to_condition(int(c)) if c is not None else "clear" for c in data["weather_code"]],
                "uv_index": [round(v, 1) if v is not None else 3.0 for v in data["uv_index"]],
                "source": "real:open-meteo-archive",
                "source_type": "real_periodic",
                "source_name": "open-meteo",
            })
            df.to_parquet(cache_path, index=False)
            logger.info(f"Fetched {len(df)} hourly records from Open-Meteo archive.")
            return df

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Open-Meteo archive request failed ({e}); falling back to synthetic weather. "
                "This is expected in offline environments."
            )
            return self._synthetic_fallback_hourly(lat, lng, start_date, end_date)

    # ------------------------------------------------------------------
    # 2. CURRENT WEATHER (for live inference at decision_timestamp t)
    # ------------------------------------------------------------------

    def get_current_weather(self, lat: float, lng: float) -> dict:
        """
        Fetches current weather observation for use at decision_timestamp t.

        Returns a dict with all feature-store-compatible weather fields plus
        full provenance metadata. Never raises — falls back to synthetic with
        explicit source_type="synthetic_fallback".

        TEMPORAL RULE: Use this for context at time t only. Do NOT use actual
        observations at t+1 for features — that would cause future data leakage.
        """
        cache_key = f"live_weather_{lat:.4f}_{lng:.4f}.json"
        cache_path = self.cache_dir / cache_key

        # Short-TTL cache: avoid hammering the API on every request
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self.cache_ttl_sec:
                with open(cache_path) as f:
                    cached = json.load(f)
                cached["cache_age_seconds"] = round(age, 1)
                cached["from_cache"] = True
                return cached

        hourly_vars = [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "weather_code", "cloud_cover", "wind_speed_10m",
            "surface_pressure", "visibility", "uv_index",
        ]
        params = {
            "latitude": lat, "longitude": lng,
            "current": ",".join(hourly_vars),
            "timezone": "auto",
            "forecast_days": 1,
        }

        try:
            resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            current = data.get("current", {})
            obs_time = current.get("time", datetime.now(timezone.utc).isoformat())

            wmo_code = int(current.get("weather_code", 0) or 0)
            result = {
                "temperature": round(float(current.get("temperature_2m") or 27.0), 1),
                "humidity": round(float(current.get("relative_humidity_2m") or 65.0), 1),
                "rainfall": round(float(current.get("precipitation") or 0.0), 2),
                "wind_speed": round(float(current.get("wind_speed_10m") or 5.0), 1),
                "pressure": round(float(current.get("surface_pressure") or 1010.0), 1),
                "cloud_cover": round(float(current.get("cloud_cover") or 20.0), 1),
                "visibility": round(min(float(current.get("visibility") or 8000.0) / 1000.0, 12.0), 2),
                "weather_condition": _wmo_to_condition(wmo_code),
                "uv_index": round(float(current.get("uv_index") or 3.0), 1),
                "source": "real:open-meteo-current",
                "source_type": "real_live",
                "source_name": "open-meteo",
                "observation_timestamp": obs_time,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "from_cache": False,
                "cache_age_seconds": 0,
            }
            with open(cache_path, "w") as f:
                json.dump(result, f)
            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Open-Meteo current weather failed ({e}); using synthetic fallback.")
            return self._synthetic_current_fallback(lat, lng)

    # ------------------------------------------------------------------
    # 3. HOURLY FORECAST (for next-hour prediction — features at t for window t+1)
    # ------------------------------------------------------------------

    def get_hourly_forecast(self, lat: float, lng: float, target_hour_offset: int = 1) -> dict:
        """
        Fetches the hourly forecast for (now + target_hour_offset hours).

        TEMPORAL RULE: This is safe to use for next-hour prediction because
        the forecast was generated BEFORE the target window — it does not
        use any data observed after the decision timestamp.

        Args:
            target_hour_offset: hours ahead (1 = next hour, the standard case)

        Returns provenance-tagged weather dict for the target window.
        """
        cache_key = f"forecast_{lat:.4f}_{lng:.4f}_h{target_hour_offset}.json"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self.cache_ttl_sec:
                with open(cache_path) as f:
                    cached = json.load(f)
                cached["cache_age_seconds"] = round(age, 1)
                cached["from_cache"] = True
                return cached

        hourly_vars = [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "weather_code", "cloud_cover", "wind_speed_10m",
            "surface_pressure", "visibility", "uv_index",
        ]
        params = {
            "latitude": lat, "longitude": lng,
            "hourly": ",".join(hourly_vars),
            "timezone": "auto",
            "forecast_days": 2,
        }

        try:
            resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            now = datetime.now(timezone.utc)
            target_times = [pd.Timestamp(t) for t in times]
            # Find the forecast slot closest to now + offset
            # tz_localize(None) makes it tz-naive to match Open-Meteo response times
            target_dt = pd.Timestamp(now).tz_localize(None) + pd.Timedelta(hours=target_hour_offset)
            target_times_naive = [t.tz_localize(None) if t.tzinfo is not None else t for t in target_times]
            idx = min(range(len(target_times_naive)), key=lambda i: abs((target_times_naive[i] - target_dt).total_seconds()))

            def _get(key, default):
                vals = hourly.get(key, [])
                v = vals[idx] if idx < len(vals) else None
                return v if v is not None else default

            wmo_code = int(_get("weather_code", 0))
            result = {
                "temperature": round(float(_get("temperature_2m", 27.0)), 1),
                "humidity": round(float(_get("relative_humidity_2m", 65.0)), 1),
                "rainfall": round(float(_get("precipitation", 0.0)), 2),
                "wind_speed": round(float(_get("wind_speed_10m", 5.0)), 1),
                "pressure": round(float(_get("surface_pressure", 1010.0)), 1),
                "cloud_cover": round(float(_get("cloud_cover", 20.0)), 1),
                "visibility": round(min(float(_get("visibility", 8000.0)) / 1000.0, 12.0), 2),
                "weather_condition": _wmo_to_condition(wmo_code),
                "uv_index": round(float(_get("uv_index", 3.0)), 1),
                "source": "real:open-meteo-forecast",
                "source_type": "real_live",
                "source_name": "open-meteo",
                "forecast_for_timestamp": target_times[idx].isoformat() if idx < len(target_times) else target_dt.isoformat(),
                "forecast_generation_timestamp": now.isoformat(),
                "retrieved_at": now.isoformat(),
                "from_cache": False,
                "cache_age_seconds": 0,
            }
            with open(cache_path, "w") as f:
                json.dump(result, f)
            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Open-Meteo forecast failed ({e}); using synthetic fallback.")
            return self._synthetic_current_fallback(lat, lng)

    # ------------------------------------------------------------------
    # SYNTHETIC FALLBACKS (explicit source tagging — never pretend it's real)
    # ------------------------------------------------------------------

    def _synthetic_current_fallback(self, lat: float, lng: float) -> dict:
        """Deterministic synthetic current weather — clearly labeled."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        doy = now.timetuple().tm_yday
        temp = round(27 + 8 * np.sin(2 * np.pi * (doy - 60) / 365) + 3 * np.sin(2 * np.pi * hour / 24), 1)
        is_monsoon = 160 < doy < 270
        rainfall = round(float(np.random.default_rng(doy + hour).uniform(0, 5) if is_monsoon else 0), 2)
        condition = "rain" if rainfall > 1.0 else ("cloudy" if is_monsoon else "clear")
        return {
            "temperature": float(temp),
            "humidity": round(65.0 + 15.0 * float(is_monsoon), 1),
            "rainfall": rainfall,
            "wind_speed": 8.0,
            "pressure": 1010.0,
            "cloud_cover": 60.0 if is_monsoon else 20.0,
            "visibility": 6.0 if is_monsoon else 10.0,
            "weather_condition": condition,
            "uv_index": max(0.0, round(8.0 * np.sin(np.pi * hour / 12), 1)) if 6 <= hour <= 18 else 0.0,
            "source": "synthetic:open-meteo-fallback",
            "source_type": "synthetic_fallback",
            "source_name": "open-meteo-synthetic",
            "observation_timestamp": now.isoformat(),
            "retrieved_at": now.isoformat(),
            "from_cache": False,
            "cache_age_seconds": 0,
            "fallback_reason": "open-meteo API unavailable",
        }

    def _synthetic_fallback_hourly(self, lat: float, lng: float, start_date: str, end_date: str) -> pd.DataFrame:
        """Deterministic synthetic hourly series for training fallback."""
        seed = int(hashlib.md5(f"{start_date}{end_date}".encode()).hexdigest()[:8], 16)
        dates = pd.date_range(start_date, end_date, freq="h")
        rng = np.random.default_rng(seed)
        doy = dates.dayofyear.to_numpy()
        hour = dates.hour.to_numpy()
        temp = 27 + 8 * np.sin(2 * np.pi * (doy - 60) / 365) + 3 * np.sin(2 * np.pi * hour / 24)
        temp += rng.normal(0, 1.5, len(dates))
        monsoon = ((doy > 160) & (doy < 270)).astype(float)
        rain_prob = np.clip(0.08 + 0.45 * monsoon + rng.normal(0, 0.03, len(dates)), 0, 0.9)
        is_raining = rng.random(len(dates)) < rain_prob
        rainfall = np.where(is_raining, rng.uniform(1, 15, len(dates)), 0.0)
        condition = np.where(is_raining, "rain", np.where(monsoon.astype(bool), "cloudy", "clear"))
        df = pd.DataFrame({
            "timestamp": dates,
            "temperature": temp.round(1),
            "humidity": np.clip(65 + 15 * monsoon + rng.normal(0, 5, len(dates)), 20, 98).round(1),
            "rainfall": rainfall.round(2),
            "wind_speed": np.clip(rng.uniform(2, 20, len(dates)), 0, 35).round(1),
            "pressure": np.clip(1010 + rng.normal(0, 3, len(dates)), 995, 1020).round(1),
            "cloud_cover": np.clip(20 + 50 * monsoon + rng.normal(0, 10, len(dates)), 0, 100).round(1),
            "visibility": np.clip(10 - 4 * monsoon + rng.normal(0, 1, len(dates)), 1, 12).round(2),
            "weather_condition": condition,
            "uv_index": np.clip(8 * np.sin(np.pi * hour / 12) * (1 - 0.5 * monsoon), 0, 11).round(1),
            "source": "synthetic:open-meteo-fallback",
            "source_type": "synthetic_fallback",
            "source_name": "open-meteo-synthetic",
        })
        return df


class OpenWeatherClient:
    """
    Alternative client for OpenWeatherMap current weather.

    Set env var: export OPENWEATHER_API_KEY="your_key_here"
    Get a free key at https://openweathermap.org/api
    Note: historical data requires a paid plan; prefer Open-Meteo above for training.
    """

    def __init__(self, api_key: str | None = None):
        import os
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenWeatherMap API key. Set OPENWEATHER_API_KEY env var, "
                "or use OpenMeteoClient (no key required)."
            )

    def get_current_weather(self, lat: float, lng: float) -> dict:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lng, "appid": self.api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        return {
            "temperature": round(d["main"]["temp"], 1),
            "humidity": round(d["main"]["humidity"], 1),
            "rainfall": round(d.get("rain", {}).get("1h", 0.0), 2),
            "wind_speed": round(d["wind"]["speed"] * 3.6, 1),  # m/s → km/h
            "pressure": round(d["main"]["pressure"], 1),
            "cloud_cover": round(d["clouds"]["all"], 1),
            "visibility": round(d.get("visibility", 8000) / 1000.0, 2),
            "weather_condition": "rain" if d.get("rain") else ("cloudy" if d["clouds"]["all"] > 50 else "clear"),
            "uv_index": 3.0,  # OWM basic plan doesn't include UV
            "source": "real:openweathermap-current",
            "source_type": "real_live",
            "source_name": "openweathermap",
            "observation_timestamp": datetime.utcfromtimestamp(d["dt"]).isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    client = OpenMeteoClient()
    print("=== Current Weather ===")
    curr = client.get_current_weather(16.5062, 80.6480)
    for k, v in curr.items():
        print(f"  {k}: {v}")

    print("\n=== Next-Hour Forecast ===")
    fcst = client.get_hourly_forecast(16.5062, 80.6480, target_hour_offset=1)
    for k, v in fcst.items():
        print(f"  {k}: {v}")
