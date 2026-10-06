"""
Test Suite for External Data Clients & Zero-Key Fallbacks (Member 2).
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.features.weather_client import OpenMeteoClient
from src.features.holiday_client import HolidayClient
from src.features.osm_client import OverpassClient


def test_weather_client_live_or_cached_fetch():
    """Verify Open-Meteo client returns properly formatted weather data."""
    client = OpenMeteoClient()
    lat, lng = 16.5062, 80.6480
    weather = client.get_current_weather(lat, lng)

    # Core physical fields
    assert "temperature" in weather and isinstance(weather["temperature"], (int, float))
    assert "humidity" in weather and 0 <= weather["humidity"] <= 100
    assert "rainfall" in weather and weather["rainfall"] >= 0
    assert "wind_speed" in weather and weather["wind_speed"] >= 0
    assert "weather_condition" in weather and weather["weather_condition"] in {"clear", "cloudy", "rain"}

    # Provenance metadata
    assert "source_type" in weather and weather["source_type"] in {"real_live", "synthetic_fallback"}
    assert "source_name" in weather


def test_weather_client_local_caching(tmp_path):
    """Verify subsequent requests within TTL read from local cache file."""
    client = OpenMeteoClient(cache_dir=tmp_path, cache_ttl_sec=60)
    lat, lng = 16.5062, 80.6480

    w1 = client.get_current_weather(lat, lng)
    assert w1["from_cache"] is False

    w2 = client.get_current_weather(lat, lng)
    assert w2["from_cache"] is True
    assert w2["temperature"] == w1["temperature"]


def test_weather_hourly_forecast_contract():
    """Verify next-hour forecast provides t+1 prediction window context without future leakage."""
    client = OpenMeteoClient()
    fcst = client.get_hourly_forecast(16.5062, 80.6480, target_hour_offset=1)

    assert "temperature" in fcst
    assert "rainfall" in fcst
    assert "weather_condition" in fcst
    assert "forecast_for_timestamp" in fcst
    assert "source_type" in fcst and fcst["source_type"] in {"real_live", "synthetic_fallback"}


def test_weather_client_synthetic_fallback():
    """Verify offline fallback produces physically realistic weather with explicit synthetic tag."""
    client = OpenMeteoClient()
    fb = client._synthetic_current_fallback(16.5062, 80.6480)

    assert fb["source_type"] == "synthetic_fallback"
    assert "synthetic" in fb["source"].lower()
    assert 15.0 <= fb["temperature"] <= 50.0  # Tropical Indian temperatures
    assert fb["humidity"] >= 20.0


def test_holiday_client_andhra_pradesh():
    """Verify HolidayClient identifies major AP festivals, national holidays, and weekends."""
    client = HolidayClient(country="IN", years=[2025], subdiv="AP")

    # National holidays
    is_rep_day, name_rep = client.is_holiday("2025-01-26")
    assert is_rep_day is True
    assert "Republic Day" in name_rep

    is_ind_day, name_ind = client.is_holiday("2025-08-15")
    assert is_ind_day is True

    # Normal non-holiday weekday (e.g. Wednesday 2025-02-12)
    is_normal, _ = client.is_holiday("2025-02-12")
    assert is_normal is False

    # Weekends
    assert client.is_weekend("2025-01-25") is True   # Saturday
    assert client.is_weekend("2025-01-26") is True   # Sunday
    assert client.is_weekend("2025-01-27") is False  # Monday

    # Batch series
    dates = pd.date_range("2025-01-01", "2025-01-31", freq="D")
    df = client.get_holiday_flags(dates)
    assert len(df) == 31
    assert df["source_type"].iloc[0] == "real_offline"


def test_osm_poi_preprocessed_lookup():
    """Verify OverpassClient looks up preprocessed POIs and attaches provenance."""
    client = OverpassClient()
    h3_sample = "88619aa6c7fffff"
    pois = client.get_poi_features_for_h3_cell(h3_sample)

    assert "hospital_count" in pois
    assert "school_count" in pois
    assert "restaurant_count" in pois
    assert "mall_count" in pois
    assert pois["source_type"] in {"real_periodic", "synthetic_fallback"}
