"""
GeoDemand AI — Central Configuration

All configurable parameters for the live recommendation system.
Values are read from environment variables with safe defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Live Data Mode & Fallback Controls
# ---------------------------------------------------------------------------
LIVE_MODE = os.getenv("LIVE_MODE", "true").lower() == "true"
ALLOW_SYNTHETIC_FALLBACK = os.getenv("ALLOW_SYNTHETIC_FALLBACK", "true").lower() == "true"
SHOW_FALLBACK_WARNINGS = os.getenv("SHOW_FALLBACK_WARNINGS", "true").lower() == "true"
MAX_LIVE_DATA_AGE_SEC = int(os.getenv("MAX_LIVE_DATA_AGE_SEC", "1800"))

# ---------------------------------------------------------------------------
# Provider Selection
# ---------------------------------------------------------------------------
WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "open_meteo")
TRAFFIC_PROVIDER = os.getenv("TRAFFIC_PROVIDER", "auto")
COMPETITION_PROVIDER = os.getenv("COMPETITION_PROVIDER", "auto")
EVENT_PROVIDER = os.getenv("EVENT_PROVIDER", "auto")
FUEL_PRICE_PROVIDER = os.getenv("FUEL_PRICE_PROVIDER", "manual")

# ---------------------------------------------------------------------------
# Spatial Config
# ---------------------------------------------------------------------------
H3_RESOLUTION = 8
DEFAULT_SEARCH_RADIUS_KM = float(os.getenv("SEARCH_RADIUS_KM", "2.0"))
DEFAULT_TOP_N = int(os.getenv("TOP_N", "5"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "50"))
AVG_URBAN_SPEED_KMH = float(os.getenv("AVG_URBAN_SPEED_KMH", "20.0"))

# ---------------------------------------------------------------------------
# Cache TTLs (seconds)
# ---------------------------------------------------------------------------
WEATHER_CACHE_TTL_SEC = int(os.getenv("WEATHER_CACHE_TTL_SEC", "300"))
TRAFFIC_CACHE_TTL_SEC = int(os.getenv("TRAFFIC_CACHE_TTL_SEC", "120"))
COMPETITION_CACHE_TTL_SEC = int(os.getenv("COMPETITION_CACHE_TTL_SEC", "1800"))
EVENTS_CACHE_TTL_SEC = int(os.getenv("EVENTS_CACHE_TTL_SEC", "900"))
OSM_CACHE_TTL_SEC = int(os.getenv("OSM_CACHE_TTL_SEC", "86400"))
FUEL_PRICE_CACHE_TTL_SEC = int(os.getenv("FUEL_PRICE_CACHE_TTL_SEC", "86400"))
HOLIDAY_CACHE_TTL_SEC = int(os.getenv("HOLIDAY_CACHE_TTL_SEC", "604800"))

# ---------------------------------------------------------------------------
# Vendor Business Defaults
# ---------------------------------------------------------------------------
DEFAULT_AOV_INR = float(os.getenv("DEFAULT_AOV_INR", "100.0"))
DEFAULT_VARIABLE_COST_RATE = float(os.getenv("DEFAULT_VARIABLE_COST_RATE", "0.38"))
DEFAULT_FIXED_COST_PER_DAY_INR = float(os.getenv("DEFAULT_FIXED_COST_PER_DAY_INR", "500.0"))

# ---------------------------------------------------------------------------
# Fuel & Economics
# ---------------------------------------------------------------------------
FUEL_PRICE_INR_PER_LITRE = float(os.getenv("FUEL_PRICE_INR_PER_LITRE", "95.0"))
FUEL_PRICE_SOURCE = os.getenv("FUEL_PRICE_SOURCE", "manual_fallback")
FUEL_PRICE_EFFECTIVE_DATE = os.getenv("FUEL_PRICE_EFFECTIVE_DATE", "2026-08-18")
FUEL_EFFICIENCY_KM_PER_L = float(os.getenv("FUEL_EFFICIENCY_KM_PER_L", "18.0"))

# ---------------------------------------------------------------------------
# Strategic Decision Policy Thresholds (INR/hour) & Travel Friction Limits
# ---------------------------------------------------------------------------
STAY_THRESHOLD_INR_PER_HOUR = float(os.getenv("STAY_THRESHOLD_INR_PER_HOUR", "50.0"))
CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR = float(os.getenv("CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR", "250.0"))
MAX_RECOMMENDED_TRAVEL_MIN = float(os.getenv("MAX_RECOMMENDED_TRAVEL_MIN", "12.0"))
MAX_RECOMMENDED_TRAVEL_KM = float(os.getenv("MAX_RECOMMENDED_TRAVEL_KM", "3.0"))
FUEL_COST_PER_KM = round(FUEL_PRICE_INR_PER_LITRE / FUEL_EFFICIENCY_KM_PER_L, 3)

# ---------------------------------------------------------------------------
# Optional External API Keys
# ---------------------------------------------------------------------------
HERE_API_KEY = os.getenv("HERE_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
EVENTBRITE_TOKEN = os.getenv("EVENTBRITE_TOKEN")
API_KEY = os.getenv("API_KEY") or os.getenv("GEO_API_KEY")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Ranking Weights
# ---------------------------------------------------------------------------
RANK_WEIGHT_PROFIT = float(os.getenv("RANK_WEIGHT_PROFIT", "0.60"))
RANK_WEIGHT_DEMAND = float(os.getenv("RANK_WEIGHT_DEMAND", "0.25"))
RANK_WEIGHT_DISTANCE = float(os.getenv("RANK_WEIGHT_DISTANCE", "0.10"))
RANK_WEIGHT_COMPETITION = float(os.getenv("RANK_WEIGHT_COMPETITION", "0.05"))

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
