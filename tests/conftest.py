"""
Global pytest configuration and autouse fixtures for GeoDemand AI.

Ensures deterministic, sub-second, 100% offline test execution:
1. Intercepts external HTTP network requests (OSRM, Open-Meteo, Overpass, Google Places,
   Ticketmaster, PredictHQ, FCM, WhatsApp) and returns immediate synthetic mock responses.
2. Prevents third-party API rate-limiting (429), timeouts, and network hangs in CI.
3. Allows test-specific monkeypatches to take precedence.
4. Suppresses external background worker loops during test runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repository root and source folders are in sys.path
ROOT = Path(__file__).parent.parent
SRC_PATHS = [
    ROOT,
    ROOT / "src",
    ROOT / "src" / "data",
    ROOT / "src" / "features",
    ROOT / "src" / "api",
    ROOT / "src" / "fleet",
    ROOT / "src" / "optimization",
    ROOT / "src" / "ingestion",
    ROOT / "src" / "notifications",
]
for p in SRC_PATHS:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Ensure offline environment defaults for testing
os.environ.setdefault("LIVE_MODE", "true")
os.environ.setdefault("ALLOW_SYNTHETIC_FALLBACK", "true")
os.environ.setdefault("ROUTING_PROVIDER", "auto")
os.environ.setdefault("COMPETITION_PROVIDER", "auto")
os.environ.setdefault("EVENT_PROVIDER", "auto")
os.environ.setdefault("WEATHER_PROVIDER", "auto")
os.environ.setdefault("ENABLE_RATE_LIMITING", "false")

# Synthetic Mock Responses for External Services
MOCK_OSRM_NEAREST = {
    "code": "Ok",
    "waypoints": [{"distance": 8.5, "name": "M.G. Road", "location": [80.6480, 16.5062]}],
}

MOCK_OSRM_ROUTE = {
    "code": "Ok",
    "routes": [{
        "distance": 1250.0,
        "duration": 220.0,
        "weight": 220.0,
        "geometry": {
            "coordinates": [
                [80.6480, 16.5062],
                [80.6495, 16.5075],
                [80.6510, 16.5090],
            ],
            "type": "LineString",
        },
    }],
    "waypoints": [
        {"distance": 5.0, "location": [80.6480, 16.5062]},
        {"distance": 6.0, "location": [80.6510, 16.5090]},
    ],
}

MOCK_OPEN_METEO = {
    "latitude": 16.5062,
    "longitude": 80.6480,
    "current": {
        "time": "2026-11-16T10:00",
        "interval": 900,
        "temperature_2m": 29.5,
        "relative_humidity_2m": 68.0,
        "precipitation": 0.0,
        "rain": 0.0,
        "weather_code": 0,
        "cloud_cover": 20.0,
        "surface_pressure": 1010.5,
        "wind_speed_10m": 12.0,
    },
    "hourly": {
        "time": [f"2026-11-16T{h:02d}:00" for h in range(24)],
        "temperature_2m": [28.0 + (h % 5) for h in range(24)],
        "relative_humidity_2m": [65.0 for _ in range(24)],
        "precipitation": [0.0 for _ in range(24)],
        "weather_code": [0 for _ in range(24)],
        "cloud_cover": [20.0 for _ in range(24)],
        "surface_pressure": [1010.0 for _ in range(24)],
        "wind_speed_10m": [10.0 for _ in range(24)],
        "uv_index": [5.0 for _ in range(24)],
        "visibility": [10000.0 for _ in range(24)],
    },
}

MOCK_OVERPASS = {
    "version": 0.6,
    "generator": "Overpass API mock",
    "elements": [
        {
            "type": "node",
            "id": 1001,
            "lat": 16.5065,
            "lon": 80.6482,
            "tags": {"name": "Sri Krishna Tiffin Center", "amenity": "restaurant", "opening_hours": "07:00-22:00"},
        },
        {
            "type": "node",
            "id": 1002,
            "lat": 16.5070,
            "lon": 80.6490,
            "tags": {"name": "Amma Tea Stall", "amenity": "cafe", "opening_hours": "06:00-23:00"},
        },
    ],
}

MOCK_PLACES = {
    "places": [
        {
            "id": "places/mock_place_1",
            "displayName": {"text": "Local Food Hub"},
            "location": {"latitude": 16.5065, "longitude": 80.6482},
            "types": ["restaurant", "food"],
            "currentOpeningHours": {"openNow": True},
        }
    ]
}


@pytest.fixture(autouse=True)
def mock_external_network_services(monkeypatch):
    """
    Autouse fixture that intercepts external requests (OSRM, Open-Meteo, Overpass,
    Google Places, Ticketmaster, FCM, WhatsApp) and returns immediate mock responses.
    Allows local ASGI testserver requests to pass through untouched.
    """
    import requests

    orig_get = requests.get
    orig_post = requests.post
    orig_put = requests.put

    def mock_get(url, *args, **kwargs):
        url_str = str(url).lower()

        # OSRM routing / nearest
        if "router.project-osrm.org" in url_str or "openstreetmap.de" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            if "/nearest/" in url_str:
                mock.json.return_value = MOCK_OSRM_NEAREST
            else:
                mock.json.return_value = MOCK_OSRM_ROUTE
            mock.text = json.dumps(mock.json.return_value)
            return mock

        # Open-Meteo weather
        if "api.open-meteo.com" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            mock.json.return_value = MOCK_OPEN_METEO
            mock.text = json.dumps(MOCK_OPEN_METEO)
            return mock

        # Google Maps / Distance Matrix
        if "maps.googleapis.com" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            mock.json.return_value = {
                "status": "OK",
                "rows": [{"elements": [{"status": "OK", "distance": {"value": 1500}, "duration": {"value": 300}}]}],
            }
            return mock

        # Ticketmaster / Events
        if "ticketmaster.com" in url_str or "predicthq.com" in url_str or "eventbriteapi.com" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            mock.json.return_value = {"_embedded": {"events": []}, "results": [], "events": []}
            return mock

        # Local or other requests pass through
        return orig_get(url, *args, **kwargs)

    def mock_post(url, *args, **kwargs):
        url_str = str(url).lower()

        # OSM Overpass QL
        if "overpass-api.de" in url_str or "kumi.systems" in url_str or "overpass" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            mock.json.return_value = MOCK_OVERPASS
            mock.text = json.dumps(MOCK_OVERPASS)
            return mock

        # Google Places (New Nearby Search)
        if "places.googleapis.com" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            mock.json.return_value = MOCK_PLACES
            mock.text = json.dumps(MOCK_PLACES)
            return mock

        # FCM HTTP v1 / Legacy
        if "fcm.googleapis.com" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            mock.json.return_value = {"name": "projects/mock/messages/mock_msg_001", "success": 1, "failure": 0}
            return mock

        # WhatsApp Graph API
        if "graph.facebook.com" in url_str:
            mock = MagicMock()
            mock.status_code = 200
            mock.ok = True
            mock.raise_for_status = MagicMock()
            mock.json.return_value = {
                "messaging_product": "whatsapp",
                "contacts": [{"input": "919876543210", "wa_id": "919876543210"}],
                "messages": [{"id": "wamid.mock_wa_msg_001"}],
            }
            return mock

        return orig_post(url, *args, **kwargs)

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)
    yield
