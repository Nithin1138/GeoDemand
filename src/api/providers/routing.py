"""
Routing / travel-time provider abstraction layer.

Provider hierarchy:
  1. GoogleMapsRoutingProvider  — real-time (requires GOOGLE_MAPS_API_KEY env var)
                                  Uses Distance Matrix API — best for Indian road conditions.
  2. OSRMRoutingProvider        — free, no key required (OpenStreetMap routing engine)

Configured via ROUTING_PROVIDER in .env:
  - "auto"         -> uses Google Maps if key present, else OSRM
  - "google_maps"  -> forces Google Maps Distance Matrix API
  - "osrm"         -> forces OSRM (free, no key)
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

OSRM_PUBLIC_URL = os.environ.get("OSRM_URL", "http://router.project-osrm.org")


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

class RoutingProvider(ABC):
    @abstractmethod
    def get_travel_time(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> dict:
        """Returns travel time and distance between two coordinates."""

    def get_route_geometry(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> dict:
        """Returns detailed road coordinates [[lat, lng], ...] for map navigation rendering."""
        return {}

    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @property
    @abstractmethod
    def source_name(self) -> str: ...


# ---------------------------------------------------------------------------
# OSRM (Free, no key)
# ---------------------------------------------------------------------------

class OSRMRoutingProvider(RoutingProvider):
    """
    OSRM (OpenStreetMap Routing Machine) — free, no key required.
    Uses the public OSRM API or self-hosted instances.
    """
    source_type = "real_live"
    source_name = "osrm"

    OSRM_MIRRORS = [
        os.environ.get("OSRM_URL", "https://router.project-osrm.org"),
        "https://routing.openstreetmap.de/routed-car",
    ]

    def __init__(self):
        self._base = OSRM_PUBLIC_URL
        self._cache: dict = {}
        self._cache_ttl = 300  # 5 minutes

    def get_travel_time(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> dict:
        cache_key = f"{origin_lat:.4f},{origin_lng:.4f}->{dest_lat:.4f},{dest_lng:.4f}"
        now = time.time()
        if cache_key in self._cache and now - self._cache[cache_key]["_ts"] < self._cache_ttl:
            return self._cache[cache_key]

        for base in self.OSRM_MIRRORS:
            try:
                url = f"{base.rstrip('/')}/route/v1/driving/{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
                resp = requests.get(url, params={"overview": "false"}, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                route = data["routes"][0]
                duration_sec = route["duration"]
                distance_m = route["distance"]

                result = {
                    "duration_sec": round(duration_sec),
                    "duration_min": round(duration_sec / 60, 1),
                    "distance_m": round(distance_m),
                    "distance_km": round(distance_m / 1000, 3),
                    "source_type": self.source_type,
                    "source_name": self.source_name,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "_ts": now,
                }
                self._cache[cache_key] = result
                return result
            except Exception as e:
                logger.debug(f"OSRM mirror {base} failed: {e}")

        logger.warning("All OSRM mirrors failed; using straight-line estimate.")
        return self._haversine_fallback(origin_lat, origin_lng, dest_lat, dest_lng, now)

    def get_route_geometry(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> dict:
        """Fetches complete street route geometry points [[lat, lng], ...] from OSRM street network."""
        for base in self.OSRM_MIRRORS:
            try:
                url = f"{base.rstrip('/')}/route/v1/driving/{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
                resp = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=6)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    route = data["routes"][0]
                    coords = [[pt[1], pt[0]] for pt in route["geometry"]["coordinates"]]
                    return {
                        "code": "Ok",
                        "coordinates": coords,
                        "distance_km": round(route["distance"] / 1000, 2),
                        "duration_min": max(1, round(route["duration"] / 60)),
                        "source": "osrm_street_network",
                    }
            except Exception as e:
                logger.debug(f"OSRM geometry mirror {base} failed: {e}")

        # If OSRM mirrors are unreachable, generate realistic Manhattan grid street waypoints
        coords = self._generate_street_grid_waypoints(origin_lat, origin_lng, dest_lat, dest_lng)
        return {
            "code": "Fallback",
            "coordinates": coords,
            "distance_km": round(self._haversine_fallback(origin_lat, origin_lng, dest_lat, dest_lng, time.time())["distance_km"], 2),
            "duration_min": round(self._haversine_fallback(origin_lat, origin_lng, dest_lat, dest_lng, time.time())["duration_min"]),
            "source": "grid_road_interpolation",
        }

    def _generate_street_grid_waypoints(
        self, lat1: float, lng1: float, lat2: float, lng2: float
    ) -> list[list[float]]:
        """Generates realistic street grid turn waypoints instead of direct diagonal line."""
        # 1. Start point
        points = [[lat1, lng1]]
        
        # Intermediate corner turn (L-shape / S-curve along city grid)
        mid_lat = lat1 + (lat2 - lat1) * 0.6
        mid_lng = lng1 + (lng2 - lng1) * 0.4

        # Add 5 intermediate smooth street waypoints
        steps = 5
        for i in range(1, steps):
            t = i / steps
            # Smooth S-curve interpolation mimicking street block layout
            curr_lat = lat1 + (mid_lat - lat1) * t if t <= 0.6 else mid_lat + (lat2 - mid_lat) * ((t - 0.6) / 0.4)
            curr_lng = lng1 + (mid_lng - lng1) * (t / 0.6) if t <= 0.6 else mid_lng + (lng2 - mid_lng) * ((t - 0.6) / 0.4)
            points.append([round(curr_lat, 6), round(curr_lng, 6)])

        points.append([lat2, lng2])
        return points


    def _haversine_fallback(
        self,
        lat1: float, lng1: float,
        lat2: float, lng2: float,
        ts: float,
    ) -> dict:
        import math
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
        dist_m = 2 * R * math.asin(math.sqrt(a))
        est_min = (dist_m / 1000) / 20 * 60  # assume 20 km/h urban speed
        return {
            "duration_sec": round(est_min * 60),
            "duration_min": round(est_min, 1),
            "distance_m": round(dist_m),
            "distance_km": round(dist_m / 1000, 3),
            "source_type": "synthetic_fallback",
            "source_name": "haversine_estimate",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "_ts": ts,
        }


# ---------------------------------------------------------------------------
# Google Maps Distance Matrix (Real-time — optional, best for Indian roads)
# ---------------------------------------------------------------------------

class GoogleMapsRoutingProvider(RoutingProvider):
    """
    Google Maps Distance Matrix API — real-time traffic-aware travel times.
    Best accuracy for Indian road conditions.

    Set GOOGLE_MAPS_API_KEY env var.
    Get free key at: https://console.cloud.google.com/
    Free tier: $200/month credit (~40,000 Distance Matrix elements/month free).

    IMPORTANT: Enable "Distance Matrix API" in your Google Cloud Console project.
    """
    source_type = "real_live"
    source_name = "google_maps_distance_matrix"

    BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GOOGLE_MAPS_API_KEY not set.")
        self._cache: dict = {}
        self._cache_ttl = 120  # 2 minutes (traffic-aware, so don't cache too long)

    def get_travel_time(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> dict:
        cache_key = f"{origin_lat:.4f},{origin_lng:.4f}->{dest_lat:.4f},{dest_lng:.4f}"
        now = time.time()
        if cache_key in self._cache and now - self._cache[cache_key]["_ts"] < self._cache_ttl:
            return self._cache[cache_key]

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "origins": f"{origin_lat},{origin_lng}",
                    "destinations": f"{dest_lat},{dest_lng}",
                    "mode": "driving",
                    "departure_time": "now",  # enables traffic-aware duration
                    "key": self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            element = data["rows"][0]["elements"][0]
            if element["status"] != "OK":
                raise ValueError(f"Google Maps element status: {element['status']}")

            duration_sec = element["duration"]["value"]
            # duration_in_traffic is only returned when departure_time=now
            duration_traffic_sec = element.get("duration_in_traffic", {}).get("value", duration_sec)
            distance_m = element["distance"]["value"]

            result = {
                "duration_sec": duration_traffic_sec,
                "duration_min": round(duration_traffic_sec / 60, 1),
                "duration_no_traffic_sec": duration_sec,
                "distance_m": distance_m,
                "distance_km": round(distance_m / 1000, 3),
                "traffic_condition": "live",
                "source_type": self.source_type,
                "source_name": self.source_name,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "_ts": now,
            }
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"Google Maps routing API failed ({e}); falling back to OSRM.")
            return OSRMRoutingProvider().get_travel_time(origin_lat, origin_lng, dest_lat, dest_lng)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RoutingProviderFactory:
    """Selects routing provider according to ROUTING_PROVIDER configuration."""

    @staticmethod
    def get_provider(preferred: str = "auto") -> RoutingProvider:
        pref = preferred.lower() if preferred != "auto" else os.getenv("ROUTING_PROVIDER", "auto").lower()

        if pref in ("google_maps", "google") or (pref == "auto" and os.environ.get("GOOGLE_MAPS_API_KEY")):
            try:
                return GoogleMapsRoutingProvider()
            except Exception as e:
                logger.info(f"Google Maps routing provider unavailable ({e}) — using OSRM.")

        logger.info("Using OSRM routing provider (free, no key required).")
        return OSRMRoutingProvider()

    create = get_provider
