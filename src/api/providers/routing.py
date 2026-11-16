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
import math
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

OSRM_PUBLIC_URL = os.environ.get("OSRM_URL", "https://router.project-osrm.org")


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def validate_coordinates(lat1: float, lng1: float, lat2: float, lng2: float) -> bool:
    """Validate latitude and longitude ranges and numeric validity."""
    for lat, lng in [(lat1, lng1), (lat2, lng2)]:
        if lat is None or lng is None:
            return False
        try:
            flat, flng = float(lat), float(lng)
            if math.isnan(flat) or math.isnan(flng) or math.isinf(flat) or math.isinf(flng):
                return False
            if not (-90.0 <= flat <= 90.0 and -180.0 <= flng <= 180.0):
                return False
        except (TypeError, ValueError):
            return False
    return True


def validate_route_response(data: dict) -> bool:
    """Validate that OSRM response JSON is well-formed with valid route metrics."""
    if not isinstance(data, dict):
        return False
    if data.get("code") != "Ok":
        return False
    routes = data.get("routes")
    if not isinstance(routes, list) or len(routes) == 0:
        return False
    route = routes[0]
    if not isinstance(route, dict):
        return False
    duration = route.get("duration")
    distance = route.get("distance")
    if duration is None or distance is None:
        return False
    try:
        f_dur = float(duration)
        f_dist = float(distance)
        if math.isnan(f_dur) or math.isnan(f_dist) or f_dur < 0 or f_dist < 0:
            return False
    except (TypeError, ValueError):
        return False
    return True


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

    def get_traffic_adjusted_travel_time(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        traffic_provider: Optional[Any] = None,
    ) -> dict:
        """
        Calculates route travel time adjusted for real-time or time-of-day traffic congestion.
        Integrates with TrafficProvider.
        """
        route = self.get_travel_time(origin_lat, origin_lng, dest_lat, dest_lng)
        base_duration_min = float(route.get("duration_min", 0.0))
        dist_km = float(route.get("distance_km", 0.0))

        if traffic_provider is not None:
            try:
                traffic = traffic_provider.get_traffic(origin_lat, origin_lng, dest_lat, dest_lng)
            except Exception as e:
                logger.debug(f"Traffic lookup failed during route adjustment: {e}")
                traffic = {}
        else:
            traffic = {}

        eff_speed = max(5.0, float(traffic.get("traffic_speed_kmh", 20.0)))
        traffic_delay = float(traffic.get("traffic_delay_min", 0.0))
        congestion = float(traffic.get("congestion_ratio", 0.0))

        if dist_km <= 0.001:
            adjusted_min = 0.0
        else:
            speed_based_min = (dist_km / eff_speed) * 60.0
            adjusted_min = round(max(base_duration_min, speed_based_min + traffic_delay), 1)

        result = dict(route)
        result["duration_min"] = adjusted_min
        result["duration_sec"] = int(round(adjusted_min * 60))
        result["base_duration_min"] = base_duration_min
        result["traffic_delay_min"] = traffic_delay
        result["congestion_ratio"] = congestion
        result["traffic_level"] = traffic.get("traffic_level", "low")
        result["traffic_source"] = traffic.get("source_type", "none")
        return result

    def get_route_fuel_cost(
        self,
        distance_km: float,
        fuel_provider: Optional[Any] = None,
        congestion_ratio: float = 0.0,
    ) -> dict:
        """
        Calculates fuel cost for route distance adjusted for traffic congestion.
        Integrates with FuelPriceProvider.
        """
        dist = max(0.0, float(distance_km))
        if fuel_provider is not None:
            try:
                base_fuel = fuel_provider.get_fuel_cost_for_distance(dist)
                fuel_cost = float(base_fuel.get("fuel_cost_inr", 0.0))
                price_per_l = float(base_fuel.get("fuel_price_per_litre_inr", 95.0))
                efficiency = float(base_fuel.get("fuel_efficiency_km_per_l", 18.0))
            except Exception:
                price_per_l = 95.0
                efficiency = 18.0
                fuel_cost = round((dist / efficiency) * price_per_l, 2)
        else:
            price_per_l = 95.0
            efficiency = 18.0
            fuel_cost = round((dist / efficiency) * price_per_l, 2)

        congestion_penalty = 1.0 + (max(0.0, float(congestion_ratio)) * 0.35)
        adjusted_fuel_cost = round(fuel_cost * congestion_penalty, 2)

        return {
            "distance_km": round(dist, 3),
            "base_fuel_cost_inr": round(fuel_cost, 2),
            "adjusted_fuel_cost_inr": adjusted_fuel_cost,
            "fuel_cost_inr": adjusted_fuel_cost,
            "congestion_penalty_multiplier": round(congestion_penalty, 3),
            "fuel_price_per_litre_inr": price_per_l,
            "fuel_efficiency_km_per_l": efficiency,
        }

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
    Uses the public OSRM API or self-hosted instances with circuit-breaking and caching.
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
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._failure_threshold = 2
        self._cooldown_sec = 60.0
        self._hits = 0
        self._misses = 0

    def is_circuit_open(self) -> bool:
        """Check whether the circuit breaker is currently open."""
        return time.time() < self._circuit_open_until

    def reset_circuit_breaker(self) -> None:
        """Reset consecutive failures and close the circuit breaker."""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def clear_cache(self) -> None:
        """Flush the in-memory route cache and reset stats."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_cache_stats(self) -> dict:
        """Return cache monitoring statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "ttl_sec": self._cache_ttl,
            "circuit_open": self.is_circuit_open(),
        }

    def get_travel_time(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        timeout_s: float = 4.0,
    ) -> dict:
        if not validate_coordinates(origin_lat, origin_lng, dest_lat, dest_lng):
            raise ValueError(
                f"Invalid coordinates: origin=({origin_lat}, {origin_lng}), "
                f"dest=({dest_lat}, {dest_lng})"
            )

        now = time.time()

        # Handle zero-distance identical point
        if abs(origin_lat - dest_lat) < 1e-6 and abs(origin_lng - dest_lng) < 1e-6:
            return {
                "duration_sec": 0,
                "duration_min": 0.0,
                "distance_m": 0,
                "distance_km": 0.0,
                "is_routable": True,
                "source_type": self.source_type,
                "source_name": "osrm_identical_point",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "_ts": now,
            }

        cache_key = f"{origin_lat:.4f},{origin_lng:.4f}->{dest_lat:.4f},{dest_lng:.4f}"
        if cache_key in self._cache and now - self._cache[cache_key]["_ts"] < self._cache_ttl:
            self._hits += 1
            return self._cache[cache_key]

        self._misses += 1

        # Circuit breaker fast-path: skip network calls when endpoints are down
        if self.is_circuit_open():
            logger.debug("OSRM circuit breaker open; returning immediate synthetic fallback.")
            return self._haversine_fallback(origin_lat, origin_lng, dest_lat, dest_lng, now)

        for base in self.OSRM_MIRRORS:
            try:
                url = f"{base.rstrip('/')}/route/v1/driving/{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
                resp = requests.get(url, params={"overview": "false"}, timeout=timeout_s)
                resp.raise_for_status()
                data = resp.json()

                if not validate_route_response(data):
                    logger.debug(f"Invalid OSRM response from {base}: {data.get('code')}")
                    continue

                route = data["routes"][0]
                duration_sec = float(route["duration"])
                distance_m = float(route["distance"])

                result = {
                    "duration_sec": round(duration_sec),
                    "duration_min": round(duration_sec / 60, 1),
                    "distance_m": round(distance_m),
                    "distance_km": round(distance_m / 1000, 3),
                    "is_routable": True,
                    "source_type": self.source_type,
                    "source_name": self.source_name,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "_ts": now,
                }
                self._cache[cache_key] = result
                self._consecutive_failures = 0
                return result

            except Exception as e:
                logger.debug(f"OSRM mirror {base} failed: {e}")

        # Trip circuit breaker upon consecutive failures
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._circuit_open_until = now + self._cooldown_sec
            logger.warning(
                f"All OSRM mirrors failed ({self._consecutive_failures}x); "
                f"tripping circuit breaker for {self._cooldown_sec}s."
            )

        return self._haversine_fallback(origin_lat, origin_lng, dest_lat, dest_lng, now)

    def get_route_geometry(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        timeout_s: float = 5.0,
    ) -> dict:
        """Fetches complete street route geometry points [[lat, lng], ...] from OSRM street network."""
        if not validate_coordinates(origin_lat, origin_lng, dest_lat, dest_lng):
            raise ValueError("Invalid coordinates for route geometry")

        # Zero distance point
        if abs(origin_lat - dest_lat) < 1e-6 and abs(origin_lng - dest_lng) < 1e-6:
            return {
                "code": "Ok",
                "coordinates": [[origin_lat, origin_lng]],
                "distance_km": 0.0,
                "duration_min": 0,
                "is_routable": True,
                "source": "identical_point",
            }

        # Circuit breaker check
        if not self.is_circuit_open():
            for base in self.OSRM_MIRRORS:
                try:
                    url = f"{base.rstrip('/')}/route/v1/driving/{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
                    resp = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=timeout_s)
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
                            "is_routable": True,
                            "source": "osrm_street_network",
                        }
                except Exception as e:
                    logger.debug(f"OSRM geometry mirror {base} failed: {e}")

        # If OSRM mirrors are unreachable, generate realistic Manhattan grid street waypoints
        coords = self._generate_street_grid_waypoints(origin_lat, origin_lng, dest_lat, dest_lng)
        fb = self._haversine_fallback(origin_lat, origin_lng, dest_lat, dest_lng, time.time())
        return {
            "code": "Fallback",
            "coordinates": coords,
            "distance_km": fb["distance_km"],
            "duration_min": round(fb["duration_min"]),
            "is_routable": True,
            "source": "grid_road_interpolation",
        }

    def _generate_street_grid_waypoints(
        self, lat1: float, lng1: float, lat2: float, lng2: float
    ) -> list[list[float]]:
        """Generates realistic street grid turn waypoints instead of direct diagonal line."""
        points = [[lat1, lng1]]
        mid_lat = lat1 + (lat2 - lat1) * 0.6
        mid_lng = lng1 + (lng2 - lng1) * 0.4

        steps = 5
        for i in range(1, steps):
            t = i / steps
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
        urban_speed_kmh: float = 20.0,
        road_winding_factor: float = 1.25,
    ) -> dict:
        """
        Realistic synthetic road distance fallback incorporating urban street winding factor (1.25x).
        """
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
        straight_dist_m = 2 * R * math.asin(math.sqrt(a))

        # Apply urban road winding factor to approximate realistic street driving distance
        road_dist_m = straight_dist_m * road_winding_factor
        road_dist_km = road_dist_m / 1000.0

        speed = max(5.0, urban_speed_kmh)
        est_min = (road_dist_km / speed) * 60.0

        return {
            "duration_sec": round(est_min * 60),
            "duration_min": round(est_min, 1),
            "distance_m": round(road_dist_m),
            "distance_km": round(road_dist_km, 3),
            "straight_line_distance_km": round(straight_dist_m / 1000.0, 3),
            "road_winding_factor": road_winding_factor,
            "is_routable": True,
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
        self._disabled_until: float = 0.0

    def get_travel_time(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> dict:
        now = time.time()
        if now < self._disabled_until:
            return OSRMRoutingProvider().get_travel_time(origin_lat, origin_lng, dest_lat, dest_lng)

        cache_key = f"{origin_lat:.4f},{origin_lng:.4f}->{dest_lat:.4f},{dest_lng:.4f}"
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
                timeout=3.0,
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
            logger.warning(f"Google Maps routing API failed ({e}); tripping circuit breaker for 3m, falling back to OSRM.")
            self._disabled_until = now + 180.0
            return OSRMRoutingProvider().get_travel_time(origin_lat, origin_lng, dest_lat, dest_lng)

    def get_route_geometry(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> dict:
        """Returns street route geometry coordinates [[lat, lng], ...] via OSRM street network."""
        return OSRMRoutingProvider().get_route_geometry(origin_lat, origin_lng, dest_lat, dest_lng)


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
