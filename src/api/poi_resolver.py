"""
GeoDemand AI — POI & Landmark Resolver

Extracts, indexes, and matches real OpenStreetMap (OSM) named POIs 
(Colleges, Universities, Hospitals, Malls, Transit Hubs, Commercial Complexes, Parks)
to candidate H3 locations for prominent UI map/card display.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h3

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
RAW_OSM_PATH = ROOT / "data" / "raw" / "osm" / "osm_vijayawada_pois.json"

H3_RESOLUTION = 8

# Priority weights for landmark matching when multiple POIs exist
CATEGORY_PRIORITIES = {
    "college": 100,
    "university": 100,
    "hospital": 85,
    "mall": 80,
    "transit": 75,
    "park": 60,
    "school": 50,
    "landmark": 40,
}

CATEGORY_EMOJIS = {
    "college": "🎓",
    "university": "🎓",
    "hospital": "🏥",
    "mall": "🏬",
    "transit": "🚆",
    "park": "🌳",
    "school": "🏫",
    "landmark": "📍",
}


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres between two GPS points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class POIResolver:
    """Singleton POI landmark spatial index based on OSM dataset."""

    _instance: Optional["POIResolver"] = None

    def __init__(self):
        self._pois: List[dict] = []
        self._h3_index: Dict[str, List[dict]] = {}
        self._loaded = False

    @classmethod
    def get(cls) -> "POIResolver":
        if cls._instance is None:
            cls._instance = cls()
        if not cls._instance._loaded:
            cls._instance._load()
        return cls._instance

    def _load(self):
        if not RAW_OSM_PATH.exists():
            logger.warning(f"OSM POI snapshot not found at {RAW_OSM_PATH}")
            self._loaded = True
            return

        try:
            with open(RAW_OSM_PATH, encoding="utf-8") as f:
                data = json.load(f)
            elements = data.get("elements", [])

            indexed_count = 0
            for elem in elements:
                tags = elem.get("tags", {})
                name = (
                    tags.get("name")
                    or tags.get("name:en")
                    or tags.get("official_name")
                )
                if not name or len(name.strip()) < 3:
                    continue

                lat = elem.get("lat") or elem.get("center", {}).get("lat")
                lng = elem.get("lon") or elem.get("center", {}).get("lon")
                if lat is None or lng is None:
                    continue

                lat, lng = float(lat), float(lng)
                amenity = str(tags.get("amenity", "")).lower()
                building = str(tags.get("building", "")).lower()
                shop = str(tags.get("shop", "")).lower()
                railway = str(tags.get("railway", "")).lower()
                leisure = str(tags.get("leisure", "")).lower()
                office = str(tags.get("office", "")).lower()
                name_lower = name.lower()

                # Categorization
                category = "landmark"
                if (
                    "college" in amenity
                    or "university" in amenity
                    or "college" in name_lower
                    or "university" in name_lower
                    or "campus" in name_lower
                    or "institute" in name_lower
                    or "academy" in name_lower
                ):
                    category = "college"
                elif "hospital" in amenity or "healthcare" in tags or "clinic" in amenity or "hospital" in name_lower:
                    category = "hospital"
                elif shop in ["mall", "supermarket", "department_store"] or "mall" in name_lower or "complex" in name_lower:
                    category = "mall"
                elif railway == "station" or amenity in ["bus_station", "bus_stop"] or "station" in name_lower or "junction" in name_lower:
                    category = "transit"
                elif leisure == "park" or "park" in name_lower or "garden" in name_lower:
                    category = "park"
                elif "school" in amenity or "school" in name_lower:
                    category = "school"
                elif office or building:
                    category = "landmark"

                try:
                    cell = h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
                except Exception:
                    continue

                poi_record = {
                    "name": name.strip(),
                    "lat": lat,
                    "lng": lng,
                    "category": category,
                    "amenity": amenity,
                    "h3_cell": cell,
                    "priority": CATEGORY_PRIORITIES.get(category, 30),
                }
                self._pois.append(poi_record)

                if cell not in self._h3_index:
                    self._h3_index[cell] = []
                self._h3_index[cell].append(poi_record)
                indexed_count += 1

            self._loaded = True
            logger.info(f"POIResolver loaded {indexed_count} named landmarks across {len(self._h3_index)} H3 cells.")

        except Exception as e:
            logger.error(f"Error loading POI dataset: {e}")
            self._loaded = True

    def resolve_landmark(
        self,
        lat: float,
        lng: float,
        h3_cell: Optional[str] = None,
        prefer_college: bool = False,
        radius_m: float = 800.0,
    ) -> dict:
        """
        Find the most prominent landmark (e.g. College name, Hospital, Mall, Transit station)
        near the candidate coordinate.
        """
        if not self._loaded:
            self._load()

        if not h3_cell and h3.is_valid_cell(h3_cell or ""):
            cell = h3_cell
        else:
            try:
                cell = h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
            except Exception:
                cell = ""

        nearby_cells = set()
        if cell and h3.is_valid_cell(cell):
            try:
                nearby_cells = set(h3.grid_disk(cell, 1))
            except Exception:
                nearby_cells = {cell}

        candidate_pois: List[dict] = []
        for c in nearby_cells:
            candidate_pois.extend(self._h3_index.get(c, []))

        # Fallback search if no indexed cells matched
        if not candidate_pois and self._pois:
            candidate_pois = [
                p for p in self._pois if abs(p["lat"] - lat) < 0.015 and abs(p["lng"] - lng) < 0.015
            ]

        best_poi = None
        best_score = -1e9

        for poi in candidate_pois:
            dist = haversine_m(lat, lng, poi["lat"], poi["lng"])
            if dist > radius_m:
                continue

            base_prio = poi["priority"]
            if prefer_college and poi["category"] == "college":
                base_prio += 200

            # Distance decay penalty
            score = base_prio - (dist / 10.0)
            if score > best_score:
                best_score = score
                best_poi = {**poi, "distance_m": round(dist, 1)}

        if best_poi:
            cat = best_poi["category"]
            emoji = CATEGORY_EMOJIS.get(cat, "📍")
            name = best_poi["name"]
            
            # Format clean title
            display_name = f"{emoji} {name}"
            is_college = cat == "college"

            return {
                "landmark_name": name,
                "landmark_display": display_name,
                "landmark_type": cat.capitalize(),
                "is_college": is_college,
                "distance_m": best_poi["distance_m"],
                "emoji": emoji,
                "has_osm_match": True,
            }

        # Fallback localized neighborhood naming if no exact OSM landmark within radius
        area_name = self._get_area_name(lat, lng)
        fallback_cat = "College Zone" if prefer_college else "Commercial Zone"
        emoji = "🎓" if prefer_college else "📍"
        title = f"{area_name} {fallback_cat}"

        return {
            "landmark_name": title,
            "landmark_display": f"{emoji} {title}",
            "landmark_type": "College" if prefer_college else "Area",
            "is_college": prefer_college,
            "distance_m": 0,
            "emoji": emoji,
            "has_osm_match": False,
        }

    def _get_area_name(self, lat: float, lng: float) -> str:
        """Helper to get Vijayawada micro-neighborhood name based on lat/lng bbox."""
        if 16.495 <= lat <= 16.515 and 80.640 <= lng <= 80.665:
            return "Benz Circle / Labbipet"
        elif 16.505 <= lat <= 16.525 and 80.620 <= lng <= 80.645:
            return "Governorpet / MG Road"
        elif 16.515 <= lat <= 16.535 and 80.610 <= lng <= 80.630:
            return "Vijayawada Railway Station / Gandhi Nagar"
        elif 16.520 <= lat <= 16.545 and 80.660 <= lng <= 80.690:
            return "Gunadala / Ramavarappadu"
        elif 16.480 <= lat <= 16.505 and 80.650 <= lng <= 80.680:
            return "Patamata / Auto Nagar"
        elif 16.530 <= lat <= 16.560 and 80.560 <= lng <= 80.590:
            return "Gollapudi Commercial Hub"
        elif 16.420 <= lat <= 16.450 and 80.550 <= lng <= 80.580:
            return "Mangalagiri Educational Belt"
        elif 16.480 <= lat <= 16.500 and 80.590 <= lng <= 80.610:
            return "Tadepalli / Seethanagaram"
        return "Vijayawada Central"


def resolve_candidate_landmark(lat: float, lng: float, h3_cell: str, features: dict = None) -> dict:
    """Convenience functional interface for POI landmark resolution."""
    resolver = POIResolver.get()
    
    # Check feature signals
    college_count = features.get("college_count", 0) if features else 0
    prefer_college = college_count > 0

    return resolver.resolve_landmark(lat, lng, h3_cell=h3_cell, prefer_college=prefer_college)
