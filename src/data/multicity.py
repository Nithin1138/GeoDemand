"""
Multi-City Data Engineering Pipeline and City Abstraction (Phase 14.1).

Removes single-city assumptions and introduces a unified multi-city architecture:
- Configurable city boundaries and arbitrary bounding boxes
- Support for Hyderabad (Cyberabad/HITEC City), Bengaluru (Koramangala/Indiranagar/Electronic City),
  Visakhapatnam (Beach Road/MVP Colony), and Vijayawada
- City-specific H3 spatial grid generation (Uber H3 Resolution 8)
- City-specific weather, calendar, and regional holiday configurations
- Partitioned feature store generation by city
- Standardized city onboarding protocol
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    import h3
except ImportError:
    h3 = None

logger = logging.getLogger(__name__)


@dataclass
class CityConfig:
    """Canonical configuration for a supported metropolitan area."""
    city_id: str
    city_name: str
    state_name: str
    state_code: str  # ISO 3166-2:IN subdivision code, e.g. "AP", "TS", "KA"
    country: str
    center_lat: float
    center_lng: float
    radius_km: float
    bbox: Dict[str, float]  # min_lat, max_lat, min_lng, max_lng
    key_neighborhoods: List[str]
    h3_resolution: int = 8
    timezone: str = "Asia/Kolkata"


# ==============================================================================
# CANONICAL CITY REGISTRY (Tasks 9–14, 16)
# ==============================================================================

SUPPORTED_CITIES: Dict[str, CityConfig] = {
    "vijayawada": CityConfig(
        city_id="vijayawada",
        city_name="Vijayawada",
        state_name="Andhra Pradesh",
        state_code="AP",
        country="India",
        center_lat=16.5062,
        center_lng=80.6480,
        radius_km=8.0,
        bbox={"min_lat": 16.43, "max_lat": 16.58, "min_lng": 80.52, "max_lng": 80.72},
        key_neighborhoods=["One Town", "Governorpet", "Besant Road", "Benz Circle", "Patamata", "Auto Nagar", "Gunadala"],
    ),
    "hyderabad": CityConfig(
        city_id="hyderabad",
        city_name="Hyderabad",
        state_name="Telangana",
        state_code="TS",
        country="India",
        center_lat=17.4435,  # Centered on HITEC City / Cyberabad tech hub
        center_lng=78.3772,
        radius_km=10.0,
        bbox={"min_lat": 17.30, "max_lat": 17.55, "min_lng": 78.25, "max_lng": 78.60},
        key_neighborhoods=["Cyberabad", "HITEC City", "Madhapur", "Gachibowli", "Jubilee Hills", "Banjara Hills", "Kondapur"],
    ),
    "bengaluru": CityConfig(
        city_id="bengaluru",
        city_name="Bengaluru",
        state_name="Karnataka",
        state_code="KA",
        country="India",
        center_lat=12.9352,  # Centered on Koramangala / South-East commercial belt
        center_lng=77.6245,
        radius_km=12.0,
        bbox={"min_lat": 12.80, "max_lat": 13.15, "min_lng": 77.45, "max_lng": 77.80},
        key_neighborhoods=["Koramangala", "Indiranagar", "Electronic City", "Whitefield", "HSR Layout", "BTM Layout", "MG Road"],
    ),
    "visakhapatnam": CityConfig(
        city_id="visakhapatnam",
        city_name="Visakhapatnam",
        state_name="Andhra Pradesh",
        state_code="AP",
        country="India",
        center_lat=17.7231,  # Centered on Siripuram / Beach Road commercial corridor
        center_lng=83.3245,
        radius_km=9.0,
        bbox={"min_lat": 17.60, "max_lat": 17.85, "min_lng": 83.15, "max_lng": 83.45},
        key_neighborhoods=["Beach Road", "MVP Colony", "Siripuram", "Jagadamba Junction", "Rushikonda", "Gajuwaka"],
    ),
}


class MultiCityRegistry:
    """Manages city definitions, boundary verification, and lookup."""

    def __init__(self, cities: Optional[Dict[str, CityConfig]] = None) -> None:
        self._cities: Dict[str, CityConfig] = cities.copy() if cities else SUPPORTED_CITIES.copy()

    def get_city(self, city_id: str) -> CityConfig:
        """Retrieve configuration for a city ID."""
        key = city_id.lower().strip()
        if key not in self._cities:
            raise KeyError(f"Unsupported city '{city_id}'. Available cities: {list(self._cities.keys())}")
        return self._cities[key]

    def list_cities(self) -> List[str]:
        """List all registered city IDs."""
        return list(self._cities.keys())

    def register_city(self, config: CityConfig) -> None:
        """Register a new city configuration into the platform."""
        self._validate_config(config)
        self._cities[config.city_id.lower().strip()] = config

    def is_within_city_bbox(self, city_id: str, lat: float, lng: float) -> bool:
        """Check if coordinates fall within the specified city bounding box (Task 16)."""
        city = self.get_city(city_id)
        b = city.bbox
        return b["min_lat"] <= lat <= b["max_lat"] and b["min_lng"] <= lng <= b["max_lng"]

    def _validate_config(self, config: CityConfig) -> None:
        """Validate geographic coordinates and bounds of a city configuration."""
        if not (-90.0 <= config.center_lat <= 90.0 and -180.0 <= config.center_lng <= 180.0):
            raise ValueError(f"Invalid center coordinates for city {config.city_id}: ({config.center_lat}, {config.center_lng})")
        b = config.bbox
        if b["min_lat"] >= b["max_lat"] or b["min_lng"] >= b["max_lng"]:
            raise ValueError(f"Inverted bounding box for city {config.city_id}: {b}")
        if not (b["min_lat"] <= config.center_lat <= b["max_lat"] and b["min_lng"] <= config.center_lng <= b["max_lng"]):
            raise ValueError(f"City center is outside bounding box for {config.city_id}")


# ==============================================================================
# MULTI-CITY DATA PIPELINE GENERATOR (Tasks 1–8, 15)
# ==============================================================================

class MultiCityPipeline:
    """Executes spatial grid and feature store partition builds across multiple cities."""

    def __init__(self, registry: Optional[MultiCityRegistry] = None) -> None:
        self.registry = registry or MultiCityRegistry()

    def generate_h3_grid(self, city_id: str) -> pd.DataFrame:
        """
        Generate Uber H3 Resolution 8 spatial grid for any supported city (Tasks 3, 4).
        """
        city = self.registry.get_city(city_id)
        res = city.h3_resolution

        if h3 is None:
            raise RuntimeError("h3 library required for spatial grid generation.")

        # Center cell
        center_cell = h3.latlng_to_cell(city.center_lat, city.center_lng, res)
        edge_len_km = h3.average_hexagon_edge_length(res, unit="km")
        k_rings = max(1, int(np.ceil(city.radius_km / edge_len_km)))

        cells = list(h3.grid_disk(center_cell, k_rings))
        
        rows = []
        for c in cells:
            c_lat, c_lng = h3.cell_to_latlng(c)
            # Filter strictly within bounding box
            if self.registry.is_within_city_bbox(city_id, c_lat, c_lng):
                area = h3.cell_area(c, unit="km^2")
                rows.append({
                    "h3_cell_id": c,
                    "city": city.city_name,
                    "city_id": city.city_id,
                    "state": city.state_name,
                    "state_code": city.state_code,
                    "country": city.country,
                    "latitude": round(c_lat, 6),
                    "longitude": round(c_lng, 6),
                    "resolution": res,
                    "boundary_area_km2": round(area, 4),
                    "is_water": False,  # Specialized water polygon masking per city
                    "source": "real:multicity_spatial_grid",
                })

        df = pd.DataFrame(rows)
        return df.sort_values("h3_cell_id").reset_index(drop=True)

    def generate_city_static_features(self, city_id: str, h3_grid: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesize or aggregate static spatial POI features for a specific city (Tasks 5, 8).
        """
        city = self.registry.get_city(city_id)
        df = h3_grid.copy()
        rng = np.random.default_rng(abs(hash(city_id)) % (2**32))

        # City density scaling factor
        density_scale = {
            "bengaluru": 1.45,
            "hyderabad": 1.30,
            "visakhapatnam": 0.90,
            "vijayawada": 1.00,
        }.get(city_id, 1.0)

        df["population_density"] = np.round(rng.uniform(3000, 32000, len(df)) * density_scale, 1)
        df["road_density"] = np.round(rng.uniform(1.2, 5.5, len(df)) * density_scale, 2)
        df["office_count"] = np.round(rng.poisson(12 * density_scale, len(df))).astype(int)
        df["restaurant_count"] = np.round(rng.poisson(15 * density_scale, len(df))).astype(int)
        df["mall_count"] = np.round(rng.poisson(2 * density_scale, len(df))).astype(int)
        df["hospital_count"] = np.round(rng.poisson(4 * density_scale, len(df))).astype(int)
        df["college_count"] = np.round(rng.poisson(3 * density_scale, len(df))).astype(int)
        df["school_count"] = np.round(rng.poisson(8 * density_scale, len(df))).astype(int)
        df["park_count"] = np.round(rng.poisson(3, len(df))).astype(int)
        df["bus_stop_count"] = np.round(rng.poisson(6 * density_scale, len(df))).astype(int)
        df["railway_station_count"] = np.round(rng.poisson(1 * density_scale, len(df))).astype(int)
        metro_lambda = 2.5 * density_scale if city_id in ("bengaluru", "hyderabad") else 0.0
        df["metro_station_count"] = np.round(rng.poisson(metro_lambda, len(df))).astype(int)
        df["parking_count"] = np.round(rng.poisson(5 * density_scale, len(df))).astype(int)
        df["building_density"] = np.round(rng.uniform(0.3, 0.85, len(df)) * min(1.0, density_scale), 3)

        res_r = rng.uniform(0.35, 0.65, len(df))
        comm_r = rng.uniform(0.20, 0.45, len(df))
        ind_r = rng.uniform(0.05, 0.20, len(df))
        tot_r = res_r + comm_r + ind_r
        df["residential_ratio"] = np.round(res_r / tot_r, 3)
        df["commercial_ratio"] = np.round(comm_r / tot_r, 3)
        df["industrial_ratio"] = np.round(1.0 - (df["residential_ratio"] + df["commercial_ratio"]), 3)

        df["source"] = f"multicity:{city_id}_static_features"
        return df

    def generate_city_feature_store(
        self,
        city_id: str,
        n_vendors: int = 50,
        start_date: str = "2025-01-01",
        end_date: str = "2025-01-31",
        output_file: Optional[Path] = None,
        seed: int = 2026,
    ) -> pd.DataFrame:
        """
        Builds a canonical 45-column feature store partition for any registered city (Tasks 6, 7).
        """
        city = self.registry.get_city(city_id)
        logger.info(f"Generating feature store for city: {city.city_name} ({start_date} to {end_date})")

        # 1. Spatial Grid & Static Features
        h3_cells = self.generate_h3_grid(city_id)
        static_features = self.generate_city_static_features(city_id, h3_cells)
        cell_ids = h3_cells["h3_cell_id"].tolist()

        # 2. Dynamic Calendar & Weather
        from src.data.calendar_dataset import build_calendar
        from src.data.weather import build_weather_hourly
        from src.data.events import build_events
        from src.data.competition import build_competition
        from src.data.vendor_profiles import build_vendor_profiles
        from src.simulation.market_simulation_v2 import generate_historical_transactions
        from src.features.feature_store import build_feature_store

        cal = build_calendar(start_date, end_date)
        weather = build_weather_hourly(city.center_lat, city.center_lng, start_date, end_date, seed=seed)
        vendors = build_vendor_profiles(cell_ids, n_vendors=n_vendors, seed=seed)
        events_df = build_events(cell_ids, start_date, end_date, seed=seed)
        comp = build_competition(cell_ids, vendors, pd.date_range(start_date, end_date))

        # 3. Market Simulation
        tx = generate_historical_transactions(
            h3_cells=h3_cells,
            static_features=static_features,
            weather_hourly=weather,
            calendar=cal,
            events=events_df,
            competition=comp,
            vendor_profiles=vendors,
            seed=seed,
        )

        # 4. Feature Store Assembly
        fs = build_feature_store(
            historical_transactions=tx,
            static_features=static_features,
            weather_hourly=weather,
            calendar=cal,
            competition=comp,
            events=events_df,
            vendor_profiles=vendors,
        )

        target_path = output_file or (Path("data/processed") / f"feature_store_{city_id}.parquet")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fs.to_parquet(target_path, index=False)
        logger.info(f"Saved {len(fs):,} feature rows for {city.city_name} to {target_path}")
        return fs

    def get_city_onboarding_guide(self) -> str:
        """
        Documentation of standardized protocol for onboarding a new city (Task 17).
        """
        return (
            "# GeoDemand AI — Standard Operating Procedure: City Onboarding Protocol\n\n"
            "To onboard a new metropolitan area into GeoDemand AI:\n"
            "1. Define Geographic Envelope:\n"
            "   - Identify centroid (center_lat, center_lng) and operating radius_km.\n"
            "   - Define bounding box (min_lat, max_lat, min_lng, max_lng).\n"
            "   - Identify state ISO code for regional holiday calendar.\n"
            "2. Register City in `MultiCityRegistry`:\n"
            "   - Instantiate CityConfig and register into registry.\n"
            "3. Generate H3 Spatial Index:\n"
            "   - Run `MultiCityPipeline.generate_h3_grid(city_id)` to partition into Res 8 cells.\n"
            "4. Extract OpenStreetMap & POI Data:\n"
            "   - Query Overpass API with bounding box for offices, restaurants, schools, transit.\n"
            "5. Configure Historical Weather:\n"
            "   - Pull Open-Meteo archive using city centroid.\n"
            "6. Build and Partition Feature Store:\n"
            "   - Run `MultiCityPipeline.generate_city_feature_store(city_id)`.\n"
            "   - Output saved to `data/processed/feature_store_{city_id}.parquet`.\n"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-City Feature Store Pre-Materialization CLI")
    parser.add_argument("--city", type=str, default="all", help="Target city ID (hyderabad, bengaluru, visakhapatnam, or all)")
    parser.add_argument("--vendors", type=int, default=50, help="Number of vendors per city")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-01-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    pipeline = MultiCityPipeline()
    targets = ["hyderabad", "bengaluru", "visakhapatnam"] if args.city == "all" else [args.city.lower()]

    for c in targets:
        print(f"\n--- Pre-materializing feature store for {c.upper()} ---")
        df = pipeline.generate_city_feature_store(
            city_id=c,
            n_vendors=args.vendors,
            start_date=args.start,
            end_date=args.end,
        )
        print(f"Complete: {c} -> {len(df):,} rows with {df.shape[1]} columns. Zero nulls: {df.isnull().sum().sum() == 0}")

