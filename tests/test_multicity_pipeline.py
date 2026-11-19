"""
Unit and Integration Tests for Multi-City Data Engineering Pipeline (Phase 14.1).
"""

import pandas as pd
import pytest

from src.data.multicity import (
    CityConfig,
    MultiCityRegistry,
    MultiCityPipeline,
    SUPPORTED_CITIES,
)


@pytest.fixture
def registry():
    return MultiCityRegistry()


@pytest.fixture
def pipeline(registry):
    return MultiCityPipeline(registry)


def test_supported_cities_registry(registry):
    """Verify all four target cities are registered with complete configuration (Tasks 1, 2, 9, 11, 13)."""
    cities = registry.list_cities()
    assert "vijayawada" in cities
    assert "hyderabad" in cities
    assert "bengaluru" in cities
    assert "visakhapatnam" in cities

    hyd = registry.get_city("hyderabad")
    assert hyd.state_code == "TS"
    assert hyd.country == "India"
    assert hyd.h3_resolution == 8

    blr = registry.get_city("bengaluru")
    assert blr.state_code == "KA"
    assert blr.radius_km >= 10.0

    viz = registry.get_city("visakhapatnam")
    assert viz.state_code == "AP"


def test_key_neighborhood_coverage(registry):
    """Verify specific commercial and tech corridors are explicitly covered (Tasks 10, 12, 14)."""
    # Hyderabad: Cyberabad and HITEC City (Task 10)
    hyd = registry.get_city("hyderabad")
    assert any("cyberabad" in n.lower() for n in hyd.key_neighborhoods)
    assert any("hitec city" in n.lower() for n in hyd.key_neighborhoods)

    # Bengaluru: Koramangala, Indiranagar, Electronic City (Task 12)
    blr = registry.get_city("bengaluru")
    assert any("koramangala" in n.lower() for n in blr.key_neighborhoods)
    assert any("indiranagar" in n.lower() for n in blr.key_neighborhoods)
    assert any("electronic city" in n.lower() for n in blr.key_neighborhoods)

    # Visakhapatnam: Beach Road and MVP Colony (Task 14)
    viz = registry.get_city("visakhapatnam")
    assert any("beach road" in n.lower() for n in viz.key_neighborhoods)
    assert any("mvp colony" in n.lower() for n in viz.key_neighborhoods)


def test_bounding_box_checks(registry):
    """Verify coordinate bounding box validation across different cities (Tasks 3, 16)."""
    # Hyderabad HITEC City coordinates (17.4435, 78.3772)
    assert registry.is_within_city_bbox("hyderabad", 17.4435, 78.3772) is True
    # Vijayawada coordinates should be rejected for Hyderabad
    assert registry.is_within_city_bbox("hyderabad", 16.5062, 80.6480) is False

    # Bengaluru Koramangala coordinates (12.9352, 77.6245)
    assert registry.is_within_city_bbox("bengaluru", 12.9352, 77.6245) is True


def test_custom_city_registration(registry):
    """Test dynamic registration of an arbitrary new city (Tasks 2, 3, 16)."""
    custom_city = CityConfig(
        city_id="chennai",
        city_name="Chennai",
        state_name="Tamil Nadu",
        state_code="TN",
        country="India",
        center_lat=13.0827,
        center_lng=80.2707,
        radius_km=10.0,
        bbox={"min_lat": 12.95, "max_lat": 13.20, "min_lng": 80.15, "max_lng": 80.35},
        key_neighborhoods=["T. Nagar", "Adyar", "Velachery", "OMR"],
    )
    registry.register_city(custom_city)
    assert "chennai" in registry.list_cities()
    assert registry.is_within_city_bbox("chennai", 13.0827, 80.2707) is True


def test_h3_grid_generation_by_city(pipeline):
    """Test generating H3 Resolution 8 spatial grid for Hyderabad and Bengaluru (Tasks 4, 15)."""
    # Hyderabad grid
    hyd_grid = pipeline.generate_h3_grid("hyderabad")
    assert len(hyd_grid) > 50
    assert hyd_grid["city"].iloc[0] == "Hyderabad"
    assert hyd_grid["state_code"].iloc[0] == "TS"
    assert hyd_grid["resolution"].iloc[0] == 8
    # Validate all generated cells are 15-char hex
    assert all(len(c) == 15 for c in hyd_grid["h3_cell_id"])

    # Bengaluru grid
    blr_grid = pipeline.generate_h3_grid("bengaluru")
    assert len(blr_grid) > 50
    assert blr_grid["city"].iloc[0] == "Bengaluru"
    assert blr_grid["state_code"].iloc[0] == "KA"


def test_city_static_features_generation(pipeline):
    """Test generating city-specific static POI features (Tasks 5, 8)."""
    hyd_grid = pipeline.generate_h3_grid("hyderabad").head(20)
    features = pipeline.generate_city_static_features("hyderabad", hyd_grid)

    assert len(features) == 20
    assert "population_density" in features.columns
    assert "office_count" in features.columns
    assert "restaurant_count" in features.columns
    assert features["population_density"].mean() > 0


def test_city_onboarding_guide(pipeline):
    """Verify city onboarding guide documentation is available (Task 17)."""
    guide = pipeline.get_city_onboarding_guide()
    assert "Standard Operating Procedure: City Onboarding Protocol" in guide
    assert "MultiCityRegistry" in guide
    assert "H3 Spatial Index" in guide


def test_city_feature_store_prematerialization(pipeline, tmp_path):
    """Verify multi-city feature store generation produces canonical 45 columns with zero nulls."""
    out_file = tmp_path / "test_hyd_feature_store.parquet"
    df = pipeline.generate_city_feature_store(
        city_id="hyderabad",
        n_vendors=5,
        start_date="2025-01-01",
        end_date="2025-01-03",
        output_file=out_file,
        seed=42,
    )
    assert out_file.exists()
    assert len(df) == 5 * 3 * 13  # 5 vendors * 3 days * 13 hours = 195 rows
    assert df.shape[1] == 45
    assert "expected_customer_count" in df.columns
    assert "temperature" in df.columns
    assert "vendor_category" in df.columns
    assert df.isnull().sum().sum() == 0

