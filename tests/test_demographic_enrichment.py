"""
Unit and Integration Tests for Census and Demographic Enrichment Pipeline (Phase 12.1).
"""

import pandas as pd
import pytest

from src.data.demographic_enrichment import (
    SOURCE_METADATA,
    WardBoundary,
    clean_polygon,
    polygon_area_km2,
    point_in_polygon,
    validate_crs,
    get_vijayawada_census_wards,
    DemographicEnrichmentPipeline,
)


def test_census_ward_boundaries_and_metadata():
    """Verify Census 2011 wards exist with populations and required fields (Tasks 1 & 2)."""
    wards = get_vijayawada_census_wards()
    assert len(wards) >= 15
    for w in wards:
        assert isinstance(w, WardBoundary)
        assert w.ward_id.startswith("WARD_")
        assert w.census_2011_population > 10000
        assert w.area_km2 > 0.1
        assert len(w.polygon) >= 4  # closed ring


def test_crs_validation():
    """Verify EPSG:4326 Coordinate Reference System bounds validation (Task 3)."""
    valid_coords = [(16.5062, 80.6480), (16.5176, 80.5278)]
    assert validate_crs(valid_coords, "EPSG:4326") is True

    # Out of bounds latitude
    invalid_lat = [(95.0, 80.6480)]
    assert validate_crs(invalid_lat, "EPSG:4326") is False

    # Out of bounds longitude
    invalid_lng = [(16.5062, 190.0)]
    assert validate_crs(invalid_lng, "EPSG:4326") is False


def test_polygon_cleaning_and_repair():
    """Verify polygon cleaning, ring closure, and degeneracy checks (Task 4)."""
    # Unclosed ring with consecutive duplicate
    unclosed = [(16.5, 80.6), (16.5, 80.6), (16.6, 80.6), (16.6, 80.7)]
    cleaned = clean_polygon(unclosed)
    
    assert len(cleaned) == 4  # 3 distinct pts + 1 closed ring point
    assert cleaned[0] == cleaned[-1]  # Closed ring

    # Degenerate polygon (< 3 distinct points)
    with pytest.raises(ValueError):
        clean_polygon([(16.5, 80.6), (16.5, 80.6)])


def test_polygon_area_and_point_in_polygon():
    """Test geometric area and point-in-polygon containment (Tasks 4 & 5)."""
    # Simple rectangular polygon
    poly = [(16.50, 80.60), (16.50, 80.70), (16.60, 80.70), (16.60, 80.60), (16.50, 80.60)]
    area = polygon_area_km2(poly)
    assert area > 10.0  # Approx ~111 km x ~106 km = ~117 km²

    # Inside point
    assert point_in_polygon(16.55, 80.65, poly) is True
    # Outside point
    assert point_in_polygon(16.40, 80.65, poly) is False


def test_h3_demographic_enrichment():
    """Test H3 cell demographic enrichment with Census, HRSL, and Kontur data (Tasks 5–8, 12, 13)."""
    pipeline = DemographicEnrichmentPipeline()

    # Create sample H3 cells (one in Benz Circle core, one in One Town, one peri-urban outside)
    h3_sample = pd.DataFrame([
        {"h3_cell_id": "88619aa6c7fffff", "latitude": 16.4990, "longitude": 80.6560},  # Benz Circle
        {"h3_cell_id": "8861892557fffff", "latitude": 16.5160, "longitude": 80.6120},  # One Town
        {"h3_cell_id": "8861892551fffff", "latitude": 16.4400, "longitude": 80.5300},  # Peri-Urban border
    ])

    enriched = pipeline.enrich_h3_cells(h3_sample)

    assert len(enriched) == 3
    # Check required demographic columns
    expected_cols = [
        "census_ward_id", "census_ward_name", "census_population_density",
        "meta_hrsl_population_density", "kontur_population_density",
        "population_density", "demographic_provenance",
    ]
    for col in expected_cols:
        assert col in enriched.columns

    # Benz Circle / Siddhartha Nagar row
    benz_row = enriched[enriched["h3_cell_id"] == "88619aa6c7fffff"].iloc[0]
    assert any(term in benz_row["census_ward_name"] for term in ["Benz Circle", "Siddhartha Nagar"])
    assert benz_row["census_population_density"] > 5000.0
    assert benz_row["meta_hrsl_population_density"] > 5000.0
    assert benz_row["kontur_population_density"] > 5000.0

    # Peri-urban fallback row
    peri_row = enriched[enriched["h3_cell_id"] == "8861892551fffff"].iloc[0]
    assert "fallback" in peri_row["demographic_provenance"]
    assert peri_row["population_density"] > 0.0


def test_multi_source_comparison_report_and_licensing():
    """Test comparison report generation and licensing verification (Tasks 9, 10, 11)."""
    pipeline = DemographicEnrichmentPipeline()
    h3_sample = pd.DataFrame([
        {"h3_cell_id": "cell_1", "latitude": 16.4990, "longitude": 80.6560},
        {"h3_cell_id": "cell_2", "latitude": 16.5160, "longitude": 80.6120},
        {"h3_cell_id": "cell_3", "latitude": 16.5140, "longitude": 80.5920},
    ])
    enriched = pipeline.enrich_h3_cells(h3_sample)
    report = pipeline.generate_comparison_report(enriched)

    assert report["total_cells_enriched"] == 3
    assert report["mean_density_census_2011"] > 0
    assert report["mean_density_meta_hrsl_2020"] > 0
    assert report["mean_density_kontur_2023"] > 0
    assert "sources" in report

    # Verify licenses
    licenses = report["sources"]
    assert "census_2011" in licenses and "GODL" in licenses["census_2011"]["license"]
    assert "meta_hrsl" in licenses and "CC-BY" in licenses["meta_hrsl"]["license"]
    assert "kontur_population" in licenses and "ODbL" in licenses["kontur_population"]["license"]
