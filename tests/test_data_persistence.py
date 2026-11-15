"""
Unit Tests for Data Persistence, PostGIS Schemas, and Migration (Phase 15.1).
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import pandas as pd

from src.data.persistence import (
    DATASET_CATALOG,
    DatabaseMigrationManager,
    DatasetCatalogEntry,
    LocationEventEntity,
    ReconciliationReport,
    ReservationEntity,
    StorageTier,
    TransactionEntity,
    VendorRecord,
    classify_dataset,
    generate_postgres_ddl,
    haversine_distance_meters,
    is_parquet_recommended,
    validate_postgis_geometry,
)


def test_dataset_classification_and_catalog():
    """Verify operational vs analytical dataset classification."""
    # Operational checks
    vendor_entry = classify_dataset("vendors")
    assert vendor_entry.tier == StorageTier.OPERATIONAL
    assert "PostgreSQL" in vendor_entry.target_engine

    trans_entry = classify_dataset("transactions")
    assert trans_entry.tier == StorageTier.OPERATIONAL
    assert trans_entry.retention_days == 90

    res_entry = classify_dataset("reservations")
    assert res_entry.tier == StorageTier.OPERATIONAL

    loc_entry = classify_dataset("location_events")
    assert loc_entry.tier == StorageTier.OPERATIONAL

    # Analytical checks
    feat_entry = classify_dataset("feature_store")
    assert feat_entry.tier == StorageTier.ANALYTICAL
    assert feat_entry.recommended_format == "parquet"

    demo_entry = classify_dataset("demographic_enrichment")
    assert demo_entry.tier == StorageTier.ANALYTICAL

    comp_entry = classify_dataset("competitor_snapshots")
    assert comp_entry.tier == StorageTier.ANALYTICAL


def test_parquet_retention_rules():
    """Verify analytical datasets are recommended for Parquet storage."""
    assert is_parquet_recommended("feature_store") is True
    assert is_parquet_recommended("hourly_actuals") is True
    assert is_parquet_recommended("static_features") is True
    assert is_parquet_recommended("transactions") is False
    assert is_parquet_recommended("vendors") is False


def test_vendor_and_transaction_schemas():
    """Verify Pydantic entity validation for vendors and transactions."""
    now = datetime.now(timezone.utc)
    vendor = VendorRecord(
        vendor_id="v_001",
        business_name="Vijayawada Chai Point",
        category="coffee_beverages",
        phone="+919876543210",
        base_latitude=16.5062,
        base_longitude=80.6480,
    )
    assert vendor.vendor_id == "v_001"
    assert vendor.base_latitude == 16.5062

    # Valid Transaction
    trans = TransactionEntity(
        transaction_id="tx_101",
        vendor_id="v_001",
        event_timestamp=now,
        amount_inr=150.0,
        payment_method="upi",
        latitude=16.5062,
        longitude=80.6480,
        h3_cell_id="88619aa6c7fffff",
    )
    assert trans.amount_inr == 150.0

    # Negative amount failure
    with pytest.raises(Exception):
        TransactionEntity(
            transaction_id="tx_bad",
            vendor_id="v_001",
            event_timestamp=now,
            amount_inr=-50.0,
            payment_method="upi",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id="88619aa6c7fffff",
        )

    # Invalid H3 cell failure
    with pytest.raises(ValueError):
        TransactionEntity(
            transaction_id="tx_bad_cell",
            vendor_id="v_001",
            event_timestamp=now,
            amount_inr=50.0,
            payment_method="upi",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id="invalid_cell_hex",
        )


def test_reservation_and_location_schemas():
    """Verify reservation and location events validation."""
    now = datetime.now(timezone.utc)
    slot_end = now + timedelta(hours=2)

    res = ReservationEntity(
        reservation_id="res_01",
        vendor_id="v_001",
        h3_cell_id="88619aa6c7fffff",
        slot_start=now,
        slot_end=slot_end,
        status="confirmed",
        deposit_amount_inr=250.0,
    )
    assert res.deposit_amount_inr == 250.0

    # Invalid slot order (end before start)
    with pytest.raises(ValueError):
        ReservationEntity(
            reservation_id="res_bad",
            vendor_id="v_001",
            h3_cell_id="88619aa6c7fffff",
            slot_start=slot_end,
            slot_end=now,  # Invalid
        )

    # Location event
    loc = LocationEventEntity(
        event_id="loc_01",
        vendor_id="v_001",
        timestamp=now,
        latitude=16.5062,
        longitude=80.6480,
        h3_cell_id="88619aa6c7fffff",
        battery_level=0.85,
        speed_kmh=12.5,
        accuracy_meters=5.0,
    )
    assert loc.battery_level == 0.85


def test_postgis_geometry_validation():
    """Verify PostGIS geometry validation and WKT/GeoJSON generation."""
    geom = validate_postgis_geometry(16.5062, 80.6480)
    assert geom["srid"] == 4326
    assert geom["wkt"] == "SRID=4326;POINT(80.648000 16.506200)"
    assert geom["geojson"]["coordinates"] == [80.648, 16.5062]
    assert "ST_SetSRID(ST_MakePoint" in geom["sql_expression"]

    # Invalid lat/lon
    with pytest.raises(ValueError):
        validate_postgis_geometry(95.0, 80.0)
    with pytest.raises(ValueError):
        validate_postgis_geometry(16.0, 195.0)


def test_postgres_ddl_generation():
    """Verify generated PostgreSQL + PostGIS DDL script."""
    ddl = generate_postgres_ddl()
    assert "CREATE EXTENSION IF NOT EXISTS postgis;" in ddl
    assert "CREATE TABLE IF NOT EXISTS vendors" in ddl
    assert "CREATE TABLE IF NOT EXISTS transactions" in ddl
    assert "CREATE TABLE IF NOT EXISTS reservations" in ddl
    assert "CREATE TABLE IF NOT EXISTS location_events" in ddl
    assert "idx_vendors_geom ON vendors USING GIST (geom);" in ddl
    assert "idx_transactions_geom ON transactions USING GIST (geom);" in ddl
    assert "idx_transactions_vendor_time" in ddl


def test_migration_and_reconciliation_workflow():
    """Verify database migration, index generation, and count reconciliation."""
    manager = DatabaseMigrationManager(":memory:")
    now = datetime.now(timezone.utc)

    # Migrate vendors
    vendors = [
        VendorRecord(vendor_id="v_1", business_name="V1 Cafe", category="coffee_beverages", base_latitude=16.5062, base_longitude=80.6480),
        VendorRecord(vendor_id="v_2", business_name="V2 Cart", category="food_cart", base_latitude=16.5100, base_longitude=80.6500),
    ]
    migrated_v = manager.migrate_vendors(vendors)
    assert migrated_v == 2

    # Reconcile vendors
    v_report = manager.verify_reconciliation("vendors", source_count=2)
    assert v_report.is_reconciled is True
    assert v_report.discrepancy == 0

    # Migrate transactions
    transactions = [
        TransactionEntity(
            transaction_id=f"tx_{i}",
            vendor_id="v_1",
            event_timestamp=now + timedelta(minutes=i * 10),
            amount_inr=100.0 + i * 20,
            payment_method="upi",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id="88619aa6c7fffff",
        )
        for i in range(5)
    ]
    migrated_t = manager.migrate_transactions(transactions)
    assert migrated_t == 5

    t_report = manager.verify_reconciliation("transactions", source_count=5)
    assert t_report.is_reconciled is True
    assert t_report.target_count == 5


def test_spatial_radius_and_bbox_queries():
    """Verify spatial radius and bounding box search functionality."""
    manager = DatabaseMigrationManager(":memory:")
    now = datetime.now(timezone.utc)

    # Add vendor in Benz Circle, Vijayawada (16.5062, 80.6480)
    manager.migrate_vendors([
        VendorRecord(vendor_id="v_benz", business_name="Benz Cafe", category="coffee_beverages", base_latitude=16.5062, base_longitude=80.6480),
        VendorRecord(vendor_id="v_ दूर", business_name="Far Out Cart", category="food_cart", base_latitude=16.6000, base_longitude=80.7500),
    ])

    # Add transaction close to Benz circle (< 200m)
    manager.migrate_transactions([
        TransactionEntity(
            transaction_id="tx_near",
            vendor_id="v_benz",
            event_timestamp=now,
            amount_inr=200.0,
            payment_method="upi",
            latitude=16.5065,
            longitude=80.6482,  # ~40 meters away
            h3_cell_id="88619aa6c7fffff",
        ),
        TransactionEntity(
            transaction_id="tx_far",
            vendor_id="v_benz",
            event_timestamp=now,
            amount_inr=500.0,
            payment_method="card",
            latitude=16.5500,
            longitude=80.7000,  # ~7 km away
            h3_cell_id="88619aa6c7fffff",
        ),
    ])

    # Test radius search within 500m
    results = manager.query_transactions_in_radius(16.5062, 80.6480, radius_meters=500.0)
    assert len(results) == 1
    assert results[0]["transaction_id"] == "tx_near"
    assert results[0]["distance_meters"] < 100.0

    # Test bounding box search
    bbox_vendors = manager.query_vendors_in_bounding_box(
        min_lat=16.5000, min_lon=80.6400,
        max_lat=16.5200, max_lon=80.6600,
    )
    assert len(bbox_vendors) == 1
    assert bbox_vendors[0]["vendor_id"] == "v_benz"


def test_backup_and_restore_lifecycle():
    """Verify logical database backup and restoration."""
    manager = DatabaseMigrationManager(":memory:")
    now = datetime.now(timezone.utc)

    # Seed data
    manager.migrate_vendors([
        VendorRecord(vendor_id="v_seed", business_name="Seed Vendor", category="food_cart", base_latitude=16.5062, base_longitude=80.6480),
    ])
    manager.migrate_transactions([
        TransactionEntity(
            transaction_id="tx_seed",
            vendor_id="v_seed",
            event_timestamp=now,
            amount_inr=120.0,
            payment_method="cash",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id="88619aa6c7fffff",
        )
    ])

    with tempfile.TemporaryDirectory() as tmp_dir:
        backup_file = Path(tmp_dir) / "test_db_backup.json"
        backup_res = manager.create_backup(backup_file)
        assert backup_res["status"] == "success"
        assert backup_file.exists()

        # Create a fresh database and restore
        fresh_manager = DatabaseMigrationManager(":memory:")
        restore_res = fresh_manager.restore_backup(backup_file)
        assert restore_res["status"] == "restored"
        assert restore_res["restored_counts"]["vendors"] == 1
        assert restore_res["restored_counts"]["transactions"] == 1

        # Verify restored contents
        report = fresh_manager.verify_reconciliation("vendors", source_count=1)
        assert report.is_reconciled is True
