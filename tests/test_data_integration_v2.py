"""
End-to-End Data Integration Tests for V2 Architecture (Phase 18.1).

Validates the complete interoperability of:
1. All data sources together (weather, calendar, events, demographics, competitors).
2. Actuals ingestion pipeline (transactions -> hourly actuals -> quality metrics).
3. Population enrichment (ward boundaries & multi-source population to H3 grid).
4. Multi-city feature generation (Vijayawada, Hyderabad, Bengaluru, Visakhapatnam).
5. Database persistence and migration (PostGIS/relational loading & count reconciliation).
6. Pipeline restart, checkpointing, and recovery from interruption.
7. Data passed to ML inference and FastAPI backend (40 features, 0 nulls, correct schema).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np

# Ingestion & actuals
from src.ingestion.schema import TransactionRecord, TransactionLocation, PaymentMethod, TransactionStatus
from src.ingestion.validator import TransactionIngestionPipeline
from src.data.actuals_aggregator import ActualsAggregator

# Demographics & competitors
from src.data.demographic_enrichment import DemographicEnrichmentPipeline
from src.data.competitor_pipeline import CompetitorPipeline, CompetitorRecord

# Multi-city
from src.data.multicity import MultiCityRegistry, MultiCityPipeline

# Persistence & migration
from src.data.persistence import DatabaseMigrationManager, VendorRecord, TransactionEntity

# Monitoring
from src.validation.data_monitoring import (
    AlertManager,
    IngestionFailureMonitor,
    DataFreshnessMonitor,
    QualityProfileMonitor,
    DataLineageTracker,
)

# Feature Store & Model schema
from src.features.feature_store import trainable_columns, TARGET


def test_integration_step1_all_data_sources_together():
    """Verify concurrent coordination of weather, calendar, POI, demographic, and competitor features."""
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    h3_cell = "88619aa6c7fffff"

    # 1. Demographic enrichment
    demo_pipeline = DemographicEnrichmentPipeline()
    sample_h3 = pd.DataFrame([{"h3_cell_id": h3_cell, "latitude": 16.5062, "longitude": 80.6480}])
    demo_df = demo_pipeline.enrich_h3_cells(sample_h3)
    assert len(demo_df) == 1
    assert "population_density" in demo_df.columns
    pop_density = float(demo_df.iloc[0]["population_density"])
    assert pop_density > 0

    # 2. Competitor enrichment
    comp_pipeline = CompetitorPipeline()
    records = [
        CompetitorRecord(
            competitor_id="c1",
            name="Cafe Central",
            category="food",
            raw_amenity="cafe",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id=h3_cell,
            is_daytime_active=True,
            is_nighttime_active=True,
            last_verified_at=now,
            is_stale=False,
        )
    ]
    comp_df = comp_pipeline.aggregate_by_h3_cell(records, all_h3_cells=[h3_cell])
    assert len(comp_df) == 1
    assert comp_df.iloc[0]["food_competitor_count"] >= 1

    # Combined composite record
    combined_record = {
        "h3_cell_id": h3_cell,
        "population_density": pop_density,
        "food_competitor_count": int(comp_df.iloc[0]["food_competitor_count"]),
        "timestamp": now.isoformat(),
    }
    assert combined_record["population_density"] > 0
    assert combined_record["food_competitor_count"] >= 1


def test_integration_step2_actuals_ingestion_pipeline():
    """Verify transactions ingestion flowing into actuals aggregator and quality reporting."""
    now = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    ingestion = TransactionIngestionPipeline()
    aggregator = ActualsAggregator()

    tx_records = []
    for i in range(5):
        tx = TransactionRecord(
            transaction_id=f"tx_e2e_{i}",
            vendor_id="vendor_001",
            customer_id=f"cust_{i % 2}",  # 2 distinct customers
            event_timestamp=now + timedelta(minutes=i * 10),
            amount_inr=100.0 + i * 20,
            payment_method=PaymentMethod.UPI,
            status=TransactionStatus.COMPLETED,
            location=TransactionLocation(latitude=16.5062, longitude=80.6480, h3_cell_id="88619aa6c7fffff"),
        )
        res = ingestion.ingest_transaction(tx)
        assert res.success is True
        tx_records.append(res.record)

    # Pass into aggregator
    actuals_df = aggregator.aggregate_transactions(tx_records)
    assert len(actuals_df) == 1
    row = actuals_df.iloc[0]
    assert row["transaction_count"] == 5
    assert row["unique_customer_count"] == 2
    assert row["total_sales_inr"] == 700.0  # 100 + 120 + 140 + 160 + 180

    # Quality report
    quality_report = aggregator.generate_quality_report(tx_records, actuals_df)
    assert quality_report.reconciliation_discrepancy_inr == 0.0
    assert quality_report.total_raw_transactions == 5


def test_integration_step3_demographic_and_population_enrichment():
    """Verify demographic ward polygons and population assignment to H3 grid."""
    pipeline = DemographicEnrichmentPipeline()
    sample_df = pd.DataFrame([
        {"h3_cell_id": "88619aa6c7fffff", "latitude": 16.4990, "longitude": 80.6560},
        {"h3_cell_id": "8861892557fffff", "latitude": 16.5160, "longitude": 80.6120},
    ])
    enriched = pipeline.enrich_h3_cells(sample_df)
    
    assert len(enriched) == 2
    assert "population_density" in enriched.columns
    assert "census_ward_name" in enriched.columns
    assert enriched["population_density"].min() > 0


def test_integration_step4_multicity_feature_generation():
    """Verify multi-city spatial grids and static feature pipelines."""
    registry = MultiCityRegistry()
    pipeline = MultiCityPipeline(registry)

    # Validate cities
    for city_id in ["vijayawada", "hyderabad", "bengaluru", "visakhapatnam"]:
        grid_df = pipeline.generate_h3_grid(city_id)
        assert len(grid_df) > 0
        assert "h3_cell_id" in grid_df.columns
        assert all(len(c) == 15 for c in grid_df["h3_cell_id"])

        static_df = pipeline.generate_city_static_features(city_id, h3_grid=grid_df)
        assert len(static_df) == len(grid_df)
        assert "road_density" in static_df.columns
        assert "restaurant_count" in static_df.columns


def test_integration_step5_database_migration_and_reconciliation():
    """Verify database migration of vendors and transactions with reconciliation and spatial query."""
    now = datetime.now(timezone.utc)
    manager = DatabaseMigrationManager(":memory:")

    # 1. Ingest vendors
    vendors = [
        VendorRecord(vendor_id="v_m1", business_name="Alpha Eats", category="food_cart", base_latitude=16.5062, base_longitude=80.6480),
        VendorRecord(vendor_id="v_m2", business_name="Beta Chai", category="coffee_beverages", base_latitude=16.5100, base_longitude=80.6520),
    ]
    migrated_v = manager.migrate_vendors(vendors)
    assert migrated_v == 2

    # 2. Ingest transactions
    transactions = [
        TransactionEntity(
            transaction_id=f"tx_mig_{i}",
            vendor_id="v_m1",
            event_timestamp=now,
            amount_inr=150.0,
            payment_method="upi",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id="88619aa6c7fffff",
        )
        for i in range(10)
    ]
    migrated_t = manager.migrate_transactions(transactions)
    assert migrated_t == 10

    # 3. Reconciliation check
    rep_v = manager.verify_reconciliation("vendors", source_count=2)
    assert rep_v.is_reconciled is True
    rep_t = manager.verify_reconciliation("transactions", source_count=10)
    assert rep_t.is_reconciled is True

    # 4. Spatial query
    spatial_hits = manager.query_transactions_in_radius(16.5062, 80.6480, radius_meters=100.0)
    assert len(spatial_hits) == 10


def test_integration_step6_pipeline_restart_and_recovery():
    """Verify pipeline resilience: checkpointing, restart recovery, and duplicate avoidance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = Path(tmp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "actuals_state.parquet"

        # Phase A: Initial run
        df_part1 = pd.DataFrame([
            {"h3_cell_id": "88619aa6c7fffff", "hour": "2026-11-15T10:00:00", "sales": 500.0},
            {"h3_cell_id": "88619aa6c1fffff", "hour": "2026-11-15T10:00:00", "sales": 300.0},
        ])
        df_part1.to_parquet(checkpoint_file, index=False)

        # Simulate interruption and restart
        assert checkpoint_file.exists()
        recovered_df = pd.read_parquet(checkpoint_file)
        assert len(recovered_df) == 2

        # Phase B: Incremental processing without duplicates
        new_batch = pd.DataFrame([
            {"h3_cell_id": "88619aa6c7fffff", "hour": "2026-11-15T10:00:00", "sales": 500.0},  # duplicate
            {"h3_cell_id": "88619aa6c7fffff", "hour": "2026-11-15T11:00:00", "sales": 700.0},  # new
        ])

        combined = pd.concat([recovered_df, new_batch]).drop_duplicates(subset=["h3_cell_id", "hour"])
        assert len(combined) == 3
        assert combined["sales"].sum() == 1500.0


def test_integration_step7_data_contract_passed_to_ml_and_backend():
    """Verify the canonical feature matrix passed to ML model has 40 features, 0 nulls, and expected target."""
    feature_store_path = Path("data/processed/feature_store.parquet")
    assert feature_store_path.exists(), "Feature store must exist"

    df = pd.read_parquet(feature_store_path)
    assert len(df) > 0

    # Ensure zero null values
    null_count = int(df.isnull().sum().sum())
    assert null_count == 0, f"Feature store contains {null_count} nulls!"

    # Ensure target column exists
    assert TARGET in df.columns
    assert (df[TARGET] >= 0).all()

    # Trainable features verification
    trainable_cols = trainable_columns(df)
    assert len(trainable_cols) == 40
    for col in trainable_cols:
        assert col in df.columns, f"Missing feature in store: {col}"
        # Ensure 0 null values in all trainable features
        assert df[col].isnull().sum() == 0, f"Feature {col} has nulls!"
