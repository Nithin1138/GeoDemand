"""
Unit Tests for Production Data Release Verification and Freezing (Phase 19.1).
"""

from pathlib import Path
import tempfile
import pytest

from src.data.release_verification import (
    verify_production_manifest,
    validate_production_datasets,
    verify_scheduled_ingestion_flows,
    verify_production_backup_restore,
    run_full_release_verification,
    ReleaseVerificationReport,
)


def test_freeze_production_dataset_versions():
    """Verify production dataset versions and manifests are frozen and deterministic (Task 1)."""
    processed_dir = Path("data/processed")
    assert processed_dir.exists()

    manifest_ok, v_hash, manifest_data = verify_production_manifest(processed_dir)
    assert manifest_ok is True
    assert v_hash == "12c8b1128aa9"
    assert manifest_data["pipeline_schema_version"] == "v2.1-next-hour-forecast"


def test_verify_production_manifests():
    """Verify production manifest accurately reflects all 9 canonical parquet files (Task 2)."""
    processed_dir = Path("data/processed")
    manifest_ok, v_hash, manifest_data = verify_production_manifest(processed_dir)
    assert manifest_ok is True
    assert "files" in manifest_data
    files = manifest_data["files"]
    assert "feature_store.parquet" in files
    assert "weather_hourly.parquet" in files
    assert files["feature_store.parquet"]["n_rows"] == 711750


def test_validate_production_data_sources():
    """Verify all production datasets are valid, non-empty, with zero nulls (Task 3)."""
    processed_dir = Path("data/processed")
    sources_ok, row_counts, errors = validate_production_datasets(processed_dir)
    assert sources_ok is True, f"Validation errors: {errors}"
    assert len(errors) == 0
    assert row_counts["feature_store.parquet"] == 711750
    assert row_counts["h3_cells.parquet"] == 817
    assert row_counts["static_features.parquet"] == 817


def test_scheduled_ingestion_jobs():
    """Verify scheduled competitor refresh and transaction ingestion run cleanly (Task 4)."""
    sched_ok, errors = verify_scheduled_ingestion_flows()
    assert sched_ok is True, f"Scheduled flow errors: {errors}"
    assert len(errors) == 0


def test_production_backup_and_restore():
    """Verify automated production backup and restoration is lossless (Task 5)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backup_ok, errors = verify_production_backup_restore(Path(tmp_dir))
        assert backup_ok is True, f"Backup/restore errors: {errors}"
        assert len(errors) == 0


def test_full_release_verification_report():
    """Verify full production release report marks status as release ready."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = run_full_release_verification(tmp_dir=Path(tmp_dir))
        assert isinstance(report, ReleaseVerificationReport)
        assert report.is_release_ready is True
        assert report.manifest_verified is True
        assert report.data_sources_valid is True
        assert report.scheduled_ingestion_verified is True
        assert report.backup_restore_verified is True
        assert len(report.validation_errors) == 0
