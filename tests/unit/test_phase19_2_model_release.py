"""
GeoDemand AI — Phase 19.2 Model Release Unit & Integration Tests

Member 2 — Machine Learning & MLOps

Validates all 6 tasks of Phase 19.2 Model Release:
1. Freeze model version (SHA-256 verification and model_version.json)
2. Freeze feature schema (45 canonical features, ordering, constraints, schema_sha256)
3. Package dependencies (serving environment requirements and MODEL_CARD.json)
4. Production smoke predictions (point, interval, multi-city, latency <= 3.0ms, edge cases)
5. Verify registry metadata (target contract: expected_customer_count, provenance)
6. Verify rollback capability (backup snapshot loadability, checksum, rollback readiness)
"""

import json
from pathlib import Path
import pytest

from src.geodemand.models.release_manager import ModelReleaseManager, ReleaseManifest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


@pytest.fixture(scope="module")
def release_manager() -> ModelReleaseManager:
    return ModelReleaseManager(models_dir=MODELS_DIR)


# ==============================================================================
# Task 1: Freeze Model Version
# ==============================================================================
def test_task1_freeze_model_version(release_manager):
    """Verify model version freeze produces SHA-256 hashes and model_version.json."""
    ver_data = release_manager.freeze_model_version(version="v2.1.0")

    assert ver_data["model_version"] == "v2.1.0"
    assert ver_data["release_tag"] == "v2.1.0-prod"
    assert "primary_checksum_sha256" in ver_data
    assert len(ver_data["primary_checksum_sha256"]) == 64  # SHA-256 is 64 hex chars
    assert "demand_model.pkl" in ver_data["all_artifact_checksums"]

    # Verify file written to disk
    ver_file = MODELS_DIR / "model_version.json"
    assert ver_file.exists()
    with open(ver_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["model_version"] == "v2.1.0"


# ==============================================================================
# Task 2: Freeze Feature Schema
# ==============================================================================
def test_task2_freeze_feature_schema(release_manager):
    """Verify feature schema freeze produces frozen_feature_schema.json with 45 ordered features."""
    schema = release_manager.freeze_feature_schema()

    assert schema["schema_version"] == "v2.2-canonical"
    assert schema["target_column"] == "expected_customer_count"
    assert schema["total_features"] == 45
    assert len(schema["feature_columns"]) == 45

    # Check strict alphabetical ordering
    assert schema["feature_columns"] == sorted(schema["feature_columns"])

    # Check domain constraints
    assert "domain_constraints" in schema
    assert schema["domain_constraints"]["expected_customer_count"]["allow_negative"] is False
    assert "schema_sha256" in schema
    assert len(schema["schema_sha256"]) == 64

    # Verify file written to disk
    schema_file = MODELS_DIR / "frozen_feature_schema.json"
    assert schema_file.exists()


# ==============================================================================
# Task 3: Package Dependencies & Model Card
# ==============================================================================
def test_task3_package_dependencies(release_manager):
    """Verify packaging of serving dependencies and generation of MODEL_CARD.json."""
    pkg = release_manager.package_dependencies()

    assert "requirements" in pkg
    assert any("lightgbm" in r for r in pkg["requirements"])
    assert any("joblib" in r for r in pkg["requirements"])
    assert any("shap" in r for r in pkg["requirements"])

    # Verify requirements file
    req_file = MODELS_DIR / "environment_ml_requirements.txt"
    assert req_file.exists()
    content = req_file.read_text(encoding="utf-8")
    assert "lightgbm" in content

    # Verify Model Card
    card_file = MODELS_DIR / "MODEL_CARD.json"
    assert card_file.exists()
    with open(card_file, "r", encoding="utf-8") as f:
        card = json.load(f)
        assert card["model_details"]["version"] == "v2.1.0"
        assert "intended_use" in card
        assert "metrics" in card


# ==============================================================================
# Task 4: Run Production Smoke Predictions
# ==============================================================================
def test_task4_production_smoke_predictions(release_manager):
    """Verify production smoke predictions pass all latency, quantile, and multi-city checks."""
    smoke = release_manager.run_production_smoke_predictions()

    assert smoke["all_passed"] is True
    check_names = [c["name"] for c in smoke["checks"]]

    assert "single_sample_point_prediction" in check_names
    assert "quantile_monotonicity" in check_names
    assert "shap_explainability" in check_names
    assert "multi_city_routing_4_cities" in check_names
    assert "boundary_safety_rejection" in check_names
    assert "latency_sla_compliance" in check_names

    # Verify latency SLA compliance (budget <= 3.0ms)
    lat_check = next(c for c in smoke["checks"] if c["name"] == "latency_sla_compliance")
    assert lat_check["passed"] is True
    assert lat_check["per_sample_latency_ms"] <= 3.0


# ==============================================================================
# Task 5: Verify Registry Metadata
# ==============================================================================
def test_task5_verify_registry_metadata(release_manager):
    """Verify target contract is strictly expected_customer_count and provenance is tracked."""
    meta_info = release_manager.verify_registry_metadata()

    assert meta_info["target_contract_verified"] is True
    assert meta_info["target"] == "expected_customer_count"
    assert meta_info["status"] == "VERIFIED"
    assert "feature_store.parquet" in meta_info["provenance"]


# ==============================================================================
# Task 6: Verify Rollback Capability
# ==============================================================================
def test_task6_verify_rollback_capability(release_manager):
    """Verify backup champion snapshot existence, loadability, and rollback readiness."""
    rb_info = release_manager.verify_rollback_capability()

    assert rb_info["backup_exists"] is True
    assert rb_info["verified_loadable"] is True
    assert rb_info["rollback_readiness"] == "READY"
    assert len(rb_info["backup_sha256"]) == 64


# ==============================================================================
# Task 7: Full End-to-End Release Manifest Generation
# ==============================================================================
def test_task7_full_release_manifest_generation(release_manager):
    """Verify end-to-end generate_release_package produces valid ReleaseManifest."""
    manifest = release_manager.generate_release_package(version="v2.1.0")

    assert isinstance(manifest, ReleaseManifest)
    assert manifest.release_version == "v2.1.0"
    assert manifest.release_tag == "v2.1.0-prod"
    assert manifest.status == "RELEASED_PRODUCTION"
    assert manifest.smoke_test_status == "ALL_PASSED"
    assert manifest.registry_verified is True
    assert manifest.rollback_verified is True

    # Verify manifest file exists and matches
    manifest_file = MODELS_DIR / "release_manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["release_version"] == "v2.1.0"
        assert data["status"] == "RELEASED_PRODUCTION"
