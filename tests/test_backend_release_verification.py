"""
GeoDemand AI — Phase 19.3 Backend Release Verification Test Suite

Member 3 — Backend, Business Economics & Decision Intelligence

Validates all 8 Phase 19.3 Backend Release verification criteria:
1. Freeze API contracts (v2.1.0-prod immutable manifest and artifact SHA-256 checksums)
2. Verify authentication (timing-safe HMAC, API key registry, and role enforcement)
3. Verify database migrations (atomic commit, transaction rollback on failure)
4. Verify provider credentials (credential masking and secret redaction)
5. Verify payment integrations (Razorpay and PhonePe signature verification)
6. Verify reservation consistency (concurrency safety and zero double-booking)
7. Run production smoke tests (all V1 endpoints executed with SLA latency assertions)
8. Verify API health and error handling (standardized error envelopes and fallback responses)
"""

from pathlib import Path
import pytest
from src.api.release_freeze import BackendReleaseManager, BackendReleaseManifest

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def release_manager():
    return BackendReleaseManager(root_dir=ROOT_DIR)


def test_freeze_api_contracts(release_manager):
    """Task 1: Verify API contract extraction, route catalogue, and immutable checksums."""
    checksums, endpoints, openapi = release_manager.freeze_api_contracts()

    assert len(endpoints) >= 25
    assert len(checksums) >= 8

    # Verify key source file checksums exist and are non-empty
    for src_name, sha in checksums.items():
        assert sha != "MISSING", f"Source file {src_name} checksum missing"
        assert len(sha) == 64

    # Verify critical routes in endpoint catalog
    paths = {ep["path"] for ep in endpoints}
    assert "/v1/system/ping" in paths
    assert "/v1/predict/demand" in paths
    assert "/v1/recommend/zones" in paths
    assert "/v1/recommendations/live" in paths
    assert "/v1/scenario/simulate" in paths
    assert "/v1/fleet/reserve-cell" in paths
    assert "/v1/fleet/dispatch" in paths


def test_verify_authentication(release_manager):
    """Task 2: Verify timing-safe API key comparison and role-based access control."""
    res = release_manager.verify_authentication()
    assert res["status"] == "VERIFIED"
    assert res["timing_safe_digest_verified"] is True
    assert "ADMIN" in res["roles_configured"]
    assert "FLEET_MANAGER" in res["roles_configured"]
    assert "VENDOR" in res["roles_configured"]
    assert "ANONYMOUS" in res["roles_configured"]


def test_verify_database_migrations(release_manager):
    """Task 3: Verify atomic transaction commit and immediate rollback on failure."""
    res = release_manager.verify_database_migrations()
    assert res["status"] == "VERIFIED"
    assert res["atomic_commit_verified"] is True
    assert res["atomic_rollback_verified"] is True
    assert res["transaction_isolation"] == "SERIALIZABLE"


def test_verify_provider_credentials(release_manager):
    """Task 4: Verify credential masking and secret redaction."""
    res = release_manager.verify_provider_credentials()
    assert res["status"] == "VERIFIED"
    assert res["credential_masking_active"] is True
    assert res["secret_redaction_verified"] is True


def test_verify_payment_integrations(release_manager):
    """Task 5: Verify signature verification across payment webhook adapters."""
    res = release_manager.verify_payment_integrations()
    assert res["status"] == "VERIFIED"
    assert res["razorpay_signature_verifier"] == "ACTIVE"
    assert res["phonepe_signature_verifier"] == "ACTIVE"


def test_verify_reservation_consistency(release_manager):
    """Task 6: Verify soft reservation expiry, capacity limits, and race condition locks."""
    res = release_manager.verify_reservation_consistency()
    assert res["status"] == "VERIFIED"
    assert res["concurrency_lock_verified"] is True
    assert res["max_capacity_enforced"] is True
    assert res["successful_under_race"] <= 2


def test_run_production_smoke_tests(release_manager):
    """Task 7: Execute full production smoke tests across all main V1 routes."""
    res = release_manager.run_production_smoke_tests()
    assert res["all_passed"] is True
    assert res["total_tested"] >= 10
    for r in res["results"]:
        assert r["passed"] is True, f"Failed smoke test: {r['method']} {r['path']} -> {r['status_code']}"


def test_verify_api_health_and_error_handling(release_manager):
    """Task 8: Verify standardized error response envelopes on invalid inputs."""
    res = release_manager.verify_api_health_and_error_handling()
    assert res["status"] == "VERIFIED"
    assert res["validation_errors_enveloped"] is True
    assert res["not_found_handled"] is True


def test_full_release_manifest_generation(release_manager):
    """Verify end-to-end generate_release_package produces valid BackendReleaseManifest."""
    manifest = release_manager.generate_release_package(version="v2.1.0")

    assert isinstance(manifest, BackendReleaseManifest)
    assert manifest.version == "v2.1.0"
    assert manifest.release_tag == "v2.1.0-prod"
    assert manifest.is_production_ready is True
    assert manifest.total_endpoints >= 25

    # Verify manifest saved on disk
    manifest_path = ROOT_DIR / "models" / "backend_api_release_manifest.json"
    assert manifest_path.exists()
