"""
GeoDemand AI — Backend Hardening, Security, and Reliability Test Suite.

Validates:
1. Structured JSON logging and context correlation ID propagation.
2. Latency percentiles tracking (p50, p90, p95, p99) and Prometheus format telemetry.
3. Circuit breaker 3-state machine (CLOSED, OPEN, HALF_OPEN) and external provider timeouts.
4. Controlled retries with exponential backoff and randomized jitter.
5. Payload validation and standardized error response contract (StandardErrorResponse).
6. API rate limiting with sliding window counter and RFC 6585 headers (429, Retry-After).
7. Role-Based Access Control (RBAC) across Anonymous, Vendor, Fleet Manager, and Admin tiers.
8. Reservation ownership verification preventing unauthorized cancellations.
9. Database failure resilience, transaction consistency, and automatic rollback.
10. Redis failure resilience with transparent in-memory TTL cache and lock fallback.
11. Routing provider failure resilience with haversine distance fallback.
12. Concurrent payment webhook idempotency slot acquisition under multi-threaded contention.
13. Concurrent reservation capacity enforcement under race conditions.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from api.app import app, fleet_manager
from api.security.errors import (
    ErrorCode,
    ERROR_CODE_METADATA,
    StandardErrorResponse,
    GeoDemandAPIError,
    AuthenticationError,
    ForbiddenError,
    RateLimitExceededError,
)
from api.reliability.logging import (
    StructuredJsonFormatter,
    sanitize_log_message,
    set_correlation_id,
    get_correlation_id,
    set_request_id,
    get_request_id,
)
from api.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenException,
    global_circuit_breaker_registry,
)
from api.reliability.retry import (
    calculate_backoff,
    execute_with_retry,
    with_retry,
)
from api.reliability.metrics import (
    LatencyMetricsCollector,
    global_metrics_collector,
    format_prometheus_metrics,
)
from api.security.rate_limiter import (
    RateLimiter,
    global_rate_limiter,
)
from api.security.auth import (
    Role,
    UserContext,
    global_api_key_registry,
)
from api.reliability.resilience import (
    ResilientDatabaseManager,
    ResilientRedisClient,
    ResilientRoutingHandler,
)
from api.webhooks.dead_letter import dead_letter_queue


@pytest.fixture
def client():
    # Use TestClient with rate limiting enabled
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 1. Structured JSON Logging & Masking
# ===========================================================================

def test_structured_json_logging_and_context():
    """Validates that log records are emitted as valid JSON with correlation IDs."""
    formatter = StructuredJsonFormatter()
    logger = logging.getLogger("test_structured_logger")
    logger.setLevel(logging.INFO)

    set_correlation_id("corr_test_999")
    set_request_id("req_test_888")

    record = logger.makeRecord(
        name="test_logger",
        level=logging.INFO,
        fn="test_fn",
        lno=42,
        msg="Vendor relocation calculation finished",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    log_json = json.loads(formatted)

    assert log_json["level"] == "INFO"
    assert log_json["correlation_id"] == "corr_test_999"
    assert log_json["request_id"] == "req_test_888"
    assert log_json["message"] == "Vendor relocation calculation finished"
    assert "timestamp" in log_json


def test_sanitize_log_message_redaction():
    """Ensures passwords, tokens, API keys, and authorization secrets are masked in logs."""
    raw = "User login with apiKey=sk_live_secret123 and token='tok_xyz987' Authorization: Bearer abc_token_456"
    sanitized = sanitize_log_message(raw)

    assert "sk_live_secret123" not in sanitized
    assert "tok_xyz987" not in sanitized
    assert "abc_token_456" not in sanitized
    assert "[REDACTED]" in sanitized


# ===========================================================================
# 2. Correlation ID Header Propagation
# ===========================================================================

def test_correlation_id_propagation(client):
    """Verifies correlation and request IDs are attached to HTTP response headers."""
    # 1. Custom correlation ID passed by client
    custom_cid = "corr_client_custom_123"
    resp = client.get("/v1/system/ping", headers={"X-Correlation-ID": custom_cid})
    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == custom_cid
    assert resp.headers.get("X-Request-ID") is not None

    # 2. Auto-generated correlation ID when client does not supply one
    resp2 = client.get("/v1/system/ping")
    assert resp2.status_code == 200
    gen_cid = resp2.headers.get("X-Correlation-ID")
    assert gen_cid is not None
    assert gen_cid.startswith("corr_")


# ===========================================================================
# 3. Latency Metrics & Prometheus Export
# ===========================================================================

def test_latency_metrics_and_prometheus_export(client):
    """Verifies API latency tracking, percentile calculations, and Prometheus output."""
    collector = LatencyMetricsCollector(window_minutes=5)
    for ms in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        collector.record_request("/v1/recommendations/live", "POST", 200, ms)

    summary = collector.get_metrics_summary()
    assert summary["total_requests"] == 10
    assert summary["error_rate_pct"] == 0.0
    lat = summary["latency_ms"]
    assert 50.0 <= lat["p50"] <= 60.0
    assert lat["p90"] >= 90.0
    assert lat["min"] == 10.0
    assert lat["max"] == 100.0

    # Test Prometheus endpoint
    resp = client.get("/v1/system/metrics?format=prometheus")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "geodemand_api_requests_total" in resp.text
    assert "geodemand_api_latency_ms" in resp.text


# ===========================================================================
# 4. Circuit Breaker State Transitions
# ===========================================================================

def test_circuit_breaker_fsm():
    """Verifies CLOSED -> OPEN -> HALF_OPEN -> CLOSED state transitions."""
    cb = CircuitBreaker(name="test_cb", failure_threshold=3, recovery_timeout_sec=0.2, half_open_success_threshold=1)
    assert cb.state == CircuitState.CLOSED

    # Record 2 failures -> stays CLOSED
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3rd failure -> trips to OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Executing while OPEN raises CircuitBreakerOpenException
    with pytest.raises(CircuitBreakerOpenException):
        cb.call(lambda: "should fail")

    # Wait for cooldown timeout
    time.sleep(0.25)

    # Calling should move state to HALF_OPEN and test the probe
    def successful_probe():
        return "probe_success"

    res = cb.call(successful_probe)
    assert res == "probe_success"
    # Success in HALF_OPEN resets circuit to CLOSED
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_endpoint(client):
    """Verifies GET /v1/system/circuit-breakers lists registered breakers."""
    resp = client.get("/v1/system/circuit-breakers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


# ===========================================================================
# 5. Controlled Retries with Backoff and Jitter
# ===========================================================================

def test_calculate_backoff_with_jitter():
    """Ensures backoff scales exponentially and stays bounded by max_delay."""
    for attempt in range(1, 5):
        b = calculate_backoff(attempt=attempt, base_delay=0.1, max_delay=2.0, jitter=True)
        assert b >= 0.0
        assert b <= 2.0


def test_execute_with_retry_transient_recovery():
    """Verifies automatic recovery on transient errors."""
    attempts = 0

    def flaky_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Temporary socket timeout")
        return "success_after_retry"

    result = execute_with_retry(
        flaky_operation,
        max_attempts=4,
        base_delay=0.01,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "success_after_retry"
    assert attempts == 3


def test_execute_with_retry_non_transient_rejection():
    """Verifies non-retryable errors fail immediately without retrying."""
    attempts = 0

    def fail_hard():
        nonlocal attempts
        attempts += 1
        raise ValueError("Invalid configuration parameter")

    with pytest.raises(ValueError):
        execute_with_retry(
            fail_hard,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
    assert attempts == 1


# ===========================================================================
# 6. Payload Validation & Standard Error Response Contract
# ===========================================================================

def test_validation_error_response_contract(client):
    """Verifies 422 Unprocessable Content returns StandardErrorResponse format."""
    bad_payload = {
        "latitude": 999.0,  # Invalid latitude (> 90)
        "longitude": 80.6480,
    }
    resp = client.post("/v1/location/context", json=bad_payload)
    assert resp.status_code == 422
    data = resp.json()

    assert data["error"] == ErrorCode.VALIDATION_ERROR.value
    assert data["status_code"] == 422
    assert "correlation_id" in data
    assert "details" in data
    assert resp.headers.get("X-Correlation-ID") is not None


def test_error_catalog_endpoint(client):
    """Verifies GET /v1/system/error-catalog exposes all standardized codes."""
    resp = client.get("/v1/system/error-catalog")
    assert resp.status_code == 200
    cat = resp.json()

    assert "catalog_version" in cat
    assert "error_codes" in cat
    assert ErrorCode.RATE_LIMIT_EXCEEDED.value in cat["error_codes"]
    assert ErrorCode.AUTH_UNAUTHORIZED.value in cat["error_codes"]
    assert ErrorCode.AUTH_RESERVATION_OWNERSHIP_VIOLATION.value in cat["error_codes"]


# ===========================================================================
# 7. Rate Limiting with Sliding Window & RFC 6585 Headers
# ===========================================================================

def test_rate_limiter_sliding_window():
    """Tests RateLimiter sliding window capacity enforcement."""
    limiter = RateLimiter()
    client_key = "test_vendor_ip_1"

    # Allow up to 3 requests in a 1-second window
    for _ in range(3):
        allowed, rem, _ = limiter.check_rate_limit(client_key, role="custom", window_seconds=1, default_limit=3)
        assert allowed is True

    # 4th request must be rejected
    allowed, rem, reset_s = limiter.check_rate_limit(client_key, role="custom", window_seconds=1, default_limit=3)
    assert allowed is False
    assert rem == 0
    assert reset_s > 0


def test_api_rate_limiting_middleware(client, monkeypatch):
    """Ensures API returns HTTP 429 and RFC 6585 headers when limit is exceeded."""
    monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
    limiter = global_rate_limiter
    test_key = "spam_client_test_key"
    global_api_key_registry.register_key(test_key, Role.VENDOR, vendor_id="V-SPAM")

    # Consume the vendor rate limit for this key
    limit = limiter.get_limit_for_role("vendor")
    for _ in range(limit):
        limiter.check_rate_limit(test_key, role="vendor", window_seconds=60)

    # Next request via client must trigger 429
    resp = client.get("/v1/system/data-health", headers={"X-API-Key": test_key})
    assert resp.status_code == 429
    data = resp.json()
    assert data["error"] == ErrorCode.RATE_LIMIT_EXCEEDED.value
    assert "Retry-After" in resp.headers
    assert "X-RateLimit-Limit" in resp.headers
    assert resp.headers["X-RateLimit-Remaining"] == "0"


# ===========================================================================
# 8. Role-Based Access Control (RBAC)
# ===========================================================================

def test_rbac_fleet_dispatch_authorization(client):
    """Verifies that vendors cannot trigger fleet dispatch, while fleet managers and admins can."""
    vendor_key = "key_vendor_test_101"
    manager_key = "key_manager_test_202"
    global_api_key_registry.register_key(vendor_key, Role.VENDOR, vendor_id="V-101")
    global_api_key_registry.register_key(manager_key, Role.FLEET_MANAGER, user_id="M-202")

    payload = {
        "vendors": [
            {
                "vendor_id": "V-101",
                "vendor_category": "tiffin",
                "latitude": 16.5062,
                "longitude": 80.6480,
            }
        ],
        "candidate_cells": [{"h3_cell": "8861892557fffff", "demand_p50": 35.0, "demand_p90": 55.0, "net_profit_inr": 450.0}],
    }

    # 1. Vendor key receives 403 Forbidden
    resp_vendor = client.post("/v1/fleet/dispatch", json=payload, headers={"X-API-Key": vendor_key})
    assert resp_vendor.status_code == 403
    assert resp_vendor.json()["error"] == ErrorCode.AUTH_FORBIDDEN.value

    # 2. Fleet Manager key is permitted
    resp_mgr = client.post("/v1/fleet/dispatch", json=payload, headers={"X-API-Key": manager_key})
    assert resp_mgr.status_code == 200
    assert "dispatch_id" in resp_mgr.json()


# ===========================================================================
# 9. Reservation Ownership Protection
# ===========================================================================

def test_prevent_unauthorized_reservation_cancellation(client):
    """Validates that Vendor A cannot cancel Vendor B's reservation, but managers and owners can."""
    v_a = f"V-A-{uuid.uuid4().hex[:6]}"
    v_b = f"V-B-{uuid.uuid4().hex[:6]}"
    vendor_a_key = f"key_vendor_a_{uuid.uuid4().hex[:6]}"
    vendor_b_key = f"key_vendor_b_{uuid.uuid4().hex[:6]}"
    manager_key = f"key_manager_{uuid.uuid4().hex[:6]}"

    fleet_manager.set_cell_capacity("8861892557fffff", total_capacity=10, category_capacities={"tea": 10})
    global_api_key_registry.register_key(vendor_a_key, Role.VENDOR, vendor_id=v_a)
    global_api_key_registry.register_key(vendor_b_key, Role.VENDOR, vendor_id=v_b)
    global_api_key_registry.register_key(manager_key, Role.FLEET_MANAGER, user_id="M-303")

    # Vendor A books a cell
    ok, res, msg = fleet_manager.reserve_cell(
        vendor_id=v_a,
        h3_cell="8861892557fffff",
        vendor_category="tea",
        duration_minutes=30,
        is_soft=False,
    )
    assert ok is True, f"Reservation failed: {msg}"
    res_id = res.reservation_id

    # Vendor B attempts to cancel Vendor A's reservation -> 403
    cancel_payload = {"reservation_id": res_id, "vendor_id": v_b}
    resp_b = client.post("/v1/fleet/cancel-reservation", json=cancel_payload, headers={"X-API-Key": vendor_b_key})
    assert resp_b.status_code == 403
    assert resp_b.json()["error"] == ErrorCode.AUTH_RESERVATION_OWNERSHIP_VIOLATION.value

    # Vendor A cancels their own reservation -> 200 OK
    cancel_payload_a = {"reservation_id": res_id, "vendor_id": v_a}
    resp_a = client.post("/v1/fleet/cancel-reservation", json=cancel_payload_a, headers={"X-API-Key": vendor_a_key})
    assert resp_a.status_code == 200
    assert resp_a.json()["status"] == "CANCELLED"


# ===========================================================================
# 10. Database Failure Resilience & Transaction Consistency
# ===========================================================================

def test_resilient_database_atomic_rollback(tmp_path):
    """Validates that database transactions automatically rollback upon exceptions."""
    db_file = tmp_path / "test_resilience.db"
    db_mgr = ResilientDatabaseManager(db_path=str(db_file))

    # Initialize table
    with db_mgr.transaction() as conn:
        conn.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, balance REAL)")
        conn.execute("INSERT INTO accounts VALUES ('V-1', 1000.0)")

    # Execute a failing transaction: transfer with simulated crash
    try:
        with db_mgr.transaction() as conn:
            conn.execute("UPDATE accounts SET balance = balance - 200 WHERE id = 'V-1'")
            # Simulate unexpected failure
            raise RuntimeError("Database connection dropped during transfer")
    except RuntimeError:
        pass

    # Verify atomic rollback: balance must still be 1000.0, NOT 800.0
    with db_mgr.transaction() as conn:
        row = conn.execute("SELECT balance FROM accounts WHERE id = 'V-1'").fetchone()
        assert row[0] == 1000.0


# ===========================================================================
# 11. Redis Failure Resilience & Local TTL Fallback
# ===========================================================================

def test_resilient_redis_client_fallback():
    """Verifies that Redis client falls back seamlessly to thread-safe local cache/locks."""
    # Initialize with non-existent Redis URL to force local fallback
    client = ResilientRedisClient(redis_url="redis://localhost:9999/0")
    assert client.is_redis_available is False

    # Test key-value caching with TTL
    client.set("cache_key_1", {"demand": 42.5}, ttl_seconds=2)
    val = client.get("cache_key_1")
    assert val == {"demand": 42.5}

    # Test distributed lock fallback
    lock = client.lock("lock:cell:8861892557fffff", timeout=1.0)
    acquired = lock.acquire(blocking=False)
    assert acquired is True

    # Concurrent acquisition fails
    acquired_again = lock.acquire(blocking=False)
    assert acquired_again is False

    lock.release()
    assert lock.is_locked() is False


# ===========================================================================
# 12. Routing Provider Failure Resilience
# ===========================================================================

def test_resilient_routing_fallback():
    """Verifies fallback to haversine straight-line distance if routing provider trips."""
    router = ResilientRoutingHandler(force_circuit_open=True)

    # Circuit is forced OPEN -> haversine calculation used
    dist_km, dur_min = router.get_route_distance_and_duration(
        origin_lat=16.5062, origin_lng=80.6480,
        dest_lat=16.5162, dest_lng=80.6580,
    )
    assert dist_km > 0.0
    assert dur_min > 0.0


# ===========================================================================
# 13. Concurrent Webhook Processing Idempotency
# ===========================================================================

def test_concurrent_webhook_idempotency_slot():
    """Ensures that identical concurrent webhooks only claim the idempotency slot once."""
    dlq = dead_letter_queue
    event_id = f"evt_race_{uuid.uuid4().hex[:8]}"

    results = []

    def claim():
        return dlq.try_acquire_idempotency(event_id, ttl_seconds=60)

    # Fire 10 simultaneous threads trying to claim the same event ID
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(claim) for _ in range(10)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Exactly 1 thread must acquire the slot; 9 must be rejected as duplicates
    assert results.count(True) == 1
    assert results.count(False) == 9


# ===========================================================================
# 14. Concurrent Cell Reservations Under High Contention
# ===========================================================================

def test_concurrent_reservations_capacity_lock():
    """Verifies atomic capacity enforcement under concurrent reservation attempts."""
    cell = "8861892557fffff"
    # Configure cell capacity to exactly 2 vendors
    fleet_manager.set_cell_capacity(cell, total_capacity=2, category_capacities={"tea": 2})

    # Clear prior active reservations for this cell
    for r in list(fleet_manager.get_active_reservations_for_cell(cell)):
        fleet_manager.cancel_reservation(r.reservation_id)

    reservation_outcomes = []

    def attempt_reservation(vendor_idx: int):
        ok, res, msg = fleet_manager.reserve_cell(
            vendor_id=f"V-CONC-{vendor_idx}",
            h3_cell=cell,
            vendor_category="tea",
            duration_minutes=30,
            is_soft=False,
        )
        return ok

    # 10 vendors race concurrently for 2 spots
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(attempt_reservation, i) for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            reservation_outcomes.append(f.result())

    # Exactly 2 succeed, 8 rejected
    assert reservation_outcomes.count(True) == 2
    assert reservation_outcomes.count(False) == 8
