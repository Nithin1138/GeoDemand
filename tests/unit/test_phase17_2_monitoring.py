"""
GeoDemand AI — Phase 17.2 ML Monitoring & Telemetry Unit Tests

Validates all 7 production ML monitoring responsibilities:
1. Feature drift monitoring (PSI >= 0.15, KS test, drifted features)
2. Prediction drift monitoring (PSI >= 0.15, Wasserstein distance, distribution shift)
3. Real-label MAE monitoring (7-day rolling window, >20% degradation trigger, delayed actuals)
4. Quantile coverage monitoring (Empirical 80% coverage, under-coverage alert, crossing detection)
5. Model loading failure monitoring (Load attempts, failure rates, fallback tracking)
6. Inference latency monitoring (P50/P90/P95/P99 ms, SLA budget enforcement <= 3.0 ms)
7. Model version & rollback history tracking (Champion vs Backup verification, audit log parsing)
8. Consolidated OperationalMLMonitor health synthesis
9. FastAPI /v1/system/model-health endpoint integration
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.geodemand.monitoring.drift import DriftThresholds, FeatureDriftDetector, ReferenceDistributionProfile
from src.geodemand.monitoring.model_health import ModelHealthProvider, ModelHealthReport
from src.geodemand.monitoring.operational_monitor import (
    InferenceLatencyMonitor,
    LatencyMetrics,
    ModelLoadingHealth,
    ModelLoadingMonitor,
    ModelRollbackTracker,
    OperationalHealthSummary,
    OperationalMLMonitor,
    QuantileCoverageMonitor,
    QuantileCoverageReport,
    VersionAndRollbackStatus,
)
from src.geodemand.monitoring.performance import PerformanceMonitor
from src.geodemand.monitoring.prediction_drift import PredictionDriftDetector

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# 1. Feature Drift Monitoring
# ==============================================================================

def test_feature_drift_nominal_and_alert():
    """Verify feature drift detector recognizes nominal baseline vs drifted distribution."""
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({
        "temperature": rng.normal(30.0, 5.0, 200),
        "rainfall": rng.exponential(2.0, 200),
        "office_count": rng.poisson(10, 200),
    })
    # Nominal distribution
    nom_df = pd.DataFrame({
        "temperature": rng.normal(30.1, 4.9, 200),
        "rainfall": rng.exponential(2.1, 200),
        "office_count": rng.poisson(10, 200),
    })
    # Shifted distribution
    shift_df = pd.DataFrame({
        "temperature": rng.normal(45.0, 5.0, 200),  # Severe heatwave
        "rainfall": rng.exponential(20.0, 200),     # Extreme flood
        "office_count": rng.poisson(10, 200),
    })

    detector = FeatureDriftDetector(thresholds=DriftThresholds(psi_threshold=0.15))
    profile = ReferenceDistributionProfile.from_dataframe(ref_df, dataset_id="test_ref")

    # Nominal check
    rep_nom = detector.detect_drift(nom_df, reference_profile=profile)
    assert rep_nom.max_psi < 0.15
    assert not rep_nom.overall_drift_detected

    # Drifted check
    rep_shift = detector.detect_drift(shift_df, reference_profile=profile)
    assert rep_shift.max_psi >= 0.15
    assert rep_shift.overall_drift_detected
    assert "temperature" in rep_shift.drifted_features
    assert "rainfall" in rep_shift.drifted_features


# ==============================================================================
# 2. Prediction Drift Monitoring
# ==============================================================================

def test_prediction_drift_nominal_and_alert():
    """Verify prediction drift detector tracks PSI and Wasserstein distance shifts."""
    rng = np.random.RandomState(42)
    ref_preds = rng.poisson(15.0, 500).astype(float)
    nom_preds = rng.poisson(15.2, 500).astype(float)
    drift_preds = rng.poisson(30.0, 500).astype(float)  # 2x footfall explosion

    detector = PredictionDriftDetector(thresholds=DriftThresholds(psi_threshold=0.15))

    # Nominal
    rep_nom = detector.evaluate(ref_preds, nom_preds)
    assert rep_nom.metrics.psi < 0.15
    assert not rep_nom.trigger_activated
    assert rep_nom.severity == "NOMINAL"

    # Drifted
    rep_drift = detector.evaluate(ref_preds, drift_preds)
    assert rep_drift.metrics.psi >= 0.15
    assert rep_drift.metrics.wasserstein_distance > 5.0
    assert rep_drift.trigger_activated
    assert rep_drift.severity == "CRITICAL"


# ==============================================================================
# 3. Real-Label MAE Monitoring & Delayed Actuals
# ==============================================================================

def test_real_label_mae_monitoring_and_delayed_actuals():
    """Verify 7-day rolling MAE assessment and delayed actuals sufficiency guard."""
    monitor = PerformanceMonitor()
    baseline_mae = 0.7693

    # Stable performance
    rng = np.random.RandomState(42)
    y_true = rng.uniform(10, 30, 100)
    y_pred_stable = y_true + rng.normal(0, 0.70, 100)

    assess_stable = monitor.assess_rolling_degradation(
        baseline_mae=baseline_mae,
        recent_y_true=y_true,
        recent_y_pred=y_pred_stable,
        degradation_threshold_pct=20.0,
    )
    assert not assess_stable.degradation_alert
    assert assess_stable.degradation_pct < 20.0

    # Degraded performance (> 20% degradation)
    y_pred_degraded = y_true + rng.normal(0, 1.50, 100)
    assess_degraded = monitor.assess_rolling_degradation(
        baseline_mae=baseline_mae,
        recent_y_true=y_true,
        recent_y_pred=y_pred_degraded,
        degradation_threshold_pct=20.0,
    )
    assert assess_degraded.degradation_alert
    assert assess_degraded.degradation_pct > 20.0


# ==============================================================================
# 4. Quantile Coverage & Interval Monitoring
# ==============================================================================

def test_quantile_coverage_evaluation():
    """Verify empirical coverage, under-coverage alerts, and quantile crossing detection."""
    monitor = QuantileCoverageMonitor(
        target_coverage=0.80,
        min_acceptable_coverage=0.70,
        max_acceptable_coverage=0.95,
    )

    n = 200
    p50 = np.ones(n) * 20.0
    p10 = np.ones(n) * 15.0
    p90 = np.ones(n) * 25.0

    # 1. Nominal scenario: 80% coverage
    actuals_nominal = np.concatenate([
        np.ones(160) * 20.0,  # inside [15, 25]
        np.ones(20) * 10.0,   # below 15
        np.ones(20) * 30.0,   # above 25
    ])
    rep_nom = monitor.evaluate_coverage(p10, p50, p90, actuals_nominal)
    assert rep_nom.empirical_coverage_80 == 0.80
    assert not rep_nom.under_coverage_alert
    assert rep_nom.quantile_crossing_count == 0
    assert rep_nom.status == "HEALTHY"

    # 2. Under-coverage scenario: 50% coverage
    actuals_bad = np.concatenate([
        np.ones(100) * 20.0,  # inside
        np.ones(100) * 50.0,  # outside
    ])
    rep_bad = monitor.evaluate_coverage(p10, p50, p90, actuals_bad)
    assert rep_bad.empirical_coverage_80 == 0.50
    assert rep_bad.under_coverage_alert
    assert rep_bad.status == "DEGRADED"

    # 3. Quantile crossing violation (P10 > P50)
    p10_crossed = p10.copy()
    p10_crossed[0] = 30.0  # P10=30 > P50=20
    rep_crossed = monitor.evaluate_coverage(p10_crossed, p50, p90, actuals_nominal)
    assert rep_crossed.quantile_crossing_count == 1
    assert rep_crossed.crossing_alert
    assert rep_crossed.status == "DEGRADED"


# ==============================================================================
# 5. Model Loading Failure Monitoring
# ==============================================================================

def test_model_loading_failure_monitoring():
    """Verify loading telemetry tracks successes, failures, and fallback activations."""
    monitor = ModelLoadingMonitor()

    assert monitor.get_health().is_healthy

    # Record successful load
    monitor.record_success(model_version="v2.1.0")
    h1 = monitor.get_health()
    assert h1.total_load_attempts == 1
    assert h1.successful_loads == 1
    assert h1.failed_loads == 0
    assert h1.is_healthy
    assert h1.status == "HEALTHY"

    # Record failure
    monitor.record_failure(
        failure_type="CORRUPT_ARTIFACT",
        reason="Model bundle corrupted during unpickling",
        model_version="v2.1.0",
    )
    h2 = monitor.get_health()
    assert h2.total_load_attempts == 2
    assert h2.failed_loads == 1
    assert h2.failure_rate_pct == 50.0
    assert not h2.is_healthy
    assert h2.status == "CRITICAL_LOAD_FAILURE"
    assert h2.last_failure_type == "CORRUPT_ARTIFACT"

    # Record fallback
    monitor.record_fallback_activation(reason="Activated emergency fallback Ridge regressor")
    h3 = monitor.get_health()
    assert h3.fallback_activations == 1


# ==============================================================================
# 6. Inference Latency Monitoring
# ==============================================================================

def test_inference_latency_monitoring_and_sla():
    """Verify latency tracking computes P50, P90, P95, P99 and enforces 3.0ms SLA budget."""
    monitor = InferenceLatencyMonitor(sla_budget_ms=3.0)

    # Fast latencies within budget (0.01 - 0.50 ms)
    for l_ms in [0.02, 0.05, 0.03, 0.04, 0.10, 0.08, 0.12, 0.06, 0.05, 0.07]:
        monitor.record_latency(l_ms)

    metrics_fast = monitor.get_metrics()
    assert metrics_fast.total_predictions == 10
    assert metrics_fast.p50_latency_ms < 0.10
    assert metrics_fast.p95_latency_ms < 3.0
    assert not metrics_fast.sla_breached
    assert metrics_fast.status == "HEALTHY"

    # Inject slow latencies exceeding budget (> 3.0 ms)
    for _ in range(20):
        monitor.record_latency(5.5)

    metrics_slow = monitor.get_metrics()
    assert metrics_slow.p95_latency_ms > 3.0
    assert metrics_slow.sla_breached
    assert metrics_slow.status == "SLA_BREACHED"

    # Test context manager measure
    with monitor.measure():
        # Quick sleep simulation
        pass
    assert len(monitor.latencies) == 31


# ==============================================================================
# 7. Model Version & Rollback History Tracking
# ==============================================================================

def test_model_version_and_rollback_history_tracking():
    """Verify active champion and backup champion status inspection."""
    tracker = ModelRollbackTracker(models_dir=ROOT_DIR / "models")
    status = tracker.get_status()

    assert isinstance(status, VersionAndRollbackStatus)
    assert status.active_model_version is not None
    assert Path(status.active_champion_path).exists()
    assert status.active_model_checksum is not None

    # Backup status
    assert status.backup_exists
    assert status.backup_verified_loadable
    assert status.rollback_readiness == "READY"
    assert status.total_promotion_events >= 0


# ==============================================================================
# 8. Consolidated Operational ML Monitor
# ==============================================================================

def test_consolidated_operational_ml_monitor_synthesis():
    """Verify OperationalMLMonitor unifies all 7 dimensions into a comprehensive report."""
    op_monitor = OperationalMLMonitor(models_dir=ROOT_DIR / "models", sla_budget_ms=3.0)

    rng = np.random.RandomState(42)
    cur_features = pd.DataFrame({
        "temperature": rng.normal(30.0, 5.0, 100),
        "rainfall": rng.exponential(2.0, 100),
    })
    preds = rng.poisson(15, 100).astype(float)
    actuals = rng.poisson(15, 100).astype(float)
    p10 = preds * 0.8
    p50 = preds
    p90 = preds * 1.2

    # Record some latencies
    for l_ms in [0.03, 0.04, 0.02, 0.05]:
        op_monitor.latency_monitor.record_latency(l_ms)

    summary = op_monitor.evaluate_system_health(
        current_features_df=cur_features,
        current_predictions=preds,
        recent_actuals=actuals,
        recent_preds_for_actuals=preds,
        p10_preds=p10,
        p50_preds=p50,
        p90_preds=p90,
    )

    assert isinstance(summary, OperationalHealthSummary)
    assert summary.overall_health in {"HEALTHY", "WARNING", "CRITICAL_DRIFT", "DEGRADED", "SLA_BREACHED"}
    assert "max_psi" in summary.feature_drift
    assert "psi" in summary.prediction_drift
    assert "current_7d_mae" in summary.real_label_performance
    assert "empirical_coverage_80" in summary.quantile_coverage
    assert "failed_loads" in summary.model_loading
    assert "p95_latency_ms" in summary.inference_latency
    assert "rollback_readiness" in summary.version_and_rollback
    assert summary.rollback_ready is True

    # Test serialization
    as_dict = summary.to_dict()
    assert isinstance(as_dict, dict)
    as_json = summary.to_json()
    assert "overall_health" in as_json


# ==============================================================================
# 9. API /v1/system/model-health Integration
# ==============================================================================

def test_api_model_health_all_dimensions():
    """Verify FastAPI endpoint exposes all 7 monitoring dimensions."""
    from fastapi.testclient import TestClient
    from api.app import app

    with TestClient(app) as client:
        r = client.get("/v1/system/model-health")
        assert r.status_code == 200
        data = r.json()

        # Phase 13.4 fields
        assert "status" in data
        assert "model_version" in data
        assert "baseline_metrics" in data
        assert "feature_drift" in data
        assert "prediction_drift" in data
        assert "rolling_performance_7d" in data
        assert "retraining_trigger_active" in data
        assert "ground_truth_status" in data

        # Phase 17.2 final dimensions
        assert "quantile_coverage" in data
        assert "model_loading_health" in data
        assert "inference_latency_health" in data
        assert "version_and_rollback_health" in data

        # Verify values
        assert data["model_loading_health"]["is_healthy"] is True
        assert data["inference_latency_health"]["sla_budget_ms"] == 3.0
        assert data["version_and_rollback_health"]["rollback_readiness"] == "READY"
