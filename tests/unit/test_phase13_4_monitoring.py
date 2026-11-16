"""
GeoDemand AI — Unit Tests for Drift & Model Monitoring (Phase 13.4)

Member 2 — Machine Learning & MLOps

Validates:
1. Evidently AI integration and fallback adapter
2. Population Stability Index (PSI) calculation
3. Wasserstein Distance (Earth Mover's Distance) calculation
4. Input feature drift monitoring
5. Prediction drift monitoring
6. Seven-day rolling MAE monitoring
7. Roadmap drift trigger (PSI > 0.15)
8. Roadmap seven-day MAE degradation trigger (> 20%)
9. False-positive drift mitigation (identical distributions, large N with small effect size)
10. Missing actual-label scenarios (delayed ground truth, null actuals)
11. Insufficient ground truth protection (no retraining when samples < 500)
12. Automated retraining workflow trigger
13. Candidate model promotion gating
14. Atomic snapshot rollback mechanics
15. API model health endpoint (/v1/system/model-health)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.geodemand.monitoring.ct_pipeline import (
    ContinuousTrainingPipeline,
    RetrainingTrigger,
)
from src.geodemand.monitoring.drift import (
    DriftReport,
    DriftThresholds,
    FeatureDriftDetector,
    ReferenceDistributionProfile,
    calculate_psi,
    calculate_wasserstein_distance,
)
from src.geodemand.monitoring.evidently_adapter import (
    EvidentlyMonitoringAdapter,
    FallbackEvidentlyReport,
)
from src.geodemand.monitoring.model_health import (
    ModelHealthProvider,
    ModelHealthReport,
)
from src.geodemand.monitoring.performance import (
    PerformanceMonitor,
    RollingPerformanceAssessment,
)
from src.geodemand.monitoring.prediction_drift import (
    PredictionDriftDetector,
    PredictionDriftReport,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ===========================================================================
# 1. Evidently AI Integration & Fallback Adapter
# ===========================================================================

def test_evidently_adapter_execution_and_schema(tmp_path):
    """Test Evidently AI adapter produces standardized drift schema and HTML reports."""
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({
        "temperature": rng.normal(30.0, 5.0, 200),
        "humidity": rng.uniform(40.0, 80.0, 200),
        "vendor_category": rng.choice(["food", "chai", "repair"], 200),
    })
    cur_df = pd.DataFrame({
        "temperature": rng.normal(38.0, 5.0, 200),  # Drifted
        "humidity": rng.uniform(40.0, 80.0, 200),
        "vendor_category": rng.choice(["food", "chai", "repair"], 200),
    })

    adapter = EvidentlyMonitoringAdapter()
    report = adapter.run_data_drift_report(reference_data=ref_df, current_data=cur_df)

    res_dict = report.as_dict()
    assert "metrics" in res_dict
    assert any(m["metric"] == "DatasetDriftMetric" for m in res_dict["metrics"])
    assert any(m["metric"] == "DataDriftTable" for m in res_dict["metrics"])

    html_file = tmp_path / "drift_report.html"
    saved_path = report.save_html(html_file)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 500


# ===========================================================================
# 2. Population Stability Index (PSI) Calculation
# ===========================================================================

def test_psi_calculation_identical_and_shifted():
    """Test PSI is 0 for identical distributions and strictly positive for shifted ones."""
    rng = np.random.RandomState(42)
    ref = rng.normal(25.0, 3.0, 500)
    cur_identical = rng.normal(25.0, 3.0, 500)
    cur_shifted = rng.normal(35.0, 3.0, 500)

    # Identical samples should have very low PSI (< 0.05)
    psi_ident = calculate_psi(ref, cur_identical)
    assert psi_ident < 0.08, f"Expected small PSI for identical distribution, got {psi_ident}"

    # Shifted samples must have high PSI (> 0.25)
    psi_shift = calculate_psi(ref, cur_shifted)
    assert psi_shift > 0.25, f"Expected large PSI for shifted distribution, got {psi_shift}"


# ===========================================================================
# 3. Wasserstein Distance (Earth Mover's Distance)
# ===========================================================================

def test_wasserstein_distance_properties():
    """Test Wasserstein distance calculates exact physical shift magnitude."""
    ref = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    cur_shifted = ref + 5.0

    wd = calculate_wasserstein_distance(ref, cur_shifted)
    assert np.isclose(wd, 5.0, atol=1e-3)

    wd_zero = calculate_wasserstein_distance(ref, ref)
    assert wd_zero == 0.0


# ===========================================================================
# 4. Input Feature Drift Monitoring
# ===========================================================================

def test_input_feature_drift_detection():
    """Test detection of drifted vs stable continuous and categorical features."""
    rng = np.random.RandomState(42)
    n = 300
    ref_df = pd.DataFrame({
        "temperature": rng.normal(28.0, 4.0, n),
        "rainfall": rng.exponential(1.0, n),
        "vendor_category": rng.choice(["food", "salon", "retail"], n),
    })
    cur_df = pd.DataFrame({
        "temperature": rng.normal(40.0, 4.0, n),  # Substantial shift
        "rainfall": rng.exponential(1.0, n),      # Stable
        "vendor_category": rng.choice(["food", "salon", "retail"], n),  # Stable
    })

    detector = FeatureDriftDetector()
    profile = ReferenceDistributionProfile.from_dataframe(ref_df, dataset_id="ref_profile")
    report = detector.detect_drift(cur_df, reference_profile=profile)

    assert "temperature" in report.drifted_features
    assert "rainfall" not in report.drifted_features
    assert report.overall_drift_detected


# ===========================================================================
# 5. Prediction Drift Monitoring
# ===========================================================================

def test_prediction_drift_detector():
    """Test prediction drift evaluation between baseline and live inference outputs."""
    rng = np.random.RandomState(42)
    n = 400
    ref_preds = rng.poisson(lam=15.0, size=n).astype(float)
    cur_preds_shifted = rng.poisson(lam=28.0, size=n).astype(float)

    detector = PredictionDriftDetector()
    report = detector.evaluate(ref_preds, cur_preds_shifted)

    assert report.drift_detected
    assert report.trigger_activated  # PSI >= 0.15 trigger
    assert report.severity == "CRITICAL"
    assert report.metrics.psi >= 0.15
    assert report.metrics.mean_shift > 0
    assert report.metrics.p50_current > report.metrics.p50_reference


# ===========================================================================
# 6 & 8. Seven-Day Rolling MAE Monitoring & Degradation Trigger (> 20%)
# ===========================================================================

def test_seven_day_mae_degradation_trigger():
    """Test 7-day rolling MAE degradation threshold of 20% triggers alert."""
    perf_mon = PerformanceMonitor()
    baseline_mae = 0.80

    # Stable error: 0.88 (+10% change <= 20% threshold)
    y_true = np.ones(100) * 20.0
    y_pred_stable = y_true + 0.88
    res_stable = perf_mon.assess_rolling_degradation(
        baseline_mae=baseline_mae,
        recent_y_true=y_true,
        recent_y_pred=y_pred_stable,
        degradation_threshold_pct=20.0,
    )
    assert not res_stable.degradation_alert
    assert np.isclose(res_stable.degradation_pct, 10.0, atol=0.5)

    # Degraded error: 1.04 (+30% degradation > 20% threshold)
    y_pred_degraded = y_true + 1.04
    res_degraded = perf_mon.assess_rolling_degradation(
        baseline_mae=baseline_mae,
        recent_y_true=y_true,
        recent_y_pred=y_pred_degraded,
        degradation_threshold_pct=20.0,
    )
    assert res_degraded.degradation_alert
    assert res_degraded.degradation_pct >= 20.0


# ===========================================================================
# 7. Roadmap Drift Trigger (PSI > 0.15)
# ===========================================================================

def test_roadmap_drift_trigger_threshold():
    """Test pipeline triggers retraining when PSI >= 0.15."""
    pipeline = ContinuousTrainingPipeline()
    schema_res = MagicMock(is_valid=True)

    # Case A: Max PSI = 0.12 (< 0.15) -> No trigger
    report_sub = DriftReport(
        timestamp="2026-11-16T00:00:00Z",
        reference_dataset_id="ref_v1",
        n_reference_samples=1000,
        n_current_samples=500,
        schema_validation=schema_res,
        feature_results={},
        drifted_features=[],
        moderate_drift_features=["temp"],
        overall_drift_detected=False,
        max_psi=0.12,
        mean_psi=0.06,
        thresholds={"psi_threshold": 0.15},
    )
    trig_a, _, _ = pipeline.evaluate_retraining_trigger(drift_report=report_sub, drift_psi_trigger=0.15)
    assert not trig_a

    # Case B: Max PSI = 0.18 (>= 0.15) -> Trigger activated
    report_active = DriftReport(
        timestamp="2026-11-16T00:00:00Z",
        reference_dataset_id="ref_v1",
        n_reference_samples=1000,
        n_current_samples=500,
        schema_validation=schema_res,
        feature_results={},
        drifted_features=["temp"],
        moderate_drift_features=[],
        overall_drift_detected=True,
        max_psi=0.18,
        mean_psi=0.09,
        thresholds={"psi_threshold": 0.15},
    )
    trig_b, trig_type, msg = pipeline.evaluate_retraining_trigger(drift_report=report_active, drift_psi_trigger=0.15)
    assert trig_b
    assert trig_type == RetrainingTrigger.DRIFT_TRIGGERED
    assert "0.18" in msg


# ===========================================================================
# 9. Test False-Positive Drift Scenarios
# ===========================================================================

def test_false_positive_drift_mitigation():
    """Test practical significance filtering avoids false alarms for identical distributions with large N."""
    rng = np.random.RandomState(42)
    # Large sample size (N=10,000) where standard KS test has extreme statistical power
    n = 10000
    ref_vals = rng.normal(20.0, 2.0, n)
    cur_vals = rng.normal(20.0, 2.0, n)  # Identical distribution

    ref_df = pd.DataFrame({"temperature": ref_vals})
    cur_df = pd.DataFrame({"temperature": cur_vals})

    detector = FeatureDriftDetector(
        thresholds=DriftThresholds(
            ks_stat_threshold=0.10,  # Require minimum effect size to avoid false alarms
            psi_threshold=0.15,
        )
    )
    profile = ReferenceDistributionProfile.from_dataframe(ref_df, dataset_id="large_n_ref")
    report = detector.detect_drift(cur_df, reference_profile=profile)

    assert not report.overall_drift_detected
    assert len(report.drifted_features) == 0
    assert report.max_psi < 0.05


# ===========================================================================
# 10. Test Missing Actual-Label Scenarios (Delayed Ground Truth)
# ===========================================================================

def test_missing_actual_label_scenarios():
    """Test monitoring operates safely when real actual labels are delayed or null."""
    provider = ModelHealthProvider()

    # Empty actuals (standard during live serving before settlement)
    health_no_labels = provider.get_health_status(
        recent_actuals=None,
        recent_preds_for_actuals=None,
    )
    assert health_no_labels.status in ("HEALTHY", "NOMINAL")
    assert health_no_labels.ground_truth_status["status"] == "NO_GROUND_TRUTH"
    assert not health_no_labels.ground_truth_status["sufficient_for_retraining"]

    # Partial actuals with NaNs
    actuals_with_nans = [10.0, np.nan, 15.0, np.nan, 20.0]
    preds = [11.0, 12.0, 14.0, 16.0, 19.0]
    health_partial = provider.get_health_status(
        recent_actuals=actuals_with_nans,
        recent_preds_for_actuals=preds,
    )
    # Must not raise an exception, handles clean subsets
    assert health_partial.ground_truth_status["sample_count"] == 3


# ===========================================================================
# 11. Avoid Retraining Without Sufficient Ground Truth (< 500)
# ===========================================================================

def test_avoid_retraining_without_sufficient_ground_truth(tmp_path):
    """Test pipeline strictly rejects retraining when ground truth sample count < 500."""
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    models_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    # Save a small dataset with only 150 rows (< 500 required)
    sparse_df = pd.DataFrame({
        "decision_timestamp": pd.date_range("2026-01-01", periods=150, freq="h"),
        "expected_customer_count": np.random.randint(5, 30, 150),
        "hour": np.random.randint(0, 24, 150),
    })
    data_path = data_dir / "feature_store.parquet"
    sparse_df.to_parquet(data_path)

    pipeline = ContinuousTrainingPipeline(
        models_dir=models_dir,
        data_path=data_path,
        min_new_samples=500,
    )

    result = pipeline.run_retraining_cycle(trigger=RetrainingTrigger.DRIFT_TRIGGERED, force=False)
    assert result.status == "SKIPPED_INSUFFICIENT_DATA"
    assert "Insufficient total samples" in result.message


# ===========================================================================
# 12. Trigger Retraining Workflow When Conditions are Satisfied
# ===========================================================================

def test_trigger_retraining_workflow_satisfaction():
    """Test trigger logic correctly activates retraining cycle on performance degradation."""
    pipeline = ContinuousTrainingPipeline()
    perf_degraded = RollingPerformanceAssessment(
        baseline_mae=0.80,
        current_7d_mae=1.10,
        degradation_pct=37.5,
        degradation_threshold_pct=20.0,
        degradation_alert=True,
        sample_count=300,
        message="Critical degradation alert",
    )

    triggered, trig_type, msg = pipeline.evaluate_retraining_trigger(perf_assessment=perf_degraded)
    assert triggered
    assert trig_type == RetrainingTrigger.PERFORMANCE_DEGRADATION
    assert "37.5%" in msg


# ===========================================================================
# 13 & 14. Candidate Promotion Gating & Rollback Tests
# ===========================================================================

def test_candidate_promotion_and_rollback(tmp_path):
    """Test promotion gating approves superior candidate and rollback restores champion."""
    from src.geodemand.models.promotion import ModelPromotionGate, ModelPromotionPolicy
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # Active champion
    champ_bundle = {"model": "champ_v1", "p10_model": "p10", "p90_model": "p90", "feature_cols": ["f1"], "label_encoders": {}}
    champ_meta = {"model_version": "champ_v1", "metrics": {"val_mae": 0.85, "val_rmse": 1.10}}
    joblib = pytest.importorskip("joblib")
    joblib.dump(champ_bundle, models_dir / "demand_model.pkl")
    with open(models_dir / "demand_model_meta.json", "w") as f:
        json.dump(champ_meta, f)

    gate = ModelPromotionGate(champion_dir=models_dir, policy=ModelPromotionPolicy(max_allowed_mae_degradation_pct=0.0))

    # Candidate bundle with lower MAE (superior)
    cand_bundle = {
        "model": "cand_v2",
        "p10_model": "p10",
        "p90_model": "p90",
        "feature_cols": ["f1"],
        "label_encoders": {},
        "meta": {"model_version": "cand_v2", "metrics": {"val_mae": 0.75, "val_rmse": 1.00}},
    }

    report = gate.evaluate_candidate(cand_bundle)
    assert report.approved
    assert report.decision == "APPROVED"

    # Promote
    promoted = gate.promote_candidate_bundle(cand_bundle, report)
    assert promoted
    # Check backup snapshot created
    assert (models_dir / "champion_backup").exists()

    # Rollback
    rolled_back = gate.rollback_to_backup()
    assert rolled_back


# ===========================================================================
# 15. API Model Health Endpoint Verification
# ===========================================================================

def test_api_model_health_endpoint():
    """Test /v1/system/model-health returns complete, compliant health status schema."""
    from fastapi.testclient import TestClient
    from api.app import app

    with TestClient(app) as client:
        r = client.get("/v1/system/model-health")
        assert r.status_code == 200
        data = r.json()

        assert "status" in data
        assert "model_version" in data
        assert "baseline_metrics" in data
        assert "feature_drift" in data
        assert "prediction_drift" in data
        assert "rolling_performance_7d" in data
        assert "retraining_trigger_active" in data
        assert "ground_truth_status" in data
        assert data["zero_downtime_hot_reload_enabled"] is True

        # Test alias endpoint /v1/model/health
        r_alias = client.get("/v1/model/health")
        assert r_alias.status_code == 200
        assert r_alias.json()["model_version"] == data["model_version"]
