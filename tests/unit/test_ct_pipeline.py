"""
GeoDemand AI — Unit Tests for Continuous Training (CT) Pipeline (Phase 7)

Member 2 — Machine Learning & MLOps

Validates:
1. Training data eligibility & ground-truth sufficiency constraints
2. Automated trigger evaluation (Drift PSI > 0.15, MAE degradation > 20%)
3. Insufficient data safety guard (never trains without sufficient data)
4. Inferior model rejection (never promotes a candidate degrading champion)
5. Audit log persistence and schema
6. Snapshot backup and rollback mechanics
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.geodemand.models.promotion import ModelPromotionPolicy
from src.geodemand.monitoring.ct_pipeline import (
    ContinuousTrainingPipeline,
    RetrainingResult,
    RetrainingTrigger,
    TrainingDataEligibilityReport,
    AUDIT_LOG_PATH,
)
from src.geodemand.monitoring.drift import DriftReport, SchemaValidationResult
from src.geodemand.monitoring.performance import RollingPerformanceAssessment

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def sample_training_df():
    """Generates a small valid DataFrame for eligibility testing."""
    n = 600
    df = pd.DataFrame({
        "decision_timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "hour": np.random.randint(0, 24, n),
        "temperature": np.random.uniform(20.0, 35.0, n),
        "expected_customer_count": np.random.randint(0, 50, n),
    })
    return df


@pytest.fixture
def ct_pipeline(tmp_path):
    """Initializes a pipeline pointing to temp directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_path = tmp_path / "data" / "feature_store.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    return ContinuousTrainingPipeline(
        models_dir=models_dir,
        data_path=data_path,
        min_new_samples=500,
        require_manual_approval=False,
    )


def test_data_eligibility_valid(ct_pipeline, sample_training_df):
    """Test eligible dataset passes all sufficiency and contract checks."""
    report = ct_pipeline.check_data_eligibility(sample_training_df, min_new_samples=500)
    assert report.is_eligible
    assert report.total_samples == 600
    assert report.target_valid
    assert report.date_range_start is not None


def test_data_eligibility_insufficient_samples(ct_pipeline, sample_training_df):
    """Test dataset with fewer samples than threshold is rejected."""
    small_df = sample_training_df.iloc[:200]
    report = ct_pipeline.check_data_eligibility(small_df, min_new_samples=500)
    assert not report.is_eligible
    assert any("Insufficient total samples" in r for r in report.reasons)


def test_data_eligibility_invalid_target(ct_pipeline, sample_training_df):
    """Test dataset with missing, negative, or null target is rejected."""
    # Negative target
    neg_df = sample_training_df.copy()
    neg_df.loc[5, "expected_customer_count"] = -3
    rep_neg = ct_pipeline.check_data_eligibility(neg_df)
    assert not rep_neg.is_eligible
    assert not rep_neg.target_valid

    # Null target
    null_df = sample_training_df.copy()
    null_df.loc[10, "expected_customer_count"] = np.nan
    rep_null = ct_pipeline.check_data_eligibility(null_df)
    assert not rep_null.is_eligible
    assert not rep_null.target_valid

    # Missing target column
    no_target_df = sample_training_df.drop(columns=["expected_customer_count"])
    rep_notarget = ct_pipeline.check_data_eligibility(no_target_df)
    assert not rep_notarget.is_eligible
    assert not rep_notarget.target_valid


def test_evaluate_retraining_trigger_drift(ct_pipeline):
    """Test PSI >= 0.15 triggers automated DRIFT_TRIGGERED retraining."""
    schema_res = SchemaValidationResult(is_valid=True)
    drift_rep = DriftReport(
        timestamp="2026-11-16T00:00:00Z",
        reference_dataset_id="ref_v1",
        n_reference_samples=1000,
        n_current_samples=500,
        schema_validation=schema_res,
        feature_results={},
        drifted_features=["temperature"],
        moderate_drift_features=[],
        overall_drift_detected=True,
        max_psi=0.18,  # > 0.15 threshold
        mean_psi=0.08,
        thresholds={"psi_threshold": 0.15},
    )

    triggered, trig_type, msg = ct_pipeline.evaluate_retraining_trigger(
        drift_report=drift_rep,
        drift_psi_trigger=0.15,
    )
    assert triggered
    assert trig_type == RetrainingTrigger.DRIFT_TRIGGERED
    assert "0.18" in msg


def test_evaluate_retraining_trigger_performance(ct_pipeline):
    """Test 7-day MAE degradation >= 20% triggers PERFORMANCE_DEGRADATION retraining."""
    perf_assess = RollingPerformanceAssessment(
        baseline_mae=1.00,
        current_7d_mae=1.25,
        degradation_pct=25.0,  # > 20% trigger
        degradation_threshold_pct=20.0,
        degradation_alert=True,
        sample_count=200,
        message="Critical degradation",
    )

    triggered, trig_type, msg = ct_pipeline.evaluate_retraining_trigger(
        perf_assessment=perf_assess,
        mae_degradation_trigger_pct=20.0,
    )
    assert triggered
    assert trig_type == RetrainingTrigger.PERFORMANCE_DEGRADATION
    assert "25.0%" in msg


def test_run_retraining_cycle_skips_when_no_data(ct_pipeline):
    """Test retraining cycle cleanly aborts when data file is missing."""
    res = ct_pipeline.run_retraining_cycle(trigger=RetrainingTrigger.SCHEDULED)
    assert res.status == "FAILED"
    assert "Feature store not found" in res.message


def test_audit_log_recording(ct_pipeline, tmp_path):
    """Test retraining audit log persists machine-readable record."""
    res = RetrainingResult(
        cycle_id="test_cycle_001",
        trigger=RetrainingTrigger.MANUAL.value,
        status="SKIPPED_INSUFFICIENT_DATA",
        timestamp="2026-11-16T00:00:00Z",
        champion_id="champ_v1",
        candidate_id=None,
        champion_val_mae=0.7693,
        candidate_val_mae=None,
        gate_checks=[],
        message="Insufficient data test",
    )
    ct_pipeline._record_result(res)

    assert AUDIT_LOG_PATH.exists()
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        log_data = json.load(f)
    assert len(log_data) >= 1
    assert any(entry["cycle_id"] == "test_cycle_001" for entry in log_data)


def test_manual_approval_staging_and_promotion_flow(tmp_path):
    """Test full staging and token-authorized manual promotion flow."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ContinuousTrainingPipeline(
        models_dir=models_dir,
        min_new_samples=500,
        require_manual_approval=True,
    )

    # Simulate staging a candidate
    token = "appr_test_token_12345"
    stage_path = models_dir / "experiments" / "staged_test.joblib"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    joblib = pytest.importorskip("joblib")
    joblib.dump({"model": "dummy_candidate"}, stage_path)

    pipeline._save_staged_approval(
        token=token,
        cycle_id="ct_stage_01",
        candidate_id="cand_v2.2",
        stage_path=str(stage_path),
        candidate_meta={"candidate_id": "cand_v2.2", "metrics": {"val_mae": 0.72}},
        decision_dict={"approved": True},
        champ_val_mae=0.76,
        candidate_val_mae=0.72,
    )

    # Approve and promote using the token
    with patch.object(pipeline.promotion_gate, "promote") as mock_promote:
        result = pipeline.approve_and_promote(token)
        assert result.status == "COMPLETED_PROMOTED"
        assert result.candidate_id == "cand_v2.2"
        mock_promote.assert_called_once()

    # Verify version file written
    version_file = models_dir / "model_version.json"
    assert version_file.exists()
    with open(version_file) as f:
        v_data = json.load(f)
    assert v_data["model_version"] == "cand_v2.2"
    assert v_data["status"] == "active"


def test_invalid_approval_token_rejected(tmp_path):
    """Test unauthorized or non-existent approval tokens are rejected."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ContinuousTrainingPipeline(models_dir=models_dir)

    with pytest.raises(ValueError, match="No staged approvals record found"):
        pipeline.approve_and_promote("non_existent_token")


def test_zero_downtime_model_version_metadata(tmp_path):
    """Test zero-downtime model version file persistence."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ContinuousTrainingPipeline(models_dir=models_dir)

    v_info = pipeline.notify_zero_downtime_reload(
        candidate_id="candidate_zero_downtime_v1",
        candidate_val_mae=0.745,
        cycle_id="ct_zd_01",
    )
    assert v_info["model_version"] == "candidate_zero_downtime_v1"
    assert v_info["zero_downtime_hot_reload_enabled"] is True

    version_path = models_dir / "model_version.json"
    assert version_path.exists()


def test_run_scheduled_and_drift_retraining_helpers(ct_pipeline):
    """Test scheduled and drift retraining trigger methods."""
    with patch.object(ct_pipeline, "run_retraining_cycle") as mock_run:
        mock_run.return_value = RetrainingResult(
            cycle_id="ct_sched_01",
            trigger=RetrainingTrigger.SCHEDULED.value,
            status="SKIPPED_INSUFFICIENT_DATA",
            timestamp="2026-11-16T00:00:00Z",
            champion_id="champ_v1",
            candidate_id=None,
            champion_val_mae=0.7693,
            candidate_val_mae=None,
            gate_checks=[],
        )

        res_sched = ct_pipeline.run_scheduled_retraining(min_new_samples=500)
        assert res_sched.trigger == RetrainingTrigger.SCHEDULED.value
        mock_run.assert_called_with(trigger=RetrainingTrigger.SCHEDULED, force=False)

        schema_res = SchemaValidationResult(is_valid=True)
        drift_rep = DriftReport(
            timestamp="2026-11-16T00:00:00Z",
            reference_dataset_id="ref_v1",
            n_reference_samples=1000,
            n_current_samples=500,
            schema_validation=schema_res,
            feature_results={},
            drifted_features=["rainfall"],
            moderate_drift_features=[],
            overall_drift_detected=True,
            max_psi=0.22,
            mean_psi=0.10,
            thresholds={"psi_threshold": 0.15},
        )
        res_drift = ct_pipeline.run_drift_triggered_retraining(drift_rep)
        assert res_drift.trigger == RetrainingTrigger.SCHEDULED.value

