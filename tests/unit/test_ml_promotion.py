"""
GeoDemand AI — Unit Tests for Model Promotion Gate & Rollback (Step 5)

Member 3 — Machine Learning & MLOps

Tests:
1. test_promotion_policy_defaults
2. test_gate_evaluation_superior_candidate_approved
3. test_gate_evaluation_inferior_candidate_rejected
4. test_gate_evaluation_schema_mismatch_rejected
5. test_gate_evaluation_nan_predictions_rejected
6. test_gate_evaluation_quantile_inversion_rejected
7. test_atomic_backup_and_rollback
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.geodemand.models.promotion import (
    GateCheckResult,
    ModelPromotionGate,
    ModelPromotionPolicy,
    PromotionEvaluationReport,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def dummy_validation_data():
    """Generates synthetic validation features and targets."""
    rng = np.random.RandomState(42)
    schema_path = ROOT_DIR / "models" / "lgbm_v1" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_cols = schema["feature_columns"]

    n = 100
    X = pd.DataFrame(rng.uniform(0.0, 10.0, size=(n, len(feature_cols))), columns=feature_cols)
    y = pd.Series(rng.poisson(lam=15.0, size=n).astype(float))
    return X, y, feature_cols


class MockModel:
    """Mock model that returns predictable predictions."""

    def __init__(self, pred_offset: float = 0.0, has_nan: bool = False):
        self.pred_offset = pred_offset
        self.has_nan = has_nan

    def predict(self, X, **kwargs):
        n = len(X)
        if self.has_nan:
            res = np.full(n, 15.0)
            res[0] = np.nan
            return res
        return np.full(n, 15.0 + self.pred_offset)


class MockQuantileModel:
    """Mock model for quantile predictions."""

    def __init__(self, val: float):
        self.val = val

    def predict(self, X, **kwargs):
        return np.full(len(X), self.val)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_promotion_policy_defaults():
    """Verify default promotion policy adheres to production constraints."""
    policy = ModelPromotionPolicy()
    assert policy.max_val_mae == 1.00
    assert policy.max_val_rmse == 1.30
    assert policy.require_monotonic_quantiles is True
    assert policy.require_shap_support is True
    assert policy.require_exact_schema_match is True
    assert policy.require_non_negative_outputs is True


def test_gate_evaluation_superior_candidate_approved(dummy_validation_data, tmp_path):
    """Test candidate with lower MAE than champion is approved."""
    X_val, y_val, feature_cols = dummy_validation_data

    gate = ModelPromotionGate(models_dir=tmp_path)
    # Mock champion metadata with MAE 1.00
    meta_path = tmp_path / "demand_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"metrics": {"val_mae": 1.00, "val_rmse": 1.20}, "model_version": "v_champ", "model_type": "xgboost"}, f)

    # Candidate with perfect predictions (MAE = 0.0)
    cand_bundle = {
        "model": MockModel(pred_offset=0.0),
        "p10_model": MockQuantileModel(val=10.0),
        "p90_model": MockQuantileModel(val=20.0),
        "feature_cols": feature_cols,
        "label_encoders": {},
        "meta": {"model_version": "v_candidate_best", "model_type": "lightgbm"},
    }

    # Set y_val to match candidate predictions so MAE is 0.0
    y_perfect = pd.Series(np.full(len(X_val), 15.0))
    report = gate.evaluate_candidate(cand_bundle, X_val, y_perfect)

    assert isinstance(report, PromotionEvaluationReport)
    assert report.decision == "APPROVED"
    assert report.passed_all_gates is True
    assert report.candidate_metrics["val_mae"] < 0.01


def test_gate_evaluation_inferior_candidate_rejected(dummy_validation_data, tmp_path):
    """Test candidate with higher MAE than champion is rejected."""
    X_val, y_val, feature_cols = dummy_validation_data

    gate = ModelPromotionGate(models_dir=tmp_path)
    meta_path = tmp_path / "demand_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"metrics": {"val_mae": 0.50, "val_rmse": 0.70}, "model_version": "v_champ", "model_type": "xgboost"}, f)

    # Candidate with offset +5.0 (MAE ~ 5.0)
    cand_bundle = {
        "model": MockModel(pred_offset=5.0),
        "p10_model": MockQuantileModel(val=10.0),
        "p90_model": MockQuantileModel(val=25.0),
        "feature_cols": feature_cols,
        "label_encoders": {},
        "meta": {"model_version": "v_candidate_poor", "model_type": "catboost"},
    }

    report = gate.evaluate_candidate(cand_bundle, X_val, y_val)
    assert report.decision == "REJECTED"
    assert report.passed_all_gates is False
    assert any("Candidate MAE" in r for r in report.reasons)


def test_gate_evaluation_schema_mismatch_rejected(dummy_validation_data, tmp_path):
    """Test candidate with mismatched feature schema is rejected."""
    X_val, y_val, feature_cols = dummy_validation_data

    gate = ModelPromotionGate(models_dir=tmp_path)
    # Setup dummy schema in tmp_path
    schema_dir = tmp_path / "lgbm_v1"
    schema_dir.mkdir(parents=True)
    with open(schema_dir / "feature_schema.json", "w") as f:
        json.dump({"feature_columns": feature_cols}, f)

    # Candidate with missing features
    cand_bundle = {
        "model": MockModel(),
        "p10_model": MockQuantileModel(val=10.0),
        "p90_model": MockQuantileModel(val=20.0),
        "feature_cols": feature_cols[:10],  # Only 10 columns
        "label_encoders": {},
        "meta": {"model_version": "v_cand_broken", "model_type": "xgboost"},
    }

    report = gate.evaluate_candidate(cand_bundle, X_val, y_val)
    assert report.decision == "REJECTED"
    assert not report.gate_checks["feature_schema_match"].passed


def test_gate_evaluation_nan_predictions_rejected(dummy_validation_data, tmp_path):
    """Test candidate returning NaN predictions is rejected."""
    X_val, y_val, feature_cols = dummy_validation_data

    gate = ModelPromotionGate(models_dir=tmp_path)
    cand_bundle = {
        "model": MockModel(has_nan=True),
        "p10_model": MockQuantileModel(val=10.0),
        "p90_model": MockQuantileModel(val=20.0),
        "feature_cols": feature_cols,
        "label_encoders": {},
        "meta": {"model_version": "v_cand_nan", "model_type": "xgboost"},
    }

    report = gate.evaluate_candidate(cand_bundle, X_val, y_val)
    assert report.decision == "REJECTED"
    assert not report.gate_checks["predictions_finite_and_valid"].passed


def test_gate_evaluation_quantile_inversion_rejected(dummy_validation_data, tmp_path):
    """Test candidate with inverted quantiles (P10 > P90) is rejected."""
    X_val, y_val, feature_cols = dummy_validation_data

    gate = ModelPromotionGate(models_dir=tmp_path)
    cand_bundle = {
        "model": MockModel(),
        "p10_model": MockQuantileModel(val=30.0),  # P10 = 30.0 > P50 (15.0)
        "p90_model": MockQuantileModel(val=10.0),  # P90 = 10.0 < P50 (15.0)
        "feature_cols": feature_cols,
        "label_encoders": {},
        "meta": {"model_version": "v_cand_inverted", "model_type": "xgboost"},
    }

    report = gate.evaluate_candidate(cand_bundle, X_val, y_val)
    assert report.decision == "REJECTED"
    assert report.gate_checks["quantile_ordering"].passed is False


def test_gate_evaluation_latency_exceeded_rejected(dummy_validation_data, tmp_path):
    """Test candidate with latency exceeding policy threshold is rejected."""
    X_val, y_val, feature_cols = dummy_validation_data
    policy = ModelPromotionPolicy(max_latency_p95_ms=0.01)
    gate = ModelPromotionGate(policy=policy, models_dir=tmp_path)

    class SlowMockModel(MockModel):
        def predict(self, X, **kwargs):
            import time
            time.sleep(0.01)
            return super().predict(X, **kwargs)

    cand_bundle = {
        "model": SlowMockModel(),
        "p10_model": MockQuantileModel(val=10.0),
        "p90_model": MockQuantileModel(val=20.0),
        "feature_cols": feature_cols,
        "label_encoders": {},
        "meta": {"model_version": "v_slow", "model_type": "xgboost"},
    }

    report = gate.evaluate_candidate(cand_bundle, X_val, y_val)
    assert report.decision == "REJECTED"
    assert report.gate_checks["inference_latency"].passed is False


def test_gate_evaluation_extreme_prediction_rejected(dummy_validation_data, tmp_path):
    """Test candidate producing extreme prediction values > 50,000 is rejected."""
    X_val, y_val, feature_cols = dummy_validation_data
    gate = ModelPromotionGate(models_dir=tmp_path)

    cand_bundle = {
        "model": MockModel(pred_offset=100000.0),  # Extreme value
        "p10_model": MockQuantileModel(val=10.0),
        "p90_model": MockQuantileModel(val=20.0),
        "feature_cols": feature_cols,
        "label_encoders": {},
        "meta": {"model_version": "v_extreme", "model_type": "xgboost"},
    }

    report = gate.evaluate_candidate(cand_bundle, X_val, y_val)
    assert report.decision == "REJECTED"
    assert report.gate_checks["predictions_finite_and_valid"].passed is False


def test_gate_evaluation_missing_keys_rejected(dummy_validation_data, tmp_path):
    """Test candidate bundle missing required keys is rejected."""
    X_val, y_val, feature_cols = dummy_validation_data
    gate = ModelPromotionGate(models_dir=tmp_path)

    # Missing p90_model
    cand_bundle = {
        "model": MockModel(),
        "p10_model": MockQuantileModel(val=10.0),
        "feature_cols": feature_cols,
        "label_encoders": {},
    }

    report = gate.evaluate_candidate(cand_bundle, X_val, y_val)
    assert report.decision == "REJECTED"
    assert report.gate_checks["artifact_structure"].passed is False


def test_atomic_backup_and_rollback(tmp_path):
    """Test snapshot backup creation and atomic rollback restore."""
    gate = ModelPromotionGate(models_dir=tmp_path)

    # 1. Create dummy original champion files
    champ_bundle = tmp_path / "demand_model.pkl"
    champ_meta = tmp_path / "demand_model_meta.json"
    champ_dir = tmp_path / "lgbm_v1"
    champ_dir.mkdir(parents=True)

    with open(champ_bundle, "w") as f:
        f.write("ORIGINAL_BUNDLE_CONTENT")
    with open(champ_meta, "w") as f:
        f.write('{"version": "ORIGINAL_CHAMPION"}')
    with open(champ_dir / "model.joblib", "w") as f:
        f.write("ORIGINAL_JOB_LIB")

    # 2. Backup champion
    backup_dir = gate.backup_current_champion()
    assert (backup_dir / "demand_model.pkl").exists()
    assert (backup_dir / "demand_model_meta.json").exists()
    assert (backup_dir / "lgbm_v1" / "model.joblib").exists()

    # 3. Simulate accidental file corruption or overwrite
    with open(champ_bundle, "w") as f:
        f.write("CORRUPTED_BUNDLE")

    # 4. Rollback
    rollback_ok = gate.rollback_to_backup()
    assert rollback_ok is True
    with open(champ_bundle, "r") as f:
        restored = f.read()
    assert restored == "ORIGINAL_BUNDLE_CONTENT"


def test_atomic_promotion_and_synchronization(tmp_path):
    """Test atomic promotion updates both root bundle and lgbm_v1 structured directory."""
    gate = ModelPromotionGate(models_dir=tmp_path)

    # Initial champion files
    champ_bundle = tmp_path / "demand_model.pkl"
    champ_meta = tmp_path / "demand_model_meta.json"
    champ_dir = tmp_path / "lgbm_v1"
    champ_dir.mkdir(parents=True)

    import joblib
    initial_bundle = {"model": MockModel(pred_offset=1.0), "p10_model": MockQuantileModel(5.0), "p90_model": MockQuantileModel(15.0), "feature_cols": ["f1"]}
    joblib.dump(initial_bundle, champ_bundle)
    with open(champ_meta, "w") as f:
        json.dump({"model_version": "v1.0.0"}, f)
    joblib.dump(initial_bundle["model"], champ_dir / "model.joblib")

    # New candidate
    candidate_bundle = {
        "model": MockModel(pred_offset=2.0),
        "p10_model": MockQuantileModel(8.0),
        "p90_model": MockQuantileModel(22.0),
        "feature_cols": ["f1"],
        "label_encoders": {},
    }
    candidate_meta = {"model_version": "v2.0.0", "metrics": {"val_mae": 0.5}}

    success = gate.promote_candidate_bundle(candidate_bundle, candidate_meta)
    assert success is True

    # Verify both bundle and lgbm_v1 were updated
    promoted_bundle = joblib.load(champ_bundle)
    assert promoted_bundle["model"].pred_offset == 2.0
    promoted_model_art = joblib.load(champ_dir / "model.joblib")
    assert promoted_model_art.pred_offset == 2.0
    with open(champ_dir / "metadata.json") as f:
        meta_loaded = json.load(f)
    assert meta_loaded["model_version"] == "v2.0.0"


def test_promotion_failure_triggers_automatic_rollback(tmp_path):
    """Test automatic rollback restores previous champion if promotion fails mid-execution."""
    gate = ModelPromotionGate(models_dir=tmp_path)

    import joblib
    champ_bundle = tmp_path / "demand_model.pkl"
    champ_meta = tmp_path / "demand_model_meta.json"
    initial_bundle = {"model": "ORIGINAL_MODEL"}
    joblib.dump(initial_bundle, champ_bundle)
    with open(champ_meta, "w") as f:
        json.dump({"model_version": "v_orig"}, f)

    # Candidate with unpicklable or error-inducing model (e.g. lambda)
    bad_candidate = {
        "model": lambda x: x,
    }
    success = gate.promote_candidate_bundle(bad_candidate, {})
    assert success is False

    # Verify champion was restored from backup
    restored_bundle = joblib.load(champ_bundle)
    assert restored_bundle["model"] == "ORIGINAL_MODEL"
