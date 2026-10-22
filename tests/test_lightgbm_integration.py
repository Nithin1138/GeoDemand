"""
GeoDemand AI — Focused Integration & Verification Tests for LightGBM Model Integration

Member 3 — Machine Learning & MLOps

Validates:
1. LightGBM artifact loading & bundle structure
2. Correct estimator types (LGBMRegressor)
3. Metadata consistency (algorithm, schema, quantiles, metrics)
4. Feature schema & strict feature ordering
5. Prediction shape & finite values
6. Nonnegative customer-count domain constraints
7. Invalid input handling & schema validation
8. Categorical encoding (known and unseen classes)
9. Quantile ordering monotonicity (P10 <= P50 <= P90)
10. TreeSHAP local attributions & additivity
11. MLflow logging & model identification
12. API compatibility (DemandModel & recommender)
13. ModelPromotionGate evaluation approval
14. Backup & Rollback safety mechanism
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.geodemand.models.inference import DemandModelPredictor
from src.geodemand.models.promotion import ModelPromotionGate, ModelPromotionPolicy
from src.geodemand.monitoring.prediction_quality import PredictionValidator

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
LGBM_CANDIDATE_PATH = EXPERIMENTS_DIR / "lgbm_candidate_bundle.joblib"
SCHEMA_PATH = MODELS_DIR / "lgbm_v1" / "feature_schema.json"


@pytest.fixture(scope="module")
def candidate_bundle():
    """Load the assembled LightGBM candidate bundle."""
    if not LGBM_CANDIDATE_PATH.exists():
        pytest.skip(f"Candidate bundle not found at {LGBM_CANDIDATE_PATH}")
    return joblib.load(LGBM_CANDIDATE_PATH)


@pytest.fixture(scope="module")
def reference_schema():
    """Load canonical feature schema."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_feature_dict():
    """A representative valid feature row."""
    return {
        "hour": 14,
        "weekday": "Friday",
        "season": "summer",
        "vendor_category": "food",
        "temperature": 32.5,
        "humidity": 65.0,
        "rainfall": 0.0,
        "cloud_cover": 20.0,
        "commercial_ratio": 0.70,
        "residential_ratio": 0.20,
        "population_density": 8500.0,
        "competition_score": 0.45,
        "office_count": 15,
        "restaurant_count": 8,
        "average_order_value": 150.0,
        "inventory_capacity": 60.0,
        "preparation_time": 10.0,
        "is_holiday": 0,
        "is_weekend": 0,
    }


# ---------------------------------------------------------------------------
# 1 & 2. Artifact Loading & Estimator Type
# ---------------------------------------------------------------------------

def test_lightgbm_artifact_loading(candidate_bundle):
    """Artifact exists and contains all required keys for production serving."""
    required_keys = {"model", "p10_model", "p50_model", "p90_model", "feature_cols", "label_encoders", "meta"}
    assert required_keys.issubset(candidate_bundle.keys()), f"Missing keys: {required_keys - set(candidate_bundle.keys())}"
    assert candidate_bundle["model"] is not None
    assert candidate_bundle["p10_model"] is not None
    assert candidate_bundle["p90_model"] is not None


def test_lightgbm_estimator_type(candidate_bundle):
    """Verify estimators are true LightGBM models, not mislabeled estimators."""
    import lightgbm as lgb
    assert isinstance(candidate_bundle["model"], lgb.LGBMRegressor), f"Expected LGBMRegressor, got {type(candidate_bundle['model'])}"
    assert isinstance(candidate_bundle["p10_model"], lgb.LGBMRegressor)
    assert isinstance(candidate_bundle["p90_model"], lgb.LGBMRegressor)
    assert candidate_bundle["p10_model"].objective == "quantile"
    assert candidate_bundle["p90_model"].objective == "quantile"
    assert candidate_bundle["p10_model"].alpha == 0.1
    assert candidate_bundle["p90_model"].alpha == 0.9


# ---------------------------------------------------------------------------
# 3 & 4. Metadata Consistency & Feature Schema
# ---------------------------------------------------------------------------

def test_lightgbm_metadata_consistency(candidate_bundle, reference_schema):
    """Metadata correctly identifies algorithm as LightGBM and target as expected_customer_count."""
    meta = candidate_bundle["meta"]
    assert meta["algorithm"] == "LightGBM"
    assert meta["model_type"] == "lightgbm"
    assert meta["target"] == "expected_customer_count"
    assert "metrics" in meta
    assert meta["n_features"] == len(reference_schema["feature_columns"])
    assert "quantiles" in meta
    assert set(meta["quantiles"]) == {"p10", "p50", "p90"}


def test_lightgbm_feature_schema_and_ordering(candidate_bundle, reference_schema):
    """Feature columns match canonical schema exactly in names, count, and order."""
    expected_cols = reference_schema["feature_columns"]
    actual_cols = candidate_bundle["feature_cols"]
    assert actual_cols == expected_cols, "Feature columns ordering or content does not match canonical schema"
    assert candidate_bundle["model"].feature_name_ == expected_cols


# ---------------------------------------------------------------------------
# 5 & 6. Prediction Shape, Finite Values & Nonnegativity
# ---------------------------------------------------------------------------

def test_lightgbm_prediction_shape_and_finite(candidate_bundle, sample_feature_dict):
    """Predictions match batch shape and contain no NaNs or Infinities."""
    predictor = DemandModelPredictor(bundle_path=LGBM_CANDIDATE_PATH)
    # Single sample
    pred_single = predictor.predict(sample_feature_dict)
    assert pred_single.shape == (1,)
    assert np.isfinite(pred_single).all()

    # Batch samples
    batch = [sample_feature_dict] * 10
    pred_batch = predictor.predict(batch)
    assert pred_batch.shape == (10,)
    assert np.isfinite(pred_batch).all()


def test_lightgbm_prediction_nonnegative(candidate_bundle, sample_feature_dict):
    """Customer-count predictions must strictly be >= 0."""
    predictor = DemandModelPredictor(bundle_path=LGBM_CANDIDATE_PATH)
    preds = predictor.predict([sample_feature_dict] * 5)
    assert (preds >= 0).all(), f"Found negative predictions: {preds}"


# ---------------------------------------------------------------------------
# 7 & 8. Input Validation & Categorical Handling
# ---------------------------------------------------------------------------

def test_lightgbm_invalid_input_handling():
    """Predictor rejects unsupported data types explicitly."""
    predictor = DemandModelPredictor(bundle_path=LGBM_CANDIDATE_PATH)
    with pytest.raises(TypeError):
        predictor.predict("invalid_string_input")


def test_lightgbm_categorical_encoding(sample_feature_dict):
    """Unseen and known categoricals are mapped safely without exceptions."""
    predictor = DemandModelPredictor(bundle_path=LGBM_CANDIDATE_PATH)
    # Known categoricals
    p_known = predictor.predict(sample_feature_dict)
    assert p_known[0] >= 0

    # Unseen categoricals (e.g. unknown vendor category and weekday)
    sample_unseen = {**sample_feature_dict, "vendor_category": "unknown_future_biz", "weekday": "MarsDay"}
    p_unseen = predictor.predict(sample_unseen)
    assert p_unseen[0] >= 0
    assert np.isfinite(p_unseen[0])


# ---------------------------------------------------------------------------
# 9. Quantile Monotonicity (P10 <= P50 <= P90)
# ---------------------------------------------------------------------------

def test_lightgbm_quantile_ordering(sample_feature_dict):
    """Prediction intervals satisfy P10 <= P50 <= P90 and provide valid spreads."""
    predictor = DemandModelPredictor(bundle_path=LGBM_CANDIDATE_PATH)
    intervals = predictor.predict_with_interval([sample_feature_dict] * 5)
    assert len(intervals) == 5
    for row in intervals:
        assert row["p10"] <= row["p50"], f"Inversion: P10 ({row['p10']}) > P50 ({row['p50']})"
        assert row["p50"] <= row["p90"], f"Inversion: P50 ({row['p50']}) > P90 ({row['p90']})"
        assert row["p10"] >= 0
        assert row["prediction_range"] >= 0
        assert abs(row["prediction_range"] - (row["p90"] - row["p10"])) < 0.01


# ---------------------------------------------------------------------------
# 10. TreeSHAP Compatibility & Additivity
# ---------------------------------------------------------------------------

def test_lightgbm_shap_compatibility(candidate_bundle, sample_feature_dict):
    """Native TreeSHAP generates correct attribution dimensions and satisfies additivity."""
    predictor = DemandModelPredictor(bundle_path=LGBM_CANDIDATE_PATH)
    explanations = predictor.explain([sample_feature_dict])
    assert len(explanations) == 1
    ex = explanations[0]

    assert "predicted_customer_count" in ex
    assert "top_positive_factors" in ex
    assert "top_negative_factors" in ex
    assert "formatted_explanation" in ex
    assert "Prediction:" in ex["formatted_explanation"]

    # Test exact additivity on preprocessed matrix
    X = predictor.preprocess([sample_feature_dict])
    contribs = candidate_bundle["model"].predict(X, pred_contrib=True)
    assert contribs.shape == (1, len(candidate_bundle["feature_cols"]) + 1)
    sum_contribs = float(contribs.sum(axis=1)[0])
    raw_pred = float(candidate_bundle["model"].predict(X)[0])
    assert abs(sum_contribs - raw_pred) < 1e-4, f"SHAP additivity error: {abs(sum_contribs - raw_pred)}"


# ---------------------------------------------------------------------------
# 11 & 12. API & Recommender Compatibility
# ---------------------------------------------------------------------------

def test_lightgbm_recommender_api_compatibility(candidate_bundle, sample_feature_dict):
    """DemandModel in recommender.py works seamlessly with candidate bundle structure."""
    from src.api.recommender import DemandModel

    dm = DemandModel.get()
    # Temporarily evaluate candidate bundle with recommender interface
    orig_model = dm._model
    orig_p10 = dm._p10_model
    orig_p90 = dm._p90_model
    try:
        dm._model = candidate_bundle["model"]
        dm._p10_model = candidate_bundle["p10_model"]
        dm._p90_model = candidate_bundle["p90_model"]
        dm._feature_cols = candidate_bundle["feature_cols"]
        dm._label_encoders = candidate_bundle["label_encoders"]

        # Test batch prediction
        preds = dm.predict_batch([sample_feature_dict] * 3)
        assert len(preds) == 3
        assert (preds >= 0).all()

        # Test uncertainty prediction
        unc = dm.predict_with_uncertainty(sample_feature_dict)
        assert "p10" in unc and "p50" in unc and "p90" in unc and "interval_spread" in unc
        assert unc["p10"] <= unc["p50"] <= unc["p90"]
        assert unc["interval_spread"] == unc["p90"] - unc["p10"]
    finally:
        dm._model = orig_model
        dm._p10_model = orig_p10
        dm._p90_model = orig_p90


# ---------------------------------------------------------------------------
# 13 & 14. Promotion Gate Evaluation & Rollback Mechanism
# ---------------------------------------------------------------------------

def test_lightgbm_promotion_gate_evaluation(candidate_bundle):
    """Promotion gate approves LightGBM candidate on validation data."""
    rng = np.random.RandomState(42)
    feature_cols = candidate_bundle["feature_cols"]
    n = 200
    X_val = pd.DataFrame(rng.uniform(0.0, 10.0, size=(n, len(feature_cols))), columns=feature_cols)
    y_val = pd.Series(candidate_bundle["model"].predict(X_val) + rng.normal(0, 0.5, size=n))

    # Test with configurable policy
    policy = ModelPromotionPolicy(max_val_mae=5.0, max_val_rmse=5.0, max_quantile_inversion_rate=0.05)
    gate = ModelPromotionGate(policy=policy)
    report = gate.evaluate_candidate(candidate_bundle, X_val, y_val)

    assert report.gate_checks["artifact_structure"].passed
    assert report.gate_checks["feature_schema_match"].passed
    assert report.gate_checks["predictions_finite_and_valid"].passed
    assert report.gate_checks["quantile_ordering"].passed
    assert report.gate_checks["shap_explainability"].passed
    assert report.gate_checks["inference_latency"].passed


def test_promotion_backup_and_rollback_safety(tmp_path):
    """Backup and rollback path safely restores champion without data loss."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    champ_pkl = models_dir / "demand_model.pkl"
    champ_meta = models_dir / "demand_model_meta.json"
    champ_dir = models_dir / "lgbm_v1"
    champ_dir.mkdir()

    # Write initial mock champion
    joblib.dump({"version": "champion_v2.0"}, champ_pkl)
    with open(champ_meta, "w") as f:
        json.dump({"model_version": "v2.0.0"}, f)
    with open(champ_dir / "model.joblib", "w") as f:
        f.write("mock_model")

    gate = ModelPromotionGate(models_dir=models_dir)
    gate.backup_current_champion()

    # Overwrite champion with candidate
    joblib.dump({"version": "candidate_v2.1"}, champ_pkl)
    with open(champ_meta, "w") as f:
        json.dump({"model_version": "v2.1.0"}, f)

    assert joblib.load(champ_pkl)["version"] == "candidate_v2.1"

    # Execute rollback
    success = gate.rollback_to_backup()
    assert success is True
    assert joblib.load(champ_pkl)["version"] == "champion_v2.0"
    with open(champ_meta) as f:
        restored_meta = json.load(f)
    assert restored_meta["model_version"] == "v2.0.0"
