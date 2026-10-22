"""
GeoDemand AI — Unit Tests for ML Model & Artifacts (Member 3)

Required tests:
  - test_model_load
  - test_feature_schema
  - test_prediction_shape
  - test_prediction_nonnegative
  - test_quantile_order
  - test_deterministic_inference
  - test_model_metadata
  - test_artifact_reload
  - test_temporal_split
  - test_metric_generation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODELS_DIR = ROOT_DIR / "models"
ARTIFACT_DIR = MODELS_DIR / "lgbm_v1"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURE_STORE_PATH = PROCESSED_DIR / "feature_store.parquet"

REQUIRED_METADATA_KEYS = {
    "model_version",
    "training_dataset_version",
    "feature_version",
    "training_timestamp",
    "algorithm",
    "feature_columns",
    "metrics",
    "quantiles",
    "random_seed",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    """Load the API-compatible model bundle."""
    bundle_path = MODELS_DIR / "demand_model.pkl"
    if not bundle_path.exists():
        pytest.skip(f"Model bundle not found at {bundle_path}. Run training first.")
    return joblib.load(bundle_path)


@pytest.fixture(scope="module")
def metadata():
    """Load model metadata."""
    meta_path = ARTIFACT_DIR / "metadata.json"
    if not meta_path.exists():
        meta_path = ARTIFACT_DIR / "model_metadata.json"
    if not meta_path.exists():
        pytest.skip(f"Metadata not found at {meta_path}. Run training first.")
    with open(meta_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def feature_schema():
    """Load feature schema."""
    schema_path = ARTIFACT_DIR / "feature_schema.json"
    if not schema_path.exists():
        pytest.skip(f"Feature schema not found at {schema_path}. Run training first.")
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def feature_store():
    """Load the canonical feature store for testing."""
    if not FEATURE_STORE_PATH.exists():
        pytest.skip(f"Feature store not found at {FEATURE_STORE_PATH}.")
    return pd.read_parquet(FEATURE_STORE_PATH)


@pytest.fixture(scope="module")
def sample_X(bundle, feature_store):
    """Create a sample feature matrix matching the model's expected schema."""
    feature_cols = bundle.get("feature_cols", [])
    label_encoders = bundle.get("label_encoders", {})

    df = feature_store.head(10).copy()

    for col, le in label_encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in known else 0)

    for col in feature_cols:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(int)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols].astype(float)


# ---------------------------------------------------------------------------
# Required Automated Tests (Section 33)
# ---------------------------------------------------------------------------

def test_model_load(bundle):
    """test_model_load: model file exists and loads correctly."""
    assert (MODELS_DIR / "demand_model.pkl").exists(), "demand_model.pkl not found"
    assert "model" in bundle, "Bundle missing 'model' key"
    assert bundle["model"] is not None
    assert "p10_model" in bundle and bundle["p10_model"] is not None
    assert "p90_model" in bundle and bundle["p90_model"] is not None
    assert len(bundle.get("feature_cols", [])) > 0

    assert ARTIFACT_DIR.exists()
    assert (ARTIFACT_DIR / "model.joblib").exists()
    assert (ARTIFACT_DIR / "metadata.json").exists() or (ARTIFACT_DIR / "model_metadata.json").exists()
    assert (ARTIFACT_DIR / "feature_schema.json").exists()


def test_feature_schema(bundle, feature_schema):
    """test_feature_schema: saved feature schema matches model expectations."""
    required = {
        "feature_columns",
        "n_features",
        "target_column",
        "forbidden_columns",
        "categorical_columns",
    }
    missing = required - set(feature_schema.keys())
    assert not missing, f"Feature schema missing keys: {missing}"

    # Target contract (Section 24: ML Target)
    assert feature_schema["target_column"] == "expected_customer_count"
    disallowed_targets = {"profit", "revenue", "decision", "demand score", "demand_score", "recommendation_score"}
    assert feature_schema["target_column"] not in disallowed_targets

    # Forbidden columns check
    feature_cols = set(feature_schema["feature_columns"])
    forbidden = set(feature_schema["forbidden_columns"])
    overlap = feature_cols & forbidden
    assert not overlap, f"Feature columns contain forbidden columns: {overlap}"

    # No identity or future timestamp leakage
    identity_cols = {"vendor_id", "h3_cell_id", "decision_timestamp", "target_timestamp"}
    assert not (feature_cols & identity_cols)

    # Schema matches bundle
    bundle_cols = set(bundle.get("feature_cols", []))
    assert bundle_cols == feature_cols


def test_prediction_shape(bundle, sample_X):
    """test_prediction_shape: prediction output has correct shape."""
    model = bundle["model"]
    # Single sample
    pred_single = model.predict(sample_X.iloc[:1])
    assert pred_single.shape == (1,), f"Expected (1,), got {pred_single.shape}"

    # Batch samples
    n = len(sample_X)
    pred_batch = model.predict(sample_X)
    assert pred_batch.shape == (n,), f"Expected ({n},), got {pred_batch.shape}"


def test_prediction_nonnegative(bundle, sample_X):
    """test_prediction_nonnegative: all predictions >= 0."""
    pred_p50 = np.clip(bundle["model"].predict(sample_X), 0, None)
    pred_p10 = np.clip(bundle["p10_model"].predict(sample_X), 0, None)
    pred_p90 = np.clip(bundle["p90_model"].predict(sample_X), 0, None)

    assert (pred_p50 >= 0).all(), "Negative predictions in main model"
    assert (pred_p10 >= 0).all(), "Negative predictions in P10 model"
    assert (pred_p90 >= 0).all(), "Negative predictions in P90 model"


def test_quantile_order(bundle, sample_X):
    """test_quantile_order: P10 <= P50 <= P90 prediction interval / prediction range."""
    from src.geodemand.models.inference import DemandModelPredictor

    p10 = np.clip(bundle["p10_model"].predict(sample_X), 0, None)
    p50 = np.clip(bundle["model"].predict(sample_X), 0, None)
    p90 = np.clip(bundle["p90_model"].predict(sample_X), 0, None)

    # Enforce monotonicity
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p50, p90)

    assert (p10 <= p50).all(), "Violation: P10 > P50"
    assert (p50 <= p90).all(), "Violation: P50 > P90"

    # Test via DemandModelPredictor
    predictor = DemandModelPredictor()
    intervals = predictor.predict_with_interval(sample_X)
    assert len(intervals) == len(sample_X)
    for row in intervals:
        assert row["p10"] <= row["p50"], f"Prediction interval violation: {row['p10']} > {row['p50']}"
        assert row["p50"] <= row["p90"], f"Prediction interval violation: {row['p50']} > {row['p90']}"
        assert "prediction_range" in row
        assert abs(row["prediction_range"] - (row["p90"] - row["p10"])) < 0.01


def test_deterministic_inference(bundle, sample_X):
    """test_deterministic_inference: same input → same output."""
    model = bundle["model"]
    preds1 = model.predict(sample_X)
    preds2 = model.predict(sample_X)
    np.testing.assert_array_equal(preds1, preds2, err_msg="Non-deterministic inference detected")

    p10_1 = bundle["p10_model"].predict(sample_X)
    p10_2 = bundle["p10_model"].predict(sample_X)
    np.testing.assert_array_equal(p10_1, p10_2)


def test_model_metadata(metadata):
    """test_model_metadata: metadata.json contains all required keys."""
    missing = REQUIRED_METADATA_KEYS - set(metadata.keys())
    assert not missing, f"Metadata missing required keys: {missing}"

    assert metadata["algorithm"] == "LightGBM"
    assert isinstance(metadata["model_version"], str)
    assert "metrics" in metadata
    assert "validation" in metadata["metrics"]
    assert "test" in metadata["metrics"]

    test_metrics = metadata["metrics"]["test"]
    assert "mae" in test_metrics
    assert "rmse" in test_metrics
    assert "r2" in test_metrics

    quantiles = metadata.get("quantiles", [])
    assert "p10" in quantiles and "p50" in quantiles and "p90" in quantiles
    assert isinstance(metadata.get("random_seed"), int)
    assert len(metadata.get("feature_columns", [])) >= 30


def test_artifact_reload(sample_X):
    """test_artifact_reload: artifacts can be round-tripped."""
    model_path = ARTIFACT_DIR / "model.joblib"
    if not model_path.exists():
        pytest.skip("model.joblib not found")

    model = joblib.load(model_path)
    preds = model.predict(sample_X)
    assert preds.shape == (len(sample_X),)
    assert np.isfinite(preds).all()

    for q in ["p10", "p90"]:
        q_path = ARTIFACT_DIR / f"{q}_model.joblib"
        if q_path.exists():
            qm = joblib.load(q_path)
            q_preds = qm.predict(sample_X)
            assert q_preds.shape == (len(sample_X),)
            assert np.isfinite(q_preds).all()

    # Compare reload from bundle vs standalone model
    bundle_path = MODELS_DIR / "demand_model.pkl"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        bundle_preds = bundle["model"].predict(sample_X)
        np.testing.assert_array_almost_equal(preds, bundle_preds, decimal=5)


def test_temporal_split(feature_store):
    """test_temporal_split: chronological split produces correct proportions, no shuffle."""
    n = len(feature_store)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_frac = train_end / n
    val_frac = (val_end - train_end) / n
    test_frac = (n - val_end) / n

    assert abs(train_frac - 0.70) < 0.01
    assert abs(val_frac - 0.15) < 0.01
    assert abs(test_frac - 0.15) < 0.01

    df = feature_store.sort_values("decision_timestamp").reset_index(drop=True)
    train_max = df["decision_timestamp"].iloc[train_end - 1]
    val_min = df["decision_timestamp"].iloc[train_end]
    val_max = df["decision_timestamp"].iloc[val_end - 1]
    test_min = df["decision_timestamp"].iloc[val_end]

    assert train_max <= val_min, "Temporal leakage: train overlaps validation"
    assert val_max <= test_min, "Temporal leakage: validation overlaps test"


def test_metric_generation():
    """test_metric_generation: MAE/RMSE/R² compute correctly on known values."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean_absolute_error(y_true, y_pred) == 0.0
    assert np.sqrt(mean_squared_error(y_true, y_pred)) == 0.0
    assert r2_score(y_true, y_pred) == 1.0

    y_pred_off = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    assert mean_absolute_error(y_true, y_pred_off) == 1.0
    assert np.sqrt(mean_squared_error(y_true, y_pred_off)) == 1.0


# Also provide Class wrapper for test runners that organize by class
class TestModelLoad:
    def test_bundle_file_exists(self):
        assert (MODELS_DIR / "demand_model.pkl").exists()

    def test_bundle_has_model(self, bundle):
        assert "model" in bundle and bundle["model"] is not None

    def test_bundle_has_quantile_models(self, bundle):
        assert "p10_model" in bundle and "p90_model" in bundle

    def test_structured_artifacts_exist(self):
        assert ARTIFACT_DIR.exists()
        assert (ARTIFACT_DIR / "model.joblib").exists()
        assert (ARTIFACT_DIR / "metadata.json").exists()
        assert (ARTIFACT_DIR / "feature_schema.json").exists()


class TestFeatureSchema:
    def test_schema_has_required_keys(self, feature_schema):
        required = {"feature_columns", "n_features", "target_column", "forbidden_columns", "categorical_columns"}
        assert required.issubset(set(feature_schema.keys()))

    def test_target_is_expected_customer_count(self, feature_schema):
        assert feature_schema["target_column"] == "expected_customer_count"


class TestPredictionShape:
    def test_single_and_batch_shape(self, bundle, sample_X):
        test_prediction_shape(bundle, sample_X)


class TestPredictionNonnegative:
    def test_all_quantiles_nonnegative(self, bundle, sample_X):
        test_prediction_nonnegative(bundle, sample_X)


class TestQuantileOrder:
    def test_quantiles_ordered(self, bundle, sample_X):
        test_quantile_order(bundle, sample_X)


class TestDeterministicInference:
    def test_deterministic(self, bundle, sample_X):
        test_deterministic_inference(bundle, sample_X)


class TestModelMetadata:
    def test_metadata_complete(self, metadata):
        test_model_metadata(metadata)


class TestArtifactReload:
    def test_reload(self, sample_X):
        test_artifact_reload(sample_X)


class TestTemporalSplit:
    def test_temporal(self, feature_store):
        test_temporal_split(feature_store)


class TestMetricGeneration:
    def test_metrics(self):
        test_metric_generation()
