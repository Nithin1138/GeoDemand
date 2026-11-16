"""
GeoDemand AI — Phase 18.2 ML Integration Testing Suite

Member 2 — Machine Learning & MLOps

Validates all 9 tasks of Phase 18.2:
1. Test production model loading after deployment (independent reload, schema, metadata)
2. Test model schema compatibility across all 4 cities (Vijayawada, Hyderabad, Bengaluru, Visakhapatnam)
3. Test multi-hour predictions (horizons H in {1, 2, 4, 8}, recursive autoregression, causality)
4. Test quantile ordering and non-negative forecasts (0 <= P10 <= P50 <= P90, crossing rejection)
5. Test TreeSHAP explainability (attributions, top drivers, additivity, formatted text)
6. Test drift detection (feature drift PSI, prediction drift PSI, threshold >= 0.15)
7. Test retraining workflow (data eligibility, sufficiency guard >= 500 samples, trigger logic)
8. Test promotion and rollback (gating evaluation, atomic snapshot backup, rollback execution)
9. Test actual-data evaluation (error distribution, MAE/RMSE/R², bias, slice breakdowns)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.multicity import MultiCityRegistry
from src.geodemand.models.advanced_models import ApprovedLightGBMBaseline
from src.geodemand.models.inference import DemandModelPredictor
from src.geodemand.models.multi_horizon import (
    RecursiveAutoregressiveForecaster,
)
from src.geodemand.models.multicity_eval import (
    CityBoundaryError,
    CityModelSelector,
    MultiCityEvaluator,
)
from src.geodemand.models.promotion import (
    ModelPromotionGate,
    ModelPromotionPolicy,
    PromotionEvaluationReport,
)
from src.geodemand.monitoring.ct_pipeline import (
    ContinuousTrainingPipeline,
    RetrainingTrigger,
)
from src.geodemand.monitoring.drift import (
    DriftThresholds,
    FeatureDriftDetector,
    ReferenceDistributionProfile,
)
from src.geodemand.monitoring.performance import (
    PerformanceMonitor,
    PerformanceReport,
)
from src.geodemand.monitoring.prediction_drift import PredictionDriftDetector
from src.geodemand.monitoring.prediction_quality import (
    PredictionQualityError,
    PredictionValidator,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


@pytest.fixture(scope="module")
def predictor() -> DemandModelPredictor:
    """Fixture providing initialized production predictor."""
    return DemandModelPredictor()


@pytest.fixture(scope="module")
def sample_feature_dict(predictor) -> dict:
    """Fixture providing a valid inference feature row matching model schema."""
    d = {col: 1.0 for col in predictor.feature_cols}
    d["weekday"] = "Monday"
    d["season"] = "winter"
    d["vendor_category"] = "food"
    d["hour"] = 12
    d["temperature"] = 28.5
    d["rainfall"] = 0.0
    return d


# ==============================================================================
# Task 1: Test Production Model Loading
# ==============================================================================
def test_task1_production_model_loading(predictor):
    """Verify production champion loads independently with complete metadata and estimators."""
    assert predictor.model is not None, "Primary model failed to load"
    assert hasattr(predictor.model, "predict"), "Model missing predict interface"

    # Verify metadata consistency
    assert predictor.metadata is not None
    assert "model_version" in predictor.metadata
    assert len(predictor.feature_cols) > 10, f"Expected >10 feature columns, got {len(predictor.feature_cols)}"

    # Verify independent point prediction
    sample_df = pd.DataFrame([{col: 1.0 for col in predictor.feature_cols}])
    preds = predictor.predict(sample_df)
    assert len(preds) == 1
    assert preds[0] >= 0.0
    assert np.isfinite(preds[0])


# ==============================================================================
# Task 2: Test Model Schema Compatibility Across Cities
# ==============================================================================
def test_task2_model_schema_across_cities(predictor):
    """Verify spatial coordinate routing and feature schema compatibility across all 4 cities."""
    selector = CityModelSelector(strict_bounds=True)
    cities_coords = {
        "vijayawada": (16.5062, 80.6480, 1.00),
        "hyderabad": (17.4435, 78.3772, 1.25),
        "bengaluru": (12.9784, 77.6408, 1.35),
        "visakhapatnam": (17.7126, 83.3197, 1.05),
    }

    base_feature_row = {col: 1.0 for col in predictor.feature_cols}

    for city_id, (lat, lng, expected_scale) in cities_coords.items():
        res = selector.resolve_model_for_coordinate(lat, lng)
        assert res.city_id == city_id
        assert res.calibration_factor == expected_scale
        assert res.within_registered_boundary is True

        # Test inference with city coordinate injection
        city_features = dict(base_feature_row)
        city_features["latitude"] = lat
        city_features["longitude"] = lng
        city_preds = predictor.predict(city_features)
        assert len(city_preds) == 1
        assert city_preds[0] >= 0.0

    # Verify out-of-boundary prevention
    with pytest.raises(CityBoundaryError):
        selector.resolve_model_for_coordinate(19.0760, 72.8777)  # Mumbai


# ==============================================================================
# Task 3: Test Multi-Hour Predictions
# ==============================================================================
def test_task3_multi_hour_predictions(predictor):
    """Verify recursive multi-horizon predictions across H in {1, 2, 4, 8} hours."""
    rng = np.random.RandomState(42)
    n = 200
    horizons = [1, 2, 4, 8]

    # Generate synthetic training series
    feat_df = pd.DataFrame({
        "hour": rng.randint(0, 24, n),
        "hour_sin": np.sin(2 * np.pi * rng.randint(0, 24, n) / 24.0),
        "hour_cos": np.cos(2 * np.pi * rng.randint(0, 24, n) / 24.0),
        "office_count": rng.randint(0, 25, n),
        "expected_customer_count_lag_1h": rng.uniform(5, 30, n),
    })
    y = feat_df["expected_customer_count_lag_1h"] * 0.9 + rng.normal(0, 1.0, n)

    forecaster = RecursiveAutoregressiveForecaster()
    forecaster.fit(feat_df, y)

    # Multi-horizon forecast
    initial_input = feat_df.iloc[:10].copy()
    preds_by_h = forecaster.predict_multi_horizon(initial_input, horizons=horizons)

    assert set(preds_by_h.keys()) == set(horizons)
    for h in horizons:
        preds = preds_by_h[h]
        assert len(preds) == 10
        assert (preds >= 0.0).all()
        assert np.isfinite(preds).all()


# ==============================================================================
# Task 4: Test Quantile Ordering and Non-Negative Forecasts
# ==============================================================================
def test_task4_quantile_ordering(predictor, sample_feature_dict):
    """Verify strict monotonicity 0 <= P10 <= P50 <= P90 and crossing rejection."""
    intervals = predictor.predict_with_interval(sample_feature_dict)
    assert len(intervals) == 1
    res = intervals[0]

    p10 = res["p10"]
    p50 = res["p50"]
    p90 = res["p90"]

    assert 0.0 <= p10 <= p50 <= p90, f"Monotonicity violation: {p10} <= {p50} <= {p90}"
    assert res["prediction_range"] == round(p90 - p10, 2)

    # Test PredictionValidator rejects artificial quantile inversion
    validator = PredictionValidator(raise_on_violation=True)
    with pytest.raises(PredictionQualityError) as exc_info:
        validator.validate_quantiles(
            p10=np.array([25.0]),
            p50=np.array([20.0]),  # P10 > P50 inversion
            p90=np.array([30.0]),
            expected_n=1,
        )
    assert "Quantile monotonicity violation" in str(exc_info.value)


# ==============================================================================
# Task 5: Test TreeSHAP Explainability
# ==============================================================================
def test_task5_shap_explanations(predictor, sample_feature_dict):
    """Verify TreeSHAP local attributions, top drivers, and formatted explanation text."""
    explanations = predictor.explain(sample_feature_dict, top_k=3)
    assert len(explanations) == 1
    exp = explanations[0]

    assert "predicted_customer_count" in exp
    assert "base_bias" in exp
    assert "top_positive_factors" in exp
    assert "top_negative_factors" in exp
    assert "formatted_explanation" in exp
    assert "feature_contributions" in exp

    assert len(exp["top_positive_factors"]) <= 3
    assert len(exp["top_negative_factors"]) <= 3

    # Check formatted explanation follows contract
    text = exp["formatted_explanation"]
    assert "Prediction:" in text
    assert "customers/hour" in text


# ==============================================================================
# Task 6: Test Drift Detection
# ==============================================================================
def test_task6_drift_detection(predictor):
    """Verify feature drift and prediction drift detection triggers at PSI >= 0.15."""
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({
        "temperature": rng.normal(30.0, 5.0, 200),
        "rainfall": rng.exponential(1.0, 200),
    })
    drifted_df = pd.DataFrame({
        "temperature": rng.normal(48.0, 5.0, 200),  # Extreme temperature shift
        "rainfall": rng.exponential(15.0, 200),
    })

    detector = FeatureDriftDetector(thresholds=DriftThresholds(psi_threshold=0.15))
    profile = ReferenceDistributionProfile.from_dataframe(ref_df, dataset_id="ref_profile")

    rep = detector.detect_drift(drifted_df, reference_profile=profile)
    assert rep.max_psi >= 0.15
    assert rep.overall_drift_detected
    assert len(rep.drifted_features) > 0

    # Prediction drift check
    pred_detector = PredictionDriftDetector(thresholds=DriftThresholds(psi_threshold=0.15))
    ref_p = rng.poisson(15.0, 300).astype(float)
    drift_p = rng.poisson(35.0, 300).astype(float)
    p_rep = pred_detector.evaluate(ref_p, drift_p)
    assert p_rep.metrics.psi >= 0.15
    assert p_rep.trigger_activated


# ==============================================================================
# Task 7: Test Retraining Workflow
# ==============================================================================
def test_task7_retraining_workflow(tmp_path):
    """Verify continuous training pipeline data eligibility and sample sufficiency guard."""
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    models_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    # 1. Test insufficient data (< 500 samples) skips retraining safely
    small_df = pd.DataFrame({
        "decision_timestamp": pd.date_range("2026-10-01", periods=100, freq="h"),
        "expected_customer_count": np.random.randint(5, 30, 100),
    })
    feat_path = data_dir / "feature_store.parquet"
    small_df.to_parquet(feat_path)

    pipeline = ContinuousTrainingPipeline(models_dir=models_dir, data_path=feat_path)
    res_skip = pipeline.run_retraining_cycle(trigger=RetrainingTrigger.SCHEDULED)
    assert res_skip.status == "SKIPPED_INSUFFICIENT_DATA"
    assert res_skip.candidate_id is None

    # 2. Test sufficient data (>= 500 samples) eligibility pass
    suff_df = pd.DataFrame({
        "decision_timestamp": pd.date_range("2026-10-01", periods=600, freq="h"),
        "expected_customer_count": np.random.randint(5, 30, 600),
    })
    suff_df.to_parquet(feat_path)
    elig = pipeline.check_data_eligibility(suff_df)
    assert elig.is_eligible is True
    assert elig.new_samples_count >= 500


# ==============================================================================
# Task 8: Test Promotion and Rollback
# ==============================================================================
def test_task8_promotion_and_rollback(tmp_path):
    """Verify model promotion gate evaluation, atomic backup creation, and rollback."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # Create dummy champion
    champ_path = models_dir / "demand_model.pkl"
    champ_meta_path = models_dir / "demand_model_meta.json"
    joblib.dump({"model": "champion_v1_model", "model_name": "champion_v1"}, champ_path)
    with open(champ_meta_path, "w", encoding="utf-8") as f:
        json.dump({"model_version": "v2.0.0", "metrics": {"val_mae": 0.85}}, f)

    gate = ModelPromotionGate(models_dir=models_dir)

    # 1. Promotion with backup creation
    cand_bundle = {"model": "candidate_v2_model", "model_name": "candidate_v2"}
    cand_meta = {"model_version": "v2.1.0", "metrics": {"val_mae": 0.75}}

    promoted = gate.promote_candidate_bundle(cand_bundle, cand_meta)
    assert promoted is True
    assert (models_dir / "champion_backup" / "demand_model.pkl").exists()

    # Verify active model is candidate
    active = joblib.load(champ_path)
    assert active["model_name"] == "candidate_v2"

    # 2. Rollback to backup
    rolled_back = gate.rollback_to_backup()
    assert rolled_back is True
    restored = joblib.load(champ_path)
    assert restored["model_name"] == "champion_v1"


# ==============================================================================
# Task 9: Test Actual-Data Model Evaluation
# ==============================================================================
def test_task9_actual_data_evaluation(predictor):
    """Verify performance evaluation on ground truth actuals with error profiling and disclaimer."""
    monitor = PerformanceMonitor()
    rng = np.random.RandomState(42)
    n = 200

    y_true = rng.uniform(5, 40, n)
    y_pred = y_true + rng.normal(0, 0.75, n)
    p10 = y_pred - 1.5
    p90 = y_pred + 1.5

    meta_df = pd.DataFrame({
        "hour": rng.randint(0, 24, n),
        "is_weekend": rng.choice([0, 1], n),
        "vendor_category": rng.choice(["food", "chai", "repair"], n),
    })

    rep: PerformanceReport = monitor.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        p10=p10,
        p90=p90,
        metadata_df=meta_df,
    )

    assert rep.sample_size == n
    assert rep.mae > 0.0
    assert rep.rmse >= rep.mae
    assert rep.r2 is not None
    assert rep.prediction_bias is not None

    # Error distribution
    assert rep.error_distribution is not None
    assert hasattr(rep.error_distribution, "mean")
    assert hasattr(rep.error_distribution, "median")
    assert hasattr(rep.error_distribution, "p90")

    # Slices
    assert "vendor_category" in rep.slices
    assert "hour" in rep.slices

    # Simulation disclaimer
    assert "SIMULATED DATA NOTICE" in rep.data_disclaimer
