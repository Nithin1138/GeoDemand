"""
GeoDemand AI — Unit Tests for ML Monitoring & Model Quality (Phase 6)

Member 3 — Machine Learning & MLOps

Tests:
1. test_no_drift_identical_distributions
2. test_continuous_drift_detection_ks_and_psi
3. test_categorical_drift_detection
4. test_schema_missing_and_unexpected_features
5. test_schema_invalid_numerical_values
6. test_prediction_quality_nan_and_infinite
7. test_prediction_quality_output_shapes
8. test_prediction_quality_quantile_ordering
9. test_performance_metric_calculations
10. test_monitoring_small_datasets
11. test_monitoring_missing_values
12. test_mlflow_monitoring_integration
13. test_practical_significance_filtering
14. test_multiple_comparison_corrections
15. test_performance_monitor_slice_breakdowns
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.geodemand.monitoring.drift import (
    DriftReport,
    DriftThresholds,
    FeatureDriftDetector,
    FeatureDriftResult,
    ReferenceDistributionProfile,
    SchemaValidationResult,
)
from src.geodemand.monitoring.mlflow_logger import (
    HAS_MLFLOW,
    MLflowMonitoringLogger,
)
from src.geodemand.monitoring.performance import (
    PerformanceMonitor,
    PerformanceReport,
    SIMULATION_DATA_DISCLAIMER,
)
from src.geodemand.monitoring.prediction_quality import (
    PredictionQualityError,
    PredictionQualityReport,
    PredictionValidator,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Synthetic Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_reference_df():
    """Generates a synthetic reference dataset with known continuous and categorical distributions."""
    rng = np.random.RandomState(42)
    n = 1000
    return pd.DataFrame({
        "temperature": rng.normal(loc=25.0, scale=5.0, size=n),
        "humidity": rng.uniform(low=30.0, high=90.0, size=n),
        "office_count": rng.poisson(lam=10.0, size=n).astype(float),
        "commercial_ratio": rng.beta(a=2.0, b=5.0, size=n),
        "weekday": rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], size=n),
        "season": rng.choice(["summer", "monsoon", "winter"], size=n, p=[0.4, 0.4, 0.2]),
        "vendor_category": rng.choice(["food", "fruits", "repair", "salon"], size=n, p=[0.5, 0.2, 0.15, 0.15]),
    })


@pytest.fixture
def detector(synthetic_reference_df):
    """Initializes FeatureDriftDetector with synthetic reference profile."""
    profile = ReferenceDistributionProfile.from_dataframe(
        df=synthetic_reference_df,
        dataset_id="synthetic_test_ref",
        categorical_columns=["weekday", "season", "vendor_category"],
        sample_size_for_ks=1000,
        random_seed=42,
    )
    thresholds = DriftThresholds(
        psi_threshold=0.20,
        psi_moderate_threshold=0.10,
        ks_stat_threshold=0.10,
        ks_alpha=0.05,
    )
    return FeatureDriftDetector(
        reference_profile=profile,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Phase 6 Automated Tests
# ---------------------------------------------------------------------------

def test_no_drift_identical_distributions(detector, synthetic_reference_df):
    """Test that comparing identical distributions detects zero drift."""
    # Use exact reference data
    report = detector.detect_drift(synthetic_reference_df)

    assert isinstance(report, DriftReport)
    assert report.schema_validation.is_valid
    assert not report.overall_drift_detected
    assert len(report.drifted_features) == 0
    # PSI should be practically 0 on identical data
    assert report.max_psi < 0.05
    for feat, res in report.feature_results.items():
        assert not res.drift_detected
        assert res.psi < 0.05


def test_continuous_drift_detection_ks_and_psi(detector, synthetic_reference_df):
    """Test that a substantial distribution shift is correctly identified by KS and PSI."""
    shifted_df = synthetic_reference_df.copy()
    # Massive temperature shift: mean 25 -> 45
    shifted_df["temperature"] = shifted_df["temperature"] + 20.0

    report = detector.detect_drift(shifted_df)

    assert "temperature" in report.drifted_features
    temp_res = report.feature_results["temperature"]
    assert temp_res.drift_detected
    assert temp_res.stat_name == "ks"
    assert temp_res.statistic > 0.3  # Large KS distance
    assert temp_res.psi > 0.20  # Significant PSI
    assert temp_res.p_value_adjusted < 0.001
    assert report.overall_drift_detected


def test_categorical_drift_detection(detector, synthetic_reference_df):
    """Test that a drastic shift in categorical frequencies triggers drift detection."""
    shifted_df = synthetic_reference_df.copy()
    # In reference: food=50%, fruits=20%, repair=15%, salon=15%
    # Shifted: 100% salon
    shifted_df["vendor_category"] = "salon"

    report = detector.detect_drift(shifted_df)

    assert "vendor_category" in report.drifted_features
    cat_res = report.feature_results["vendor_category"]
    assert cat_res.drift_detected
    assert cat_res.feature_type == "categorical"
    assert cat_res.psi > 0.20
    assert report.overall_drift_detected


def test_schema_missing_and_unexpected_features(detector, synthetic_reference_df):
    """Test detection of missing required features and unexpected non-whitelisted columns."""
    corrupted_df = synthetic_reference_df.copy()
    # Drop required feature
    corrupted_df = corrupted_df.drop(columns=["temperature"])
    # Add unexpected non-whitelisted feature
    corrupted_df["unexpected_leakage_feature"] = 123.45
    # Add allowed metadata column
    corrupted_df["h3_cell_id"] = "88619aa6c3fffff"

    res = detector.validate_schema(corrupted_df)

    assert not res.is_valid
    assert "temperature" in res.missing_features
    assert "unexpected_leakage_feature" in res.unexpected_features
    assert "h3_cell_id" not in res.unexpected_features  # Whitelisted


def test_schema_invalid_numerical_values(detector, synthetic_reference_df):
    """Test detection of NaN, infinite, and illegal negative values in physical features."""
    invalid_df = synthetic_reference_df.copy()
    # Inject NaN
    invalid_df.loc[0, "humidity"] = np.nan
    # Inject Infinite
    invalid_df.loc[1, "humidity"] = np.inf
    # Inject negative value into strictly non-negative column
    invalid_df.loc[2, "office_count"] = -15.0

    res = detector.validate_schema(invalid_df)

    assert not res.is_valid
    assert "humidity" in res.null_counts
    assert "humidity" in res.infinite_counts
    assert "office_count" in res.negative_value_counts


def test_prediction_quality_nan_and_infinite():
    """Test that NaN and Infinite model outputs are rejected by PredictionValidator."""
    validator = PredictionValidator(raise_on_violation=True)

    # NaN output
    preds_nan = np.array([12.0, np.nan, 30.0])
    with pytest.raises(PredictionQualityError) as exc_nan:
        validator.validate_predictions(preds_nan, expected_n=3)
    assert "NaN predictions" in str(exc_nan.value)

    # Infinite output
    preds_inf = np.array([12.0, np.inf, 30.0])
    with pytest.raises(PredictionQualityError) as exc_inf:
        validator.validate_predictions(preds_inf, expected_n=3)
    assert "infinite predictions" in str(exc_inf.value)


def test_prediction_quality_output_shapes():
    """Test that mismatch between input batch size and output array raises an error."""
    validator = PredictionValidator(raise_on_violation=True)
    preds = np.array([10.0, 20.0])  # Length 2

    # Expecting length 5
    with pytest.raises(PredictionQualityError) as exc:
        validator.validate_predictions(preds, expected_n=5)
    assert "shape mismatch" in str(exc.value)


def test_prediction_quality_quantile_ordering():
    """Test that quantile inversion P10 > P50 or P50 > P90 is detected and rejected."""
    validator = PredictionValidator(raise_on_violation=True)

    # Valid quantiles
    p10_valid = np.array([5.0, 10.0])
    p50_valid = np.array([10.0, 15.0])
    p90_valid = np.array([15.0, 25.0])
    rep = validator.validate_quantiles(p10_valid, p50_valid, p90_valid, expected_n=2)
    assert rep.is_valid
    assert not rep.has_quantile_inversion

    # Inverted P10 > P50
    p10_inverted = np.array([12.0, 10.0])  # 12.0 > 10.0
    with pytest.raises(PredictionQualityError) as exc_1:
        validator.validate_quantiles(p10_inverted, p50_valid, p90_valid, expected_n=2)
    assert "P10 > P50" in str(exc_1.value)

    # Inverted P50 > P90
    p90_inverted = np.array([8.0, 25.0])  # 8.0 < 10.0
    with pytest.raises(PredictionQualityError) as exc_2:
        validator.validate_quantiles(p10_valid, p50_valid, p90_inverted, expected_n=2)
    assert "P50 > P90" in str(exc_2.value)


def test_performance_metric_calculations():
    """Test exact mathematical correctness of MAE, RMSE, R², Bias, and interval coverage."""
    monitor = PerformanceMonitor()

    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 39.0])
    # Errors: [+2.0, -2.0, +3.0, -1.0]
    # Abs errors: [2, 2, 3, 1] -> MAE = 2.0
    # Sq errors: [4, 4, 9, 1] = 18 / 4 = 4.5 -> RMSE = sqrt(4.5) ~ 2.1213
    # Bias = (+2 - 2 + 3 - 1) / 4 = 2 / 4 = +0.5

    p10 = np.array([8.0, 15.0, 25.0, 35.0])
    p90 = np.array([15.0, 22.0, 35.0, 42.0])
    # Actuals in [p10, p90]: all 4 fall inside! Coverage = 1.0 (100%)
    # Widths: [7, 7, 10, 7] -> Mean width = 31 / 4 = 7.75

    rep = monitor.evaluate(y_true, y_pred, p10=p10, p90=p90)

    assert rep.sample_size == 4
    assert abs(rep.mae - 2.0) < 1e-4
    assert abs(rep.rmse - np.sqrt(4.5)) < 1e-4
    assert abs(rep.prediction_bias - 0.5) < 1e-4
    assert rep.r2 is not None and rep.r2 > 0.95
    assert rep.quantile_coverage_80 == 1.0
    assert abs(rep.mean_interval_width - 7.75) < 1e-4
    assert rep.data_disclaimer == SIMULATION_DATA_DISCLAIMER


def test_performance_monitor_slice_breakdowns():
    """Test performance calculation across contextual cohorts/slices."""
    monitor = PerformanceMonitor()

    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([10.0, 20.0, 35.0, 45.0])
    # First 2 are weekday (errors 0, 0 -> MAE 0)
    # Next 2 are weekend (errors 5, 5 -> MAE 5)
    meta_df = pd.DataFrame({
        "hour": [12, 12, 19, 19],
        "is_weekend": [0, 0, 1, 1],
        "vendor_category": ["food", "food", "fruits", "fruits"],
    })

    rep = monitor.evaluate(y_true, y_pred, metadata_df=meta_df)

    assert "is_weekend" in rep.slices
    assert "vendor_category" in rep.slices
    assert "hour" in rep.slices

    weekend_slices = {s.slice_value: s for s in rep.slices["is_weekend"]}
    assert "0" in weekend_slices and "1" in weekend_slices
    assert weekend_slices["0"].mae == 0.0
    assert weekend_slices["1"].mae == 5.0


def test_monitoring_small_datasets(detector):
    """Test that monitoring very small sample sizes (N=3) runs safely without crashing."""
    small_df = pd.DataFrame({
        "temperature": [24.0, 25.0, 26.0],
        "humidity": [50.0, 55.0, 60.0],
        "office_count": [10.0, 11.0, 12.0],
        "commercial_ratio": [0.3, 0.4, 0.35],
        "weekday": ["Monday", "Tuesday", "Wednesday"],
        "season": ["summer", "summer", "summer"],
        "vendor_category": ["food", "fruits", "food"],
    })

    report = detector.detect_drift(small_df)
    assert isinstance(report, DriftReport)
    assert report.n_current_samples == 3
    assert report.schema_validation.is_valid


def test_monitoring_missing_values(detector, synthetic_reference_df):
    """Test that missing value rates above threshold trigger schema warnings."""
    null_df = synthetic_reference_df.copy()
    # Inject 10% nulls in humidity (threshold is 5%)
    null_df.loc[:100, "humidity"] = np.nan

    res = detector.validate_schema(null_df)
    assert not res.is_valid
    assert "humidity" in res.null_counts
    assert any("humidity" in err for err in res.errors)


def test_practical_significance_filtering(detector):
    """
    Test Requirement 8: Avoid treating statistical significance alone as proof of drift.
    With large N, a tiny shift in mean (e.g. 0.01 deg) might have p < 0.05,
    but KS stat and PSI will be tiny, so it should NOT be flagged as practical drift.
    """
    rng = np.random.RandomState(999)
    # Generate 5,000 samples with virtually identical distribution
    n = 5000
    subtle_df = pd.DataFrame({
        "temperature": rng.normal(loc=25.02, scale=5.0, size=n),  # tiny 0.02 shift
        "humidity": rng.uniform(low=30.0, high=90.0, size=n),
        "office_count": rng.poisson(lam=10.0, size=n).astype(float),
        "commercial_ratio": rng.beta(a=2.0, b=5.0, size=n),
        "weekday": rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], size=n),
        "season": rng.choice(["summer", "monsoon", "winter"], size=n, p=[0.4, 0.4, 0.2]),
        "vendor_category": rng.choice(["food", "fruits", "repair", "salon"], size=n, p=[0.5, 0.2, 0.15, 0.15]),
    })

    report = detector.detect_drift(subtle_df)
    temp_res = report.feature_results["temperature"]

    # Even if p-value is somewhat small, effect size / KS stat < 0.10 and PSI < 0.20
    assert temp_res.effect_size < 0.10
    assert temp_res.psi < 0.10
    assert not temp_res.drift_detected  # Correctly filtered out by practical significance filter


def test_multiple_comparison_corrections():
    """Test Bonferroni and Benjamini-Hochberg multiple comparisons corrections."""
    p_values = [0.001, 0.01, 0.03, 0.05, 0.20]

    # Bonferroni: multiply by m=5
    bonf = FeatureDriftDetector.adjust_p_values(p_values, method="bonferroni")
    expected_bonf = [0.005, 0.05, 0.15, 0.25, 1.0]
    np.testing.assert_array_almost_equal(bonf, expected_bonf)

    # Benjamini-Hochberg: monotonic FDR
    bh = FeatureDriftDetector.adjust_p_values(p_values, method="benjamini-hochberg")
    assert all(a >= p for a, p in zip(bh, p_values))
    assert all(bh[i] <= bh[i + 1] for i in range(len(bh) - 1))


def test_mlflow_monitoring_integration(detector, synthetic_reference_df):
    """Test MLflow monitoring logger records parameters, metrics, and JSON artifacts."""
    mlflow_logger = MLflowMonitoringLogger(experiment_name="GeoDemand_Test")
    report = detector.detect_drift(synthetic_reference_df)

    # Test logging executes without raising exceptions
    run_id = mlflow_logger.log_drift_report(report, run_name="test_monitoring_run")
    if HAS_MLFLOW:
        assert run_id is not None or not mlflow_logger._configured

    # Test performance logging executes without raising
    perf_mon = PerformanceMonitor()
    perf_rep = perf_mon.evaluate(np.array([10.0, 20.0]), np.array([11.0, 19.0]))
    perf_run_id = mlflow_logger.log_performance_report(perf_rep, run_name="test_perf_run")
    if HAS_MLFLOW:
        assert perf_run_id is not None or not mlflow_logger._configured
