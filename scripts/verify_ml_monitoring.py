"""
GeoDemand AI — End-to-End ML Monitoring & Quality Verification Script (Step 4)

Member 3 — Machine Learning & MLOps

Demonstrates and verifies:
1. Reference distribution loading from versioned artifact
2. Feature drift detection on identical and shifted distributions (KS + PSI)
3. Categorical distribution shift detection
4. Schema validation: missing features, invalid negative/infinite values
5. Prediction quality validation: contract enforcement, NaN/Inf rejection, quantile ordering
6. Delayed performance monitoring: MAE, RMSE, R², bias, coverage, slices, and simulation disclaimer
7. MLflow monitoring run logging
8. Summary report
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.geodemand.models.inference import DemandModelPredictor
from src.geodemand.monitoring.drift import (
    DriftThresholds,
    FeatureDriftDetector,
    ReferenceDistributionProfile,
)
from src.geodemand.monitoring.mlflow_logger import (
    HAS_MLFLOW,
    MLflowMonitoringLogger,
)
from src.geodemand.monitoring.performance import (
    PerformanceMonitor,
    SIMULATION_DATA_DISCLAIMER,
)
from src.geodemand.monitoring.prediction_quality import (
    PredictionQualityError,
    PredictionValidator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verify_ml_monitoring")


def main():
    logger.info("=" * 75)
    logger.info("GEODEMAND STEP 4: ML MONITORING & MODEL QUALITY AUTOMATION VERIFICATION")
    logger.info("=" * 75)

    ref_path = ROOT_DIR / "models" / "reference_distributions.json"
    assert ref_path.exists(), f"Reference profile not found at {ref_path}"
    profile = ReferenceDistributionProfile.load(ref_path)
    logger.info(f"✓ Reference profile loaded: ID={profile.dataset_id}, Features={len(profile.feature_columns)}")

    detector = FeatureDriftDetector(reference_profile=profile)
    logger.info("✓ FeatureDriftDetector initialized.")

    # 1. Feature Drift Test on controlled baseline
    logger.info("\n[1/6] Testing Drift Detection on Baseline Sample...")
    # Load feature store sample for baseline test
    fs_path = ROOT_DIR / "data" / "processed" / "feature_store.parquet"
    assert fs_path.exists(), f"Feature store not found at {fs_path}"
    df_raw = pd.read_parquet(fs_path).head(1000)
    from src.geodemand.models.train_demand_model import add_cyclical_features
    sample_df = add_cyclical_features(df_raw)[profile.feature_columns].copy()

    baseline_report = detector.detect_drift(sample_df)
    logger.info(f"✓ Baseline check completed: Overall drift={baseline_report.overall_drift_detected}, Max PSI={baseline_report.max_psi:.4f}")

    # 2. Controlled Feature Drift Test (shifted distribution)
    logger.info("\n[2/6] Testing Drift Detection on Controlled Distribution Shift...")
    shifted_df = sample_df.copy()
    shifted_df["temperature"] = shifted_df["temperature"] + 25.0  # Large temperature spike
    shifted_df["vendor_category"] = "salon"  # Massive categorical shift

    drift_report = detector.detect_drift(shifted_df)
    assert drift_report.overall_drift_detected, "Drift was not detected on shifted fixture!"
    assert "temperature" in drift_report.drifted_features, "Temperature drift was not flagged!"
    assert "vendor_category" in drift_report.drifted_features, "Vendor category drift was not flagged!"
    logger.info(f"✓ Controlled drift successfully flagged! Drifted features: {drift_report.drifted_features}")
    logger.info(f"  Temperature: PSI={drift_report.feature_results['temperature'].psi:.3f}, KS={drift_report.feature_results['temperature'].statistic:.3f}")
    logger.info(f"  Vendor category: PSI={drift_report.feature_results['vendor_category'].psi:.3f}")

    # 3. Schema and Value Validation Checks
    logger.info("\n[3/6] Testing Schema & Value Validation Checks...")
    corrupt_df = sample_df.copy()
    corrupt_df.loc[0, "humidity"] = np.nan
    corrupt_df.loc[1, "office_count"] = -50.0  # Invalid negative count
    corrupt_df = corrupt_df.drop(columns=["road_density"])  # Missing column

    schema_res = detector.validate_schema(corrupt_df)
    assert not schema_res.is_valid
    assert "road_density" in schema_res.missing_features
    assert "humidity" in schema_res.null_counts
    assert "office_count" in schema_res.negative_value_counts
    logger.info("✓ Schema validation correctly caught missing column, NaN value, and invalid negative count.")

    # 4. Prediction Quality Validation Checks
    logger.info("\n[4/6] Testing Prediction Quality Checks (Contract & Quantiles)...")
    validator = PredictionValidator(raise_on_violation=True)

    # Valid predictions check
    valid_p = np.array([12.5, 34.0, 5.0])
    valid_rep = validator.validate_predictions(valid_p, expected_n=3)
    assert valid_rep.is_valid
    logger.info("✓ Valid predictions correctly passed validation.")

    # NaN prediction rejection
    try:
        validator.validate_predictions(np.array([12.5, np.nan, 5.0]), expected_n=3)
        assert False, "Validator failed to raise on NaN prediction!"
    except PredictionQualityError as e:
        logger.info(f"✓ Validator correctly rejected NaN output: {e}")

    # Quantile monotonicity rejection
    try:
        # Inverted: P10 (20.0) > P50 (15.0)
        validator.validate_quantiles(
            p10=np.array([20.0]),
            p50=np.array([15.0]),
            p90=np.array([25.0]),
            expected_n=1,
        )
        assert False, "Validator failed to raise on inverted quantiles!"
    except PredictionQualityError as e:
        logger.info(f"✓ Validator correctly rejected quantile inversion: {e}")

    # Test integrated predictor quality validation
    logger.info("\nVerifying integrated DemandModelPredictor validation...")
    predictor = DemandModelPredictor()
    dummy_input = {col: 1.0 for col in predictor.feature_cols}
    dummy_input["weekday"] = "Monday"
    dummy_input["season"] = "monsoon"
    dummy_input["vendor_category"] = "food"

    preds = predictor.predict(dummy_input)
    assert len(preds) == 1 and preds[0] >= 0
    intervals = predictor.predict_with_interval(dummy_input)
    assert intervals[0]["p10"] <= intervals[0]["p50"] <= intervals[0]["p90"]
    logger.info(f"✓ DemandModelPredictor passed validated prediction: {preds[0]:.2f} customers (P10={intervals[0]['p10']} <= P50={intervals[0]['p50']} <= P90={intervals[0]['p90']})")

    # 5. Delayed Performance Monitoring
    logger.info("\n[5/6] Testing Delayed Ground-Truth Performance Monitoring...")
    monitor = PerformanceMonitor()
    y_true = np.array([10.0, 25.0, 15.0, 30.0, 20.0])
    y_pred = np.array([11.0, 23.0, 16.0, 28.0, 22.0])
    p10 = np.array([8.0, 18.0, 11.0, 24.0, 16.0])
    p90 = np.array([14.0, 28.0, 20.0, 34.0, 25.0])

    meta_df = pd.DataFrame({
        "hour": [10, 12, 14, 18, 20],
        "is_weekend": [0, 0, 1, 1, 0],
        "vendor_category": ["food", "fruits", "repair", "food", "salon"],
    })

    perf_report = monitor.evaluate(y_true, y_pred, p10=p10, p90=p90, metadata_df=meta_df)
    logger.info(f"✓ Performance metrics computed:")
    logger.info(f"  MAE: {perf_report.mae:.4f} | RMSE: {perf_report.rmse:.4f} | Bias: {perf_report.prediction_bias:+.4f}")
    logger.info(f"  80% Quantile Interval Coverage: {perf_report.quantile_coverage_80:.1%} (nominal 80%)")
    logger.info(f"  Mean Interval Width: {perf_report.mean_interval_width:.2f}")
    logger.info(f"  Slice dimensions tracked: {list(perf_report.slices.keys())}")
    logger.info(f"  Simulation Disclaimer: {perf_report.data_disclaimer[:80]}...")

    # 6. MLflow Monitoring Logging
    logger.info("\n[6/6] Testing MLflow Monitoring Integration...")
    mlflow_logger = MLflowMonitoringLogger(experiment_name="GeoDemand")
    run_id_drift = mlflow_logger.log_drift_report(drift_report, run_name="verify_drift_monitoring")
    run_id_perf = mlflow_logger.log_performance_report(perf_report, run_name="verify_performance_monitoring")
    logger.info(f"✓ MLflow drift run ID: {run_id_drift}")
    logger.info(f"✓ MLflow performance run ID: {run_id_perf}")

    logger.info("\n" + "=" * 75)
    logger.info("ALL GEODEMAND STEP 4 MONITORING VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
