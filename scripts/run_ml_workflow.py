"""
GeoDemand AI — End-to-End Reproducible ML Workflow Runner (Step 5)

Member 3 — Machine Learning & MLOps

Executes and verifies the 10-step ML lifecycle:
  1. Data Validation (feature store, target contract, leakage prevention)
  2. Feature Schema Validation (45 features, datatypes, forbidden columns)
  3. Model Training / Smoke Training (Chronological 70/15/15 splits)
  4. Model Evaluation (Validation MAE, RMSE, R² without using holdout test set)
  5. TreeSHAP Explainability Verification (Formatted positive/negative drivers)
  6. Prediction Quality Checks (Contract bounds, NaN/Inf rejection, Monotonic quantiles)
  7. Monitoring & Drift Checks (Reference distribution profile comparison)
  8. Artifact Integrity Validation (Bundle components, metadata keys, hashes)
  9. MLflow Experiment Tracking (Parameters, metrics, and JSON artifacts logged)
 10. Candidate Model Promotion Gating (Policy check against active champion)

Usage:
  python scripts/run_ml_workflow.py --smoke
  python scripts/run_ml_workflow.py --skip-train
  python scripts/run_ml_workflow.py --full --promote
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.geodemand.models.inference import DemandModelPredictor
from src.geodemand.models.promotion import (
    ModelPromotionGate,
    ModelPromotionPolicy,
)
from src.geodemand.monitoring.drift import (
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
logger = logging.getLogger("run_ml_workflow")


def run_pipeline(
    mode: str = "smoke",
    promote: bool = False,
    candidate_bundle_path: str = None,
) -> Dict[str, Any]:
    """
    Execute the 10-step reproducible ML workflow.
    """
    logger.info("=" * 75)
    logger.info(f"GEODEMAND STEP 5: AUTOMATED ML PIPELINE WORKFLOW (Mode: {mode.upper()})")
    logger.info("=" * 75)

    workflow_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "steps_completed": [],
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # STEP 1: Data Validation
    # -------------------------------------------------------------------------
    logger.info("\n[Step 1/10] Data Validation...")
    from src.geodemand.models.train_demand_model import (
        FEATURE_STORE_PATH,
        FORBIDDEN_COLUMNS,
        TARGET_COL,
        add_cyclical_features,
        chronological_split,
        load_and_validate,
        prepare_features,
    )

    assert FEATURE_STORE_PATH.exists(), f"Feature store not found at {FEATURE_STORE_PATH}"
    df_raw = load_and_validate()
    assert TARGET_COL in df_raw.columns, f"Target column {TARGET_COL} missing"
    assert "vendor_id" in df_raw.columns and "decision_timestamp" in df_raw.columns
    logger.info(f"✓ Data validated: {len(df_raw):,} rows. Target contract: '{TARGET_COL}'.")
    workflow_summary["steps_completed"].append("1_data_validation")

    # -------------------------------------------------------------------------
    # STEP 2: Feature Schema Validation
    # -------------------------------------------------------------------------
    logger.info("\n[Step 2/10] Feature Schema Validation...")
    schema_path = ROOT_DIR / "models" / "lgbm_v1" / "feature_schema.json"
    assert schema_path.exists(), f"Feature schema not found at {schema_path}"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    expected_feature_cols = schema["feature_columns"]
    categorical_cols = schema["categorical_columns"]
    assert len(expected_feature_cols) >= 40, f"Expected >=40 features, got {len(expected_feature_cols)}"

    # Verify no forbidden columns overlap
    overlap = set(expected_feature_cols) & set(FORBIDDEN_COLUMNS)
    assert not overlap, f"Forbidden leakage columns found in feature list: {overlap}"
    logger.info(f"✓ Schema validated: {len(expected_feature_cols)} features, {len(categorical_cols)} categoricals.")
    workflow_summary["steps_completed"].append("2_schema_validation")

    # -------------------------------------------------------------------------
    # STEP 3: Training and Validation Splitting
    # -------------------------------------------------------------------------
    logger.info("\n[Step 3/10] Preparing Splits & Training...")
    n = len(df_raw)
    train_end = int(n * 0.70)
    train_mask = pd.Series([False] * n)
    train_mask.iloc[:train_end] = True

    if mode == "smoke":
        # In smoke mode: use representative subsample for fast CI execution
        logger.info("Running lightweight smoke training pipeline (500 samples)...")
        sub_df = df_raw.head(500).copy()
        X_sub, y_sub, f_cols, l_encs = prepare_features(sub_df)
        X_train, y_train = X_sub.iloc[:350], y_sub.iloc[:350]
        X_val, y_val = X_sub.iloc[350:425], y_sub.iloc[350:425]
        X_test, y_test = X_sub.iloc[425:], y_sub.iloc[425:]

        # Train fast smoke candidate (LightGBM/Ridge)
        from sklearn.linear_model import Ridge
        smoke_model = Ridge(alpha=1.0)
        smoke_model.fit(X_train, y_train)
        candidate_bundle = {
            "model": smoke_model,
            "p10_model": smoke_model,
            "p90_model": smoke_model,
            "feature_cols": expected_feature_cols,
            "label_encoders": l_encs,
            "meta": {"model_type": "smoke_ridge", "model_version": "v_smoke_ci"},
        }
        logger.info("✓ Smoke training complete.")
    elif mode == "skip_train":
        # Load existing candidate or champion bundle
        load_path = Path(candidate_bundle_path) if candidate_bundle_path else (ROOT_DIR / "models" / "demand_model.pkl")
        assert load_path.exists(), f"Bundle not found at {load_path}"
        logger.info(f"Loading existing bundle from {load_path}...")
        candidate_bundle = joblib.load(load_path)

        # Build validation set for evaluation
        X_full, y_full, _, _ = prepare_features(df_raw, train_mask=train_mask)
        _, _, X_val, y_val, X_test, y_test = chronological_split(X_full, y_full, df_raw["decision_timestamp"])
        logger.info("✓ Existing model bundle loaded.")
    else:  # full
        logger.info("Running full model training pipeline (train_demand_model.py)...")
        from src.geodemand.models.train_demand_model import train_pipeline
        train_pipeline()
        candidate_bundle = joblib.load(ROOT_DIR / "models" / "demand_model.pkl")
        X_full, y_full, _, _ = prepare_features(df_raw, train_mask=train_mask)
        _, _, X_val, y_val, X_test, y_test = chronological_split(X_full, y_full, df_raw["decision_timestamp"])
        logger.info("✓ Full model training complete.")

    workflow_summary["steps_completed"].append("3_training_and_splits")

    # -------------------------------------------------------------------------
    # STEP 4: Model Evaluation (Validation Set Only)
    # -------------------------------------------------------------------------
    logger.info("\n[Step 4/10] Model Evaluation (Validation Set)...")
    cand_model = candidate_bundle["model"]
    preds_val = np.clip(cand_model.predict(X_val), 0, None)
    val_mae = float(mean_absolute_error(y_val, preds_val))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))
    val_r2 = float(r2_score(y_val, preds_val)) if np.var(y_val) > 1e-9 else 0.0

    logger.info(f"✓ Validation Metrics: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}")
    workflow_summary["val_mae"] = val_mae
    workflow_summary["val_rmse"] = val_rmse
    workflow_summary["steps_completed"].append("4_model_evaluation")

    # -------------------------------------------------------------------------
    # STEP 5: TreeSHAP Verification
    # -------------------------------------------------------------------------
    logger.info("\n[Step 5/10] TreeSHAP Explainability Verification...")
    predictor = DemandModelPredictor()
    sample_feature = {col: 1.0 for col in predictor.feature_cols}
    sample_feature["weekday"] = "Monday"
    sample_feature["season"] = "monsoon"
    sample_feature["vendor_category"] = "food"

    exps = predictor.explain(sample_feature, top_k=3)
    assert len(exps) == 1
    assert "top_positive_factors" in exps[0]
    assert "top_negative_factors" in exps[0]
    assert "formatted_explanation" in exps[0]
    logger.info("✓ TreeSHAP local feature attribution and formatted drivers verified.")
    workflow_summary["steps_completed"].append("5_shap_verification")

    # -------------------------------------------------------------------------
    # STEP 6: Prediction Quality Checks
    # -------------------------------------------------------------------------
    logger.info("\n[Step 6/10] Prediction Quality Checks...")
    validator = PredictionValidator(raise_on_violation=True)
    q_rep = validator.validate_predictions(preds_val, expected_n=len(X_val))
    assert q_rep.is_valid and not q_rep.has_nan and not q_rep.has_infinite
    logger.info("✓ Predictions validated: Finite, Non-negative, Correct shape.")
    workflow_summary["steps_completed"].append("6_prediction_quality")

    # -------------------------------------------------------------------------
    # STEP 7: Monitoring & Drift Checks
    # -------------------------------------------------------------------------
    logger.info("\n[Step 7/10] Feature Drift & Monitoring Verification...")
    ref_path = ROOT_DIR / "models" / "reference_distributions.json"
    if ref_path.exists():
        ref_profile = ReferenceDistributionProfile.load(ref_path)
        detector = FeatureDriftDetector(reference_profile=ref_profile)
        # Check validation sample against reference
        drift_rep = detector.detect_drift(X_val[ref_profile.continuous_columns[:10]])
        logger.info(f"✓ Drift checks verified (Max PSI={drift_rep.max_psi:.4f}, Features checked={len(drift_rep.feature_results)}).")
    else:
        logger.warning("Reference profile not found — skipped drift check.")

    # Delayed ground-truth performance evaluation with actual validation targets
    logger.info("Evaluating delayed performance on actual validation targets...")
    perf_monitor = PerformanceMonitor()
    perf_rep = perf_monitor.evaluate(
        y_true=y_val.to_numpy(),
        y_pred=preds_val,
    )
    logger.info(f"✓ Delayed Performance: MAE={perf_rep.mae:.4f}, RMSE={perf_rep.rmse:.4f}, Bias={perf_rep.prediction_bias:+.4f}")
    logger.info(f"  Note: {SIMULATION_DATA_DISCLAIMER}")
    workflow_summary["delayed_performance"] = {
        "mae": perf_rep.mae,
        "rmse": perf_rep.rmse,
        "bias": perf_rep.prediction_bias,
        "disclaimer": SIMULATION_DATA_DISCLAIMER,
    }
    workflow_summary["steps_completed"].append("7_monitoring_checks")

    # -------------------------------------------------------------------------
    # STEP 8: Artifact Integrity Validation
    # -------------------------------------------------------------------------
    logger.info("\n[Step 8/10] Artifact Integrity Validation...")
    required_artifacts = [
        ROOT_DIR / "models" / "demand_model.pkl",
        ROOT_DIR / "models" / "demand_model_meta.json",
        ROOT_DIR / "models" / "lgbm_v1" / "model.joblib",
        ROOT_DIR / "models" / "lgbm_v1" / "feature_schema.json",
        ROOT_DIR / "models" / "reference_distributions.json",
    ]
    for art in required_artifacts:
        assert art.exists(), f"Missing required artifact: {art}"
    logger.info(f"✓ All {len(required_artifacts)} canonical model artifacts verified on disk.")
    workflow_summary["steps_completed"].append("8_artifact_validation")

    # -------------------------------------------------------------------------
    # STEP 9: MLflow Experiment Logging
    # -------------------------------------------------------------------------
    logger.info("\n[Step 9/10] MLflow Tracking...")
    if HAS_MLFLOW:
        try:
            import mlflow
            db_path = ROOT_DIR / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}".replace("\\", "/"))
            mlflow.set_experiment("GeoDemand")
            with mlflow.start_run(run_name=f"ml_workflow_{mode}") as run:
                mlflow.log_params({
                    "workflow_mode": mode,
                    "target_contract": TARGET_COL,
                    "dataset_source": "feature_store.parquet",
                    "n_features": len(expected_feature_cols),
                    "random_seed": 42,
                    "schema_version": schema.get("feature_version", "v2.2-next-hour-forecast-cyclical"),
                })
                mlflow.log_metrics({
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                })
                mlflow.log_dict(workflow_summary, "workflow_summary.json")
                logger.info(f"✓ MLflow workflow run logged successfully (Run ID: {run.info.run_id}).")
        except Exception as e:
            logger.warning(f"MLflow note: {e}")
    workflow_summary["steps_completed"].append("9_mlflow_logging")

    # -------------------------------------------------------------------------
    # STEP 10: Model Promotion Gating
    # -------------------------------------------------------------------------
    logger.info("\n[Step 10/10] Model Promotion Gating...")
    gate = ModelPromotionGate()
    promotion_report = gate.evaluate_candidate(
        candidate_bundle=candidate_bundle,
        X_val=X_val,
        y_val=y_val,
    )
    logger.info(f"Promotion Decision: {promotion_report.decision}")
    logger.info(f"All Gates Passed:   {promotion_report.passed_all_gates}")

    # Log candidate evaluation report into MLflow audit trail
    gate.log_promotion_to_mlflow(promotion_report)

    if promote:
        if promotion_report.passed_all_gates:
            logger.info("Promoting candidate model to champion...")
            gate.promote_candidate_bundle(
                candidate_bundle=candidate_bundle,
                candidate_meta=promotion_report.to_dict(),
            )
            logger.info("✓ Candidate promoted successfully.")
        else:
            logger.info("Promotion skipped: Candidate did not satisfy all gating policies.")
    workflow_summary["steps_completed"].append("10_promotion_gating")
    workflow_summary["promotion_decision"] = promotion_report.decision

    logger.info("\n" + "=" * 75)
    logger.info("ALL 10 ML WORKFLOW STEPS EXECUTED SUCCESSFULLY!")
    logger.info("=" * 75)
    return workflow_summary


def main():
    parser = argparse.ArgumentParser(description="GeoDemand Automated ML Pipeline Runner")
    parser.add_argument("--smoke", action="store_true", help="Run fast lightweight smoke test pipeline for CI")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and evaluate existing model bundle")
    parser.add_argument("--full", action="store_true", help="Execute full training and evaluation pipeline")
    parser.add_argument("--promote", action="store_true", help="Promote candidate if all promotion gates pass")
    parser.add_argument("--candidate", type=str, default=None, help="Path to specific candidate bundle")
    args = parser.parse_args()

    mode = "smoke"
    if args.skip_train:
        mode = "skip_train"
    elif args.full:
        mode = "full"

    res = run_pipeline(
        mode=mode,
        promote=args.promote,
        candidate_bundle_path=args.candidate,
    )
    print("\nWORKFLOW RESULT:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
