"""
GeoDemand AI — ML Model Training, Quantile Calibration & Explainability Pipeline

Member 3 — Machine Learning & MLOps

Phases:
  - Benchmark: Multi-Model Benchmark with chronological split
    (Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost)
    evaluated on MAE, RMSE, R² using validation set for selection, holdout test for final eval.
  - Quantile Regression: Rigorous prediction intervals (P10, P50, P90).
  - TreeSHAP: Native TreeSHAP explainability for local positive & negative drivers.
  - MLflow: Comprehensive experiment tracking and artifact logging.

Data contract:
  - Source: data/processed/feature_store.parquet (canonical v2.1 training dataset)
  - Target: expected_customer_count (representing next-hour customer count)
  - Temporal contract: features at decision_timestamp (t) → predict target at t+1h
  - Split: chronological 70% train / 15% validation / 15% test (strictly no shuffle)

NOTE: All results are based on SIMULATED data from the GeoDemand market simulator.
Do not interpret metrics as real-world accuracy guarantees.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_STORE_PATH = PROCESSED_DIR / "feature_store.parquet"
MODELS_DIR = ROOT / "models"
ARTIFACT_DIR = MODELS_DIR / "lgbm_v1"

# ---------------------------------------------------------------------------
# Contract constants (Section 24: ML Target)
# ---------------------------------------------------------------------------
# Primary target contract: expected_customer_count_next_hour
PRIMARY_TARGET_CONTRACT = "expected_customer_count_next_hour"
TARGET_COL = "expected_customer_count"
RANDOM_SEED = 42

# Strict prohibition per Section 24: Never change target to profit/revenue/decision/demand score
DISALLOWED_TARGETS = {
    "profit",
    "revenue",
    "decision",
    "demand score",
    "demand_score",
    "recommendation_score",
}

# Identity / temporal / target / business decision columns — NEVER used as trainable features
FORBIDDEN_COLUMNS = {
    "vendor_id",
    "h3_cell_id",
    "decision_timestamp",
    "target_timestamp",
    "customer_count",
    "expected_customer_count",
    "date",
    "timestamp",
    # Business outcomes (must be calculated downstream, never learned by ML)
    "profit",
    "revenue",
    "decision",
    "demand_score",
    "recommendation_score",
    "operating_cost",
    "fuel_cost",
}

# Categorical columns that need label encoding for tree models
CATEGORICAL_COLUMNS = ["weekday", "season", "vendor_category"]

# Zero-variance columns to drop (weather_condition has only 'clear')
DROP_COLUMNS = {"weather_condition"}


def _file_hash(path: Path) -> str:
    """SHA-256 of the first 1MB of a file for version tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def load_and_validate() -> pd.DataFrame:
    """Load feature_store.parquet and run contract validations."""
    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Canonical training dataset not found: {FEATURE_STORE_PATH}\n"
            "Run the data pipeline first (Member 2)."
        )

    logger.info(f"Loading training data from {FEATURE_STORE_PATH}")
    df = pd.read_parquet(FEATURE_STORE_PATH)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

    # --- Validate target (Section 24: ML Target) ---
    if TARGET_COL in DISALLOWED_TARGETS:
        raise ValueError(
            f"Architecture violation (Section 24): Target cannot be '{TARGET_COL}'. "
            "ML contract must remain customer-count prediction (expected_customer_count_next_hour), "
            "never profit, revenue, decision, or demand score without an explicit architecture review."
        )
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}\n"
            "Contract violation: ML target must be customer-count prediction."
        )

    # --- Validate timestamps ---
    for ts_col in ["decision_timestamp", "target_timestamp"]:
        if ts_col not in df.columns:
            raise ValueError(f"Required timestamp column '{ts_col}' not found.")

    # --- Check for unexpected leakage columns ---
    leakage_cols = {
        "hidden_skill_factor",
        "popularity_score",
        "quality_score",
        "repeat_customer_rate",
        "cell_reputation",
    }
    leaked = leakage_cols.intersection(df.columns)
    if leaked:
        raise ValueError(f"Leakage columns detected in feature store: {leaked}")

    # --- Check nulls ---
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        raise ValueError(f"Null values found in training data:\n{null_cols}")

    # --- Check duplicate keys ---
    if "vendor_id" in df.columns and "decision_timestamp" in df.columns:
        n_dups = df.duplicated(subset=["vendor_id", "decision_timestamp"]).sum()
        if n_dups > 0:
            logger.warning(f"Found {n_dups} duplicate (vendor_id, decision_timestamp) rows.")

    # --- Sort chronologically ---
    df = df.sort_values("decision_timestamp").reset_index(drop=True)
    logger.info(
        f"Data sorted chronologically by decision_timestamp: "
        f"{df['decision_timestamp'].iloc[0]} → {df['decision_timestamp'].iloc[-1]}"
    )

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, LabelEncoder]]:
    """
    Prepare feature matrix X and target y from the feature store.

    - Drops forbidden columns (identity, timestamps, target)
    - Drops zero-variance columns
    - Label-encodes categorical string columns for model compatibility
    - Converts booleans to int
    - Returns (X, y, feature_columns, label_encoders)
    """
    df = df.copy()

    # Extract target
    y = df[TARGET_COL].astype(float)

    # Determine trainable columns: everything except forbidden and zero-variance
    trainable = [
        c for c in df.columns if c not in FORBIDDEN_COLUMNS and c not in DROP_COLUMNS
    ]

    # Label-encode categoricals
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        if col in trainable and (df[col].dtype == object or not pd.api.types.is_numeric_dtype(df[col])):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Label-encoded '{col}': {list(le.classes_)}")

    # Convert booleans to int
    for col in trainable:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Build X
    feature_cols = sorted(trainable)
    X = df[feature_cols].astype(float)
    logger.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    logger.info(f"Feature columns: {feature_cols}")

    return X, y, feature_cols, label_encoders


# ---------------------------------------------------------------------------
# Chronological split (Section 25)
# ---------------------------------------------------------------------------

def chronological_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Chronological 70/15/15 split (data must already be sorted by time).
    Strictly prevents future leakage backward. Never shuffles.
    """
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    logger.info(
        f"Chronological split: Train={len(X_train):,} ({len(X_train)/n:.1%}) | "
        f"Val={len(X_val):,} ({len(X_val)/n:.1%}) | "
        f"Test={len(X_test):,} ({len(X_test)/n:.1%})"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation (Section 27)
# ---------------------------------------------------------------------------

def evaluate(
    name: str, model: Any, X: pd.DataFrame, y: pd.Series, split_name: str = "test"
) -> Dict[str, Any]:
    """Evaluate model with non-negative clipping. Returns metrics dict."""
    t0 = time.perf_counter()
    preds = model.predict(X)
    latency_ms = (time.perf_counter() - t0) * 1000

    preds = np.clip(preds, 0, None)
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = r2_score(y, preds)

    result = {
        "model_name": name,
        "split": split_name,
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "latency_ms": round(latency_ms, 2),
        "n_samples": len(y),
    }
    logger.info(
        f"  {name:25s} [{split_name:10s}] → MAE: {result['mae']:.4f}  "
        f"RMSE: {result['rmse']:.4f}  R²: {result['r2']:.4f}  "
        f"({result['latency_ms']:.1f}ms)"
    )
    return result


# ---------------------------------------------------------------------------
# MLflow Experiment Logging (Section 32)
# ---------------------------------------------------------------------------

def _log_to_mlflow(
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Optional[Any] = None,
    artifacts: Optional[List[Path]] = None,
) -> None:
    """Log run to MLflow if available."""
    if not HAS_MLFLOW:
        logger.info("MLflow not installed — skipping experiment tracking.")
        return

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            if model is not None:
                try:
                    if isinstance(model, (lgb.LGBMRegressor, lgb.Booster)):
                        mlflow.lightgbm.log_model(model, artifact_path="model")
                    else:
                        try:
                            mlflow.sklearn.log_model(model, artifact_path="model", skops_trusted_types=True)
                        except TypeError:
                            mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception as m_err:
                    logger.debug(f"Model serialization skipped for MLflow run {run_name}: {m_err}")

            if artifacts:
                for art_path in artifacts:
                    if art_path.exists():
                        try:
                            mlflow.log_artifact(str(art_path))
                        except Exception:
                            pass

        logger.info(f"MLflow run '{run_name}' logged successfully.")
    except Exception as e:
        logger.warning(f"MLflow logging for '{run_name}' failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_pipeline() -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("GeoDemand AI — ML Training Pipeline (Member 3)")
    logger.info("Target: expected_customer_count (next-hour forecast)")
    logger.info("NOTE: All data and results are SIMULATED.")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load & validate
    # ------------------------------------------------------------------
    df = load_and_validate()
    dataset_hash = _file_hash(FEATURE_STORE_PATH)

    X, y, feature_cols, label_encoders = prepare_features(df)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    # ------------------------------------------------------------------
    # 2. Benchmark — Multi-Model Benchmark (Section 26)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Multi-Model Benchmark (Linear, RF, XGBoost, LightGBM)")
    logger.info("=" * 70)

    benchmark_models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            max_samples=0.2,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }

    if HAS_XGBOOST:
        benchmark_models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        logger.info("XGBoost not installed; skipping in benchmark.")

    if HAS_CATBOOST:
        benchmark_models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_SEED,
            verbose=0,
        )
    else:
        logger.info("CatBoost not installed; skipping in benchmark.")

    # LightGBM (Primary candidate)
    benchmark_models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
    )

    val_results = []
    trained_models = {}

    for name, model in benchmark_models.items():
        logger.info(f"\nTraining {name}...")
        t0 = time.perf_counter()

        if name == "LightGBM":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        elif name == "XGBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        train_time = time.perf_counter() - t0
        logger.info(f"  Training time: {train_time:.1f}s")

        metrics = evaluate(name, model, X_val, y_val, split_name="validation")
        metrics["train_time_s"] = round(train_time, 2)
        val_results.append(metrics)
        trained_models[name] = model

        # Log individual benchmark models to MLflow
        if HAS_MLFLOW:
            _log_to_mlflow(
                experiment_name="GeoDemand",
                run_name=f"benchmark_{name.lower().replace(' ', '_')}",
                params={
                    "algorithm": name,
                    "n_features": len(feature_cols),
                    "n_train_rows": len(X_train),
                    "random_seed": RANDOM_SEED,
                },
                metrics={
                    "val_mae": metrics["mae"],
                    "val_rmse": metrics["rmse"],
                    "val_r2": metrics["r2"],
                    "train_time_s": metrics["train_time_s"],
                },
                model=model,
            )

    # Select best model by validation MAE (Section 26)
    val_results_sorted = sorted(val_results, key=lambda x: x["mae"])
    best_name = val_results_sorted[0]["model_name"]
    logger.info(
        f"\n>>> Best model by validation MAE: {best_name} "
        f"(MAE={val_results_sorted[0]['mae']:.4f})"
    )

    # Log validation benchmark summary
    logger.info("\n--- Benchmark Summary (Validation) ---")
    for r in val_results_sorted:
        logger.info(
            f"  {r['model_name']:25s}  MAE={r['mae']:.4f}  "
            f"RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Final model evaluation on holdout test set (Section 28)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Champion Model Evaluation on Holdout Test Set")
    logger.info("=" * 70)

    # Primary candidate is LightGBM
    champion_model = trained_models["LightGBM"]
    test_metrics = evaluate(
        "LightGBM (Champion)", champion_model, X_test, y_test, split_name="test"
    )

    test_results = []
    for name, model in trained_models.items():
        m = evaluate(name, model, X_test, y_test, split_name="test")
        test_results.append(m)

    logger.info(
        f"\nBaseline Verification (Section 28):"
        f"\n  Achieved Test MAE:  {test_metrics['mae']:.4f}"
        f"\n  Achieved Test RMSE: {test_metrics['rmse']:.4f}"
        f"\n  Achieved Test R²:   {test_metrics['r2']:.4f}"
        f"\n  (Reference baseline on simulated data: MAE ~6.38, RMSE ~7.47, R² ~0.438)"
    )

    # ------------------------------------------------------------------
    # 4. Quantile Regression (Section 29: P10, P50, P90)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Quantile Regression — Prediction Intervals (P10, P50, P90)")
    logger.info("=" * 70)

    quantile_alphas = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
    quantile_models = {}

    for qname, alpha in quantile_alphas.items():
        logger.info(f"Training {qname.upper()} quantile model (alpha={alpha})...")
        qmodel = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )
        qmodel.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=25, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        quantile_models[qname] = qmodel

    # Validate quantile ordering on test set: assert P10 <= P50 <= P90
    q_preds = {}
    for qname, qmodel in quantile_models.items():
        raw = qmodel.predict(X_test)
        q_preds[qname] = np.clip(raw, 0, None)

    violations_10_50 = (q_preds["p10"] > q_preds["p50"]).sum()
    violations_50_90 = (q_preds["p50"] > q_preds["p90"]).sum()
    total_test = len(X_test)

    if violations_10_50 > 0 or violations_50_90 > 0:
        logger.warning(
            f"Quantile ordering violations: P10>P50={violations_10_50}, P50>P90={violations_50_90}"
        )
        # Enforce monotonicity: P10 = min(P10, P50), P90 = max(P50, P90)
        q_preds["p10"] = np.minimum(q_preds["p10"], q_preds["p50"])
        q_preds["p90"] = np.maximum(q_preds["p50"], q_preds["p90"])
        logger.info("Quantile ordering enforced post-hoc.")
    else:
        logger.info(f"✓ Quantile ordering P10 <= P50 <= P90 valid on all {total_test:,} test samples.")

    # Non-negativity check
    for qn, qp in q_preds.items():
        assert (qp >= 0).all(), f"Negative predictions in {qn}!"
    logger.info("✓ All quantile predictions are non-negative.")

    logger.info("\n--- Prediction Interval Examples (first 5 test samples) ---")
    for i in range(min(5, len(X_test))):
        actual = y_test.iloc[i]
        logger.info(
            f"  Row {i}: Actual={actual:.0f} | "
            f"P10={q_preds['p10'][i]:.1f} | "
            f"P50={q_preds['p50'][i]:.1f} | "
            f"P90={q_preds['p90'][i]:.1f} | "
            f"Interval Width={q_preds['p90'][i] - q_preds['p10'][i]:.1f}"
        )

    # ------------------------------------------------------------------
    # 5. TreeSHAP Explainability (Section 31)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("TreeSHAP Local Explainability")
    logger.info("=" * 70)

    n_shap_samples = min(5, len(X_test))
    shap_sample = X_test.iloc[:n_shap_samples]

    # TreeSHAP via LightGBM native pred_contrib: shape (n_samples, n_features + 1)
    shap_matrix = champion_model.predict(shap_sample, pred_contrib=True)
    logger.info(f"TreeSHAP contribution matrix shape: {shap_matrix.shape}")

    shap_examples = []
    for i in range(n_shap_samples):
        contribs = shap_matrix[i]
        feature_contribs = contribs[:-1]  # exclude bias
        bias = contribs[-1]

        sorted_idx = np.argsort(np.abs(feature_contribs))[::-1]
        top_drivers = []
        for idx in sorted_idx[:5]:
            val = feature_contribs[idx]
            feat = feature_cols[idx]
            top_drivers.append({
                "feature": feat,
                "shap_value": round(float(val), 4),
                "feature_value": round(float(shap_sample.iloc[i, idx]), 4),
                "impact": "positive" if val > 0 else "negative",
            })

        pred_val = float(np.clip(champion_model.predict(shap_sample.iloc[[i]])[0], 0, None))
        top_pos = [d["feature"] for d in top_drivers if d["impact"] == "positive"][:3]
        top_neg = [d["feature"] for d in top_drivers if d["impact"] == "negative"][:3]

        formatted_explanation = (
            f"Prediction: {int(round(pred_val))} customers/hour\n\n"
            f"Top positive factors:\n" + "\n".join(f"+ {f}" for f in top_pos) + "\n\n"
            f"Top negative factors:\n" + "\n".join(f"- {f}" for f in top_neg)
        )

        shap_examples.append({
            "sample_index": i,
            "prediction": round(pred_val, 2),
            "actual": float(y_test.iloc[i]),
            "bias": round(float(bias), 4),
            "top_drivers": top_drivers,
            "top_positive": top_pos,
            "top_negative": top_neg,
            "formatted_explanation": formatted_explanation,
        })

        logger.info(f"\n--- SHAP Explanation Sample {i} ---")
        logger.info(formatted_explanation)

    # ------------------------------------------------------------------
    # 6. Save Model Artifacts (Section 30)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Saving Model Artifacts")
    logger.info("=" * 70)

    training_timestamp = datetime.now(timezone.utc).isoformat()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Champion model
    joblib.dump(champion_model, ARTIFACT_DIR / "model.joblib")
    logger.info(f"  Saved: {ARTIFACT_DIR / 'model.joblib'}")

    # Quantile models
    for qname, qmodel in quantile_models.items():
        joblib.dump(qmodel, ARTIFACT_DIR / f"{qname}_model.joblib")
        logger.info(f"  Saved: {ARTIFACT_DIR / f'{qname}_model.joblib'}")

    # Label encoders
    joblib.dump(label_encoders, ARTIFACT_DIR / "label_encoders.joblib")
    logger.info(f"  Saved: {ARTIFACT_DIR / 'label_encoders.joblib'}")

    # Feature schema (Section 30 & Section 24)
    feature_schema = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "categorical_columns": list(label_encoders.keys()),
        "categorical_mappings": {
            col: list(le.classes_) for col, le in label_encoders.items()
        },
        "target_contract": PRIMARY_TARGET_CONTRACT,
        "target_column": TARGET_COL,
        "disallowed_targets": sorted(DISALLOWED_TARGETS),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "dropped_columns": sorted(DROP_COLUMNS),
    }
    with open(ARTIFACT_DIR / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)
    logger.info(f"  Saved: {ARTIFACT_DIR / 'feature_schema.json'}")

    # Metadata (Section 30: model_version, training_dataset_version, feature_version,
    # training_timestamp, metrics, feature_columns, algorithm)
    metadata = {
        "model_version": "v1.0.0",
        "training_dataset_version": f"feature_store_{dataset_hash}",
        "feature_version": "v2.1-next-hour-forecast",
        "training_timestamp": training_timestamp,
        "algorithm": "LightGBM",
        "target_contract": PRIMARY_TARGET_CONTRACT,
        "target_column": TARGET_COL,
        "contract_principle": "Predict expected customer count, never profit/revenue/decision/demand score directly.",
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "categorical_columns": list(label_encoders.keys()),
        "metrics": {
            "validation": {
                k: v
                for k, v in val_results_sorted[0].items()
                if k in ("mae", "rmse", "r2")
            },
            "test": {
                k: v for k, v in test_metrics.items() if k in ("mae", "rmse", "r2")
            },
        },
        "benchmark_results": val_results_sorted,
        "test_results": test_results,
        "quantiles": list(quantile_alphas.keys()),
        "quantile_ordering_valid": violations_10_50 == 0 and violations_50_90 == 0,
        "random_seed": RANDOM_SEED,
        "n_train_rows": len(X_train),
        "n_val_rows": len(X_val),
        "n_test_rows": len(X_test),
        "shap_enabled": True,
        "shap_method": "lightgbm_native_pred_contrib",
        "shap_examples": shap_examples,
        "data_note": "SIMULATED — do not interpret as real-world accuracy",
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved: {ARTIFACT_DIR / 'metadata.json'}")

    # Also save model_metadata.json (Section 30 naming)
    with open(ARTIFACT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved: {ARTIFACT_DIR / 'model_metadata.json'}")

    # API-compatible bundle for Member 4 (recommender.py)
    bundle = {
        "model": champion_model,
        "p10_model": quantile_models["p10"],
        "p90_model": quantile_models["p90"],
        "feature_cols": feature_cols,
        "label_encoders": label_encoders,
        "benchmark_metrics": val_results_sorted,
        "metrics": test_metrics,
        "meta": {
            "model_type": "lightgbm_quantile_bundle",
            "model_version": "v1.0.0",
            "n_features": len(feature_cols),
            "target": TARGET_COL,
            "trained_at": training_timestamp,
            "n_train_rows": len(X_train),
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "quantiles_supported": ["p10", "p50", "p90"],
            "shap_enabled": True,
            "random_seed": RANDOM_SEED,
            "data_note": "SIMULATED",
        },
    }
    bundle_path = MODELS_DIR / "demand_model.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"  Saved API bundle: {bundle_path}")

    # Also save models/demand_model_meta.json
    api_meta = {
        "model_type": "lightgbm",
        "trained_at": training_timestamp,
        "feature_columns": feature_cols,
        "categorical_columns": list(label_encoders.keys()),
        "target": TARGET_COL,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "metrics": {
            "val_mae": metadata["metrics"]["validation"]["mae"],
            "val_rmse": metadata["metrics"]["validation"]["rmse"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        },
        "source": f"feature_store.parquet (v2.1-next-hour-forecast, hash={dataset_hash})",
        "data_note": "SIMULATED",
    }
    with open(MODELS_DIR / "demand_model_meta.json", "w") as f:
        json.dump(api_meta, f, indent=2)
    logger.info(f"  Saved: {MODELS_DIR / 'demand_model_meta.json'}")

    # ------------------------------------------------------------------
    # 7. Log Champion & Artifacts to MLflow (Section 32)
    # ------------------------------------------------------------------
    if HAS_MLFLOW:
        _log_to_mlflow(
            experiment_name="GeoDemand",
            run_name="lightgbm_champion",
            params={
                "algorithm": "LightGBM",
                "model_version": "v1.0.0",
                "dataset_version": metadata["training_dataset_version"],
                "feature_version": metadata["feature_version"],
                "n_features": len(feature_cols),
                "n_train_rows": len(X_train),
                "n_val_rows": len(X_val),
                "n_test_rows": len(X_test),
                "random_seed": RANDOM_SEED,
                "quantiles": "p10,p50,p90",
            },
            metrics={
                "val_mae": metadata["metrics"]["validation"]["mae"],
                "val_rmse": metadata["metrics"]["validation"]["rmse"],
                "val_r2": metadata["metrics"]["validation"]["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            },
            model=champion_model,
            artifacts=[
                ARTIFACT_DIR / "metadata.json",
                ARTIFACT_DIR / "feature_schema.json",
            ],
        )

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Training Pipeline Complete (Member 3)")
    logger.info("=" * 70)
    logger.info(f"Champion Model: LightGBM")
    logger.info(f"Test MAE:  {test_metrics['mae']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test R²:   {test_metrics['r2']:.4f}")
    logger.info(f"Artifacts: {ARTIFACT_DIR}")
    logger.info(f"API bundle: {bundle_path}")
    logger.info("NOTE: Results are SIMULATED — not real-world accuracy.")

    return metadata


if __name__ == "__main__":
    train_pipeline()
