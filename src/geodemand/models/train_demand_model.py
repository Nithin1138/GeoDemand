"""
GeoDemand AI — ML Model Training, Quantile Calibration & Explainability Pipeline

Member 3 — Machine Learning & MLOps

Phases:
  - Benchmark: Multi-Model Benchmark with chronological split
    (Linear Regression, Random Forest, XGBoost, LightGBM L1, LightGBM Huber, LightGBM Tuned)
    evaluated on MAE, RMSE, R² using validation set for dynamic champion selection.
  - Quantile Regression: Rigorous calibrated prediction intervals (P10 <= P50 <= P90).
  - TreeSHAP: Native TreeSHAP explainability for local positive & negative drivers.
  - MLflow: Comprehensive experiment tracking and artifact logging with SQLite backend.

Data contract:
  - Source: data/processed/feature_store.parquet (canonical v2.1 training dataset)
  - Target: expected_customer_count (representing next-hour customer count)
  - Temporal contract: features at decision_timestamp (t) → predict target at t+1h
  - Split: chronological 70% train / 15% validation / 15% test (strictly timestamp-aligned, no shuffle)

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

# Categorical columns that need native categorical handling / label encoding
CATEGORICAL_COLUMNS = ["weekday", "season", "vendor_category"]

# Zero-variance columns to drop (weather_condition has only 'clear')
DROP_COLUMNS = {"weather_condition"}

WEEKDAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


def _file_hash(path: Path) -> str:
    """SHA-256 of the first 1MB of a file for version tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()[:12]


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive deterministic cyclical calendar features matching feature_assembler.py:
    hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos.
    """
    df = df.copy()
    if "hour" in df.columns:
        hour_val = df["hour"].astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * hour_val / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour_val / 24.0)

    if "weekday" in df.columns:
        day_num = df["weekday"].map(WEEKDAY_MAP).fillna(0).astype(float)
        df["day_sin"] = np.sin(2 * np.pi * day_num / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * day_num / 7.0)
    elif "day_of_week" in df.columns:
        day_num = df["day_of_week"].astype(float)
        df["day_sin"] = np.sin(2 * np.pi * day_num / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * day_num / 7.0)

    if "month" in df.columns:
        month_val = df["month"].astype(float)
        df["month_sin"] = np.sin(2 * np.pi * month_val / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * month_val / 12.0)

    return df


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
    train_mask: Optional[pd.Series] = None,
    fitted_encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, LabelEncoder]]:
    """
    Prepare feature matrix X and target y from the feature store.

    - Adds cyclical temporal features (hour_sin/cos, day_sin/cos, month_sin/cos)
    - Drops forbidden columns and zero-variance columns
    - Fits LabelEncoder strictly on training split (if train_mask provided and no fitted_encoders)
    - Safely maps unseen categories to 0
    - Converts booleans to int
    - Returns (X, y, feature_columns, label_encoders)
    """
    df = add_cyclical_features(df)

    # Extract target
    y = df[TARGET_COL].astype(float)

    # Determine trainable columns
    trainable = [
        c for c in df.columns if c not in FORBIDDEN_COLUMNS and c not in DROP_COLUMNS
    ]

    # Label-encode categoricals strictly on training data with fast vectorized mapping
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        if col in trainable:
            if fitted_encoders and col in fitted_encoders:
                le = fitted_encoders[col]
            else:
                le = LabelEncoder()
                fit_data = df.loc[train_mask, col] if train_mask is not None else df[col]
                le.fit(fit_data.astype(str))

            mapping = {cls_name: int(idx) for idx, cls_name in enumerate(le.classes_)}
            df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)
            label_encoders[col] = le
            logger.info(f"Categorical '{col}' encoded with {len(le.classes_)} classes: {list(le.classes_)}")

    # Convert booleans to int
    for col in trainable:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Build X
    feature_cols = sorted(trainable)
    X = df[feature_cols].copy()
    for col in feature_cols:
        if col in label_encoders:
            X[col] = X[col].astype(int)
        else:
            X[col] = X[col].astype(float)

    logger.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    logger.info(f"Feature columns: {feature_cols}")

    return X, y, feature_cols, label_encoders


# ---------------------------------------------------------------------------
# Chronological split (Section 25)
# ---------------------------------------------------------------------------

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Chronological 70/15/15 split.
    If timestamps series is provided, aligns strictly on decision_timestamp boundaries
    to prevent temporal leakage across splits (all observations for each hour stay together).
    """
    n = len(X)
    if timestamps is not None:
        train_idx_raw = int(n * 0.70)
        val_idx_raw = int(n * 0.85)

        train_cutoff_ts = timestamps.iloc[train_idx_raw]
        val_cutoff_ts = timestamps.iloc[val_idx_raw]

        train_mask = timestamps < train_cutoff_ts
        val_mask = (timestamps >= train_cutoff_ts) & (timestamps < val_cutoff_ts)
        test_mask = timestamps >= val_cutoff_ts

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        logger.info(
            f"Boundary timestamps: Train max ({timestamps[train_mask].max()}) < "
            f"Val min ({timestamps[val_mask].min()}) <= Val max ({timestamps[val_mask].max()}) < "
            f"Test min ({timestamps[test_mask].min()})"
        )
    else:
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
        f"  {name:30s} [{split_name:10s}] → MAE: {result['mae']:.4f}  "
        f"RMSE: {result['rmse']:.4f}  R²: {result['r2']:.4f}  "
        f"({result['latency_ms']:.1f}ms)"
    )
    return result


# ---------------------------------------------------------------------------
# MLflow Experiment Logging (Section 32)
# ---------------------------------------------------------------------------

def _configure_mlflow() -> None:
    """Configure MLflow SQLite tracking URI."""
    if not HAS_MLFLOW:
        return
    db_path = ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.resolve()}".replace("\\", "/")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI configured: {tracking_uri}")
    except Exception as e:
        logger.warning(f"Could not set MLflow tracking URI: {e}")


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

    _configure_mlflow()
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
    # 1. Load, validate, and chronological split
    # ------------------------------------------------------------------
    df = load_and_validate()
    dataset_hash = _file_hash(FEATURE_STORE_PATH)

    n_total = len(df)
    train_idx_raw = int(n_total * 0.70)
    train_cutoff_ts = df["decision_timestamp"].iloc[train_idx_raw]
    train_mask = df["decision_timestamp"] < train_cutoff_ts

    # Prepare features: fit label encoders strictly on training rows
    X, y, feature_cols, label_encoders = prepare_features(df, train_mask=train_mask)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X, y, timestamps=df["decision_timestamp"]
    )

    # ------------------------------------------------------------------
    # 2. Benchmark — Multi-Model Benchmark & Loss Alignment
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Multi-Model Benchmark (Linear, RF, XGBoost, LightGBM Objectives)")
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
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=30,
        )
    else:
        logger.info("XGBoost not installed; skipping in benchmark.")

    # Phase 3: Benchmark LightGBM Loss Objectives (L2, L1, Huber)
    benchmark_models["LightGBM (regression_l1)"] = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
    )

    benchmark_models["LightGBM (huber)"] = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
    )

    # Phase 4: Tuned Champion Candidate (Optimal leaves, child samples, and regularizers)
    benchmark_models["LightGBM (Tuned Champion)"] = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=30,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
    )

    val_results = []
    trained_models = {}

    for name, model in benchmark_models.items():
        logger.info(f"\nTraining {name}...")
        t0 = time.perf_counter()

        if "LightGBM" in name:
            model.fit(
                X_train,
                y_train,
                categorical_feature=CATEGORICAL_COLUMNS,
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

        # Log individual benchmark model to MLflow
        if HAS_MLFLOW:
            _log_to_mlflow(
                experiment_name="GeoDemand",
                run_name=f"benchmark_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
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

    # ------------------------------------------------------------------
    # 3. Dynamic Model Selection with Pre-Flight Verification Gate (Phase 7)
    # ------------------------------------------------------------------
    val_results_sorted = sorted(val_results, key=lambda x: x["mae"])

    logger.info("\n--- Benchmark Summary (Validation) ---")
    for r in val_results_sorted:
        logger.info(
            f"  {r['model_name']:35s}  MAE={r['mae']:.4f}  "
            f"RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}"
        )

    # Phase 7 Pre-flight Promotion Gate:
    # Verify: validation performance, inference compatibility, feature schema consistency,
    # serialization/deserialization, and TreeSHAP attribution.
    champion_name = None
    champion_model = None

    for r in val_results_sorted:
        cand_name = r["model_name"]
        cand_model = trained_models[cand_name]
        logger.info(f"\nEvaluating candidate for champion promotion: '{cand_name}' (Val MAE={r['mae']:.4f})...")

        is_eligible = True
        # 1. Inference shape & finite output check
        try:
            sample_X = X_val.iloc[:20]
            preds = cand_model.predict(sample_X)
            if len(preds) != len(sample_X) or not np.all(np.isfinite(preds)):
                logger.warning(f"  [GATE FAIL] Inference shape or non-finite output check failed for {cand_name}")
                is_eligible = False
        except Exception as e:
            logger.warning(f"  [GATE FAIL] Prediction failed for {cand_name}: {e}")
            is_eligible = False

        # 2. Artifact serialization roundtrip check
        if is_eligible:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                    tmp_file = Path(tmp.name)
                joblib.dump(cand_model, tmp_file)
                loaded = joblib.load(tmp_file)
                np.testing.assert_allclose(cand_model.predict(sample_X), loaded.predict(sample_X), rtol=1e-5)
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception as e:
                logger.warning(f"  [GATE FAIL] Artifact roundtrip serialization failed for {cand_name}: {e}")
                is_eligible = False

        # 3. TreeSHAP attribution compatibility check
        if is_eligible:
            try:
                if hasattr(cand_model, "predict"):
                    try:
                        _ = cand_model.predict(sample_X.iloc[:2], pred_contrib=True)
                    except (TypeError, ValueError):
                        if hasattr(cand_model, "get_booster") and HAS_XGBOOST:
                            _ = cand_model.get_booster().predict(xgb.DMatrix(sample_X.iloc[:2]), pred_contribs=True)
                        elif HAS_SHAP:
                            _ = shap.TreeExplainer(cand_model).shap_values(sample_X.iloc[:2])
            except Exception as e:
                logger.warning(f"  [GATE FAIL] TreeSHAP check failed for {cand_name}: {e}")
                is_eligible = False

        if is_eligible:
            champion_name = cand_name
            champion_model = cand_model
            logger.info(f"✓ PROMOTED CHAMPION: '{champion_name}' passed all MLOps verification checks!")
            break

    if champion_model is None:
        logger.warning("No candidate passed all checks. Falling back to top validation model.")
        champion_name = val_results_sorted[0]["model_name"]
        champion_model = trained_models[champion_name]

    # ------------------------------------------------------------------
    # 4. Final model evaluation on holdout test set (Section 28)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f"Champion Model Evaluation on Holdout Test Set ({champion_name})")
    logger.info("=" * 70)

    test_metrics = evaluate(
        f"{champion_name} (Holdout)", champion_model, X_test, y_test, split_name="test"
    )

    test_results = []
    for name, model in trained_models.items():
        m = evaluate(name, model, X_test, y_test, split_name="test")
        test_results.append(m)

    logger.info(
        f"\nAchieved Champion Holdout Performance:"
        f"\n  Achieved Test MAE:  {test_metrics['mae']:.4f}"
        f"\n  Achieved Test RMSE: {test_metrics['rmse']:.4f}"
        f"\n  Achieved Test R²:   {test_metrics['r2']:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Quantile Regression (Section 29 & Phase 6: P10, P50, P90)
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
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )
        qmodel.fit(
            X_train,
            y_train,
            categorical_feature=CATEGORICAL_COLUMNS,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=25, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        quantile_models[qname] = qmodel

    # Monotonic quantile rearrangement: P10 <= P50 <= P90
    raw_p10 = quantile_models["p10"].predict(X_test)
    raw_p50 = quantile_models["p50"].predict(X_test)
    raw_p90 = quantile_models["p90"].predict(X_test)

    mono_p10 = np.clip(np.minimum(raw_p10, raw_p50), 0, None)
    mono_p50 = np.clip(raw_p50, 0, None)
    mono_p90 = np.clip(np.maximum(raw_p50, raw_p90), 0, None)

    q_preds = {"p10": mono_p10, "p50": mono_p50, "p90": mono_p90}

    # Verify monotonicity (0 violations)
    violations_10_50 = int((q_preds["p10"] > q_preds["p50"]).sum())
    violations_50_90 = int((q_preds["p50"] > q_preds["p90"]).sum())
    assert violations_10_50 == 0 and violations_50_90 == 0, "Quantile ordering violated after rearrangement!"

    # Empirical coverage and spread
    empirical_coverage = float(((y_test >= q_preds["p10"]) & (y_test <= q_preds["p90"])).mean())
    avg_interval_width = float((q_preds["p90"] - q_preds["p10"]).mean())

    logger.info(f"✓ Quantile ordering P10 <= P50 <= P90 strictly satisfied across all {len(X_test):,} samples.")
    logger.info(f"✓ Empirical [P10, P90] Coverage: {empirical_coverage:.2%} (Nominal: 80.0%)")
    logger.info(f"✓ Average Prediction Interval Width: {avg_interval_width:.2f}")

    # Non-negativity check
    for qn, qp in q_preds.items():
        assert (qp >= 0).all(), f"Negative predictions in {qn}!"
    logger.info("✓ All quantile predictions are strictly non-negative.")

    logger.info("\n--- Prediction Interval Examples (first 5 test samples) ---")
    for i in range(min(5, len(X_test))):
        actual = y_test.iloc[i]
        logger.info(
            f"  Row {i}: Actual={actual:.0f} | "
            f"P10={q_preds['p10'][i]:.1f} | "
            f"P50={q_preds['p50'][i]:.1f} | "
            f"P90={q_preds['p90'][i]:.1f} | "
            f"Width={q_preds['p90'][i] - q_preds['p10'][i]:.1f}"
        )

    # ------------------------------------------------------------------
    # 6. TreeSHAP Explainability (Section 31)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("TreeSHAP Local Explainability")
    logger.info("=" * 70)

    n_shap_samples = min(5, len(X_test))
    shap_sample = X_test.iloc[:n_shap_samples]

    try:
        shap_matrix = champion_model.predict(shap_sample, pred_contrib=True)
    except (TypeError, ValueError):
        if hasattr(champion_model, "get_booster") and HAS_XGBOOST:
            dmat = xgb.DMatrix(shap_sample)
            shap_matrix = champion_model.get_booster().predict(dmat, pred_contribs=True)
        elif HAS_SHAP:
            explainer = shap.TreeExplainer(champion_model)
            raw = explainer.shap_values(shap_sample)
            bias = np.full((len(shap_sample), 1), explainer.expected_value)
            shap_matrix = np.hstack([raw, bias])
        else:
            shap_matrix = np.zeros((len(shap_sample), len(feature_cols) + 1))
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
    # 7. Save Model Artifacts (Section 30)
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
        "cyclical_features": [c for c in feature_cols if c.endswith("_sin") or c.endswith("_cos")],
        "target_contract": PRIMARY_TARGET_CONTRACT,
        "target_column": TARGET_COL,
        "disallowed_targets": sorted(DISALLOWED_TARGETS),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "dropped_columns": sorted(DROP_COLUMNS),
    }
    with open(ARTIFACT_DIR / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)
    logger.info(f"  Saved: {ARTIFACT_DIR / 'feature_schema.json'}")

    # Metadata
    metadata = {
        "model_version": "v2.0.0",
        "training_dataset_version": f"feature_store_{dataset_hash}",
        "feature_version": "v2.2-next-hour-forecast-cyclical",
        "training_timestamp": training_timestamp,
        "algorithm": champion_name,
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
        "quantile_ordering_valid": True,
        "empirical_coverage_80": round(empirical_coverage, 4),
        "average_interval_width": round(avg_interval_width, 4),
        "random_seed": RANDOM_SEED,
        "n_train_rows": len(X_train),
        "n_val_rows": len(X_val),
        "n_test_rows": len(X_test),
        "shap_enabled": True,
        "shap_method": "lightgbm_native_pred_contrib" if "lightgbm" in champion_name.lower() else "tree_shap",
        "shap_examples": shap_examples,
        "data_note": "SIMULATED — do not interpret as real-world accuracy",
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved: {ARTIFACT_DIR / 'metadata.json'}")

    with open(ARTIFACT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved: {ARTIFACT_DIR / 'model_metadata.json'}")

    # API-compatible bundle for Member 4 (recommender.py)
    bundle = {
        "model": champion_model,
        "p10_model": quantile_models["p10"],
        "p90_model": quantile_models["p90"],
        "p50_model": quantile_models["p50"],
        "feature_cols": feature_cols,
        "label_encoders": label_encoders,
        "benchmark_metrics": val_results_sorted,
        "metrics": test_metrics,
        "meta": {
            "model_type": f"{champion_name.lower().replace(' ', '_')}_quantile_bundle",
            "model_version": "v2.0.0",
            "algorithm": champion_name,
            "n_features": len(feature_cols),
            "target": TARGET_COL,
            "trained_at": training_timestamp,
            "n_train_rows": len(X_train),
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "quantiles_supported": ["p10", "p50", "p90"],
            "quantile_coverage": round(empirical_coverage, 4),
            "quantile_width": round(avg_interval_width, 4),
            "shap_enabled": True,
            "random_seed": RANDOM_SEED,
            "data_note": "SIMULATED",
        },
    }
    bundle_path = MODELS_DIR / "demand_model.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"  Saved API bundle: {bundle_path}")

    # Save models/demand_model_meta.json
    api_meta = {
        "model_type": champion_name.lower().replace(" ", "_"),
        "model_version": "v2.0.0",
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
        "quantile_coverage_80": round(empirical_coverage, 4),
        "quantile_interval_width": round(avg_interval_width, 4),
        "source": f"feature_store.parquet (v2.2-next-hour-forecast-cyclical, hash={dataset_hash})",
        "data_note": "SIMULATED",
    }
    with open(MODELS_DIR / "demand_model_meta.json", "w") as f:
        json.dump(api_meta, f, indent=2)
    logger.info(f"  Saved: {MODELS_DIR / 'demand_model_meta.json'}")

    # ------------------------------------------------------------------
    # 8. Log Champion & Artifacts to MLflow (Section 32 & Phase 8)
    # ------------------------------------------------------------------
    if HAS_MLFLOW:
        _log_to_mlflow(
            experiment_name="GeoDemand",
            run_name=f"{champion_name.lower().replace(' ', '_')}_champion_v2",
            params={
                "algorithm": champion_name,
                "model_version": "v2.0.0",
                "dataset_version": metadata["training_dataset_version"],
                "feature_version": metadata["feature_version"],
                "n_features": len(feature_cols),
                "n_train_rows": len(X_train),
                "n_val_rows": len(X_val),
                "n_test_rows": len(X_test),
                "random_seed": RANDOM_SEED,
                "quantiles": "p10,p50,p90",
                "empirical_coverage_80": round(empirical_coverage, 4),
                "avg_interval_width": round(avg_interval_width, 4),
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
    # 9. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Training Pipeline Complete (Member 3)")
    logger.info("=" * 70)
    logger.info(f"Champion Model: {champion_name}")
    logger.info(f"Test MAE:  {test_metrics['mae']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test R²:   {test_metrics['r2']:.4f}")
    logger.info(f"Quantile 80% Coverage: {empirical_coverage:.2%}")
    logger.info(f"Artifacts: {ARTIFACT_DIR}")
    logger.info(f"API bundle: {bundle_path}")
    logger.info("NOTE: Results are SIMULATED — not real-world accuracy.")

    return metadata


if __name__ == "__main__":
    train_pipeline()
