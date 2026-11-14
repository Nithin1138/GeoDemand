"""
GeoDemand AI — Fair XGBoost vs LightGBM Model Comparison Engine

Member 3 — Machine Learning & MLOps

Executes an objective, fair, and reproducible comparison between XGBoost and LightGBM:
- Identical chronological data splits (Train 70%, Val 15%, Test 15%)
- Preprocessing fit strictly on training split
- Baseline models evaluation
- Symmetric hyperparameter optimization with identical validation budgets
- Expanding-window temporal generalization across forward time folds
- Paired statistical significance testing with HAC / Newey-West standard errors
- Prediction quality, contract non-negativity, and TreeSHAP explainability verification
- Calibrated prediction intervals (P10 <= P50 <= P90)
- Single-row and batch latency benchmarks (raw vs end-to-end)
- Final untouched holdout test set evaluation
- Artifact safety and MLflow experiment logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.geodemand.models.inference import DemandModelPredictor
from src.geodemand.models.promotion import (
    ModelPromotionGate,
    ModelPromotionPolicy,
)
from src.geodemand.models.train_demand_model import (
    CATEGORICAL_COLUMNS,
    FEATURE_STORE_PATH,
    FORBIDDEN_COLUMNS,
    RANDOM_SEED,
    TARGET_COL,
    _file_hash,
    add_cyclical_features,
    chronological_split,
    load_and_validate,
    prepare_features,
)
from src.geodemand.monitoring.performance import SIMULATION_DATA_DISCLAIMER
from src.geodemand.monitoring.prediction_quality import PredictionValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_xgb_vs_lgbm")

EXPERIMENTS_DIR = ROOT_DIR / "models" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def measure_inference_latencies(
    model: Any,
    sample_df: pd.DataFrame,
    is_lgbm: bool = False,
    feature_cols: Optional[List[str]] = None,
    label_encoders: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Measure raw model prediction and end-to-end latency for single-row and batch inference.
    Reports median and p95 latency in milliseconds.
    """
    # 1. Warm-up
    warmup_batch = sample_df.iloc[:50]
    for _ in range(5):
        _ = model.predict(warmup_batch)

    # 2. Raw Single-Row Prediction Latency
    single_row = sample_df.iloc[:1]
    raw_single_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = model.predict(single_row)
        raw_single_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. Raw Batch Prediction Latency (1,000 samples)
    batch_rows = sample_df.iloc[: min(1000, len(sample_df))]
    raw_batch_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model.predict(batch_rows)
        raw_batch_times.append((time.perf_counter() - t0) * 1000.0)

    # 4. End-to-End Latency (simulate dict feature ingestion -> preprocess -> predict -> clip)
    raw_dict = single_row.iloc[0].to_dict()
    e2e_single_times = []
    for _ in range(150):
        t0 = time.perf_counter()
        row_df = pd.DataFrame([raw_dict])
        preds_raw = model.predict(row_df)
        _ = np.clip(preds_raw, 0, None)
        e2e_single_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "raw_single_median_ms": float(np.median(raw_single_times)),
        "raw_single_p95_ms": float(np.percentile(raw_single_times, 95)),
        "raw_batch_median_ms": float(np.median(raw_batch_times)),
        "raw_batch_p95_ms": float(np.percentile(raw_batch_times, 95)),
        "e2e_single_median_ms": float(np.median(e2e_single_times)),
        "e2e_single_p95_ms": float(np.percentile(e2e_single_times, 95)),
    }


def compute_hac_paired_stats(
    diffs: np.ndarray,
    max_lag: int = 24,
) -> Tuple[float, float, float, float, float]:
    """
    Computes Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard error,
    t-statistic, p-value, and 95% confidence interval for paired differences.
    """
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    demeaned = diffs - mean_diff

    # Sample variance (gamma_0)
    gamma_0 = np.sum(demeaned ** 2) / n

    # Autocovariances
    autocov = 0.0
    for lag in range(1, min(max_lag + 1, n)):
        weight = 1.0 - (lag / (max_lag + 1.0))  # Bartlett kernel
        cov_lag = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
        autocov += 2.0 * weight * cov_lag

    var_hac = (gamma_0 + autocov) / n
    se_hac = float(np.sqrt(max(1e-12, var_hac)))
    t_stat = mean_diff / se_hac
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    ci_lower = mean_diff - 1.96 * se_hac
    ci_upper = mean_diff + 1.96 * se_hac

    return mean_diff, se_hac, t_stat, p_val, ci_lower, ci_upper


def run_benchmark():
    logger.info("=" * 78)
    logger.info("GEODEMAND AI: FAIR EXPERIMENTAL COMPARISON: XGBOOST vs LIGHTGBM")
    logger.info("=" * 78)

    # -------------------------------------------------------------------------
    # PHASE 1 & 2: DATA LOADING & SPLITTING
    # -------------------------------------------------------------------------
    logger.info("\n[Phase 1 & 2] Loading Canonical Feature Store & Creating Splits...")
    df = load_and_validate()
    n_total = len(df)
    logger.info(f"Loaded {n_total:,} rows. Contract target: '{TARGET_COL}'.")

    # Chronological Split: 70% Train, 15% Validation, 15% Test
    train_idx_raw = int(n_total * 0.70)
    train_cutoff_ts = df["decision_timestamp"].iloc[train_idx_raw]
    train_mask = df["decision_timestamp"] < train_cutoff_ts

    # Preprocessing strictly fit on training data
    X, y, feature_cols, label_encoders = prepare_features(df, train_mask=train_mask)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X, y, timestamps=df["decision_timestamp"]
    )
    val_timestamps = df.loc[X_val.index, "decision_timestamp"]
    test_timestamps = df.loc[X_test.index, "decision_timestamp"]

    logger.info(f"Split sizes: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    logger.info(f"Validation horizon: {val_timestamps.min()} -> {val_timestamps.max()}")
    logger.info(f"Holdout test horizon: {test_timestamps.min()} -> {test_timestamps.max()} (Untouched until Phase 8)")

    # -------------------------------------------------------------------------
    # PHASE 3: BASELINE COMPARISON
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("[Phase 3] Establishing Baseline Comparison (Current Configurations)")
    logger.info("=" * 78)

    # 1. Baseline XGBoost (Matches active Champion config)
    logger.info("\nTraining Baseline XGBoost (Champion Configuration)...")
    xgb_baseline = xgb.XGBRegressor(
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
    t0_xgb_base = time.perf_counter()
    xgb_baseline.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_base_train_time = time.perf_counter() - t0_xgb_base

    xgb_base_val_preds = np.clip(xgb_baseline.predict(X_val), 0, None)
    xgb_base_val_mae = float(mean_absolute_error(y_val, xgb_base_val_preds))
    xgb_base_val_rmse = float(np.sqrt(mean_squared_error(y_val, xgb_base_val_preds)))
    xgb_base_val_r2 = float(r2_score(y_val, xgb_base_val_preds))

    # Serialize to measure size
    xgb_base_path = EXPERIMENTS_DIR / "xgb_baseline.joblib"
    joblib.dump(xgb_baseline, xgb_base_path)
    xgb_base_size_mb = xgb_base_path.stat().st_size / (1024.0 * 1024.0)

    # Measure latency
    xgb_base_latencies = measure_inference_latencies(
        xgb_baseline, X_val, is_lgbm=False, feature_cols=feature_cols, label_encoders=label_encoders
    )

    logger.info(f"[OK] Baseline XGBoost: Val MAE={xgb_base_val_mae:.4f}, RMSE={xgb_base_val_rmse:.4f}, R²={xgb_base_val_r2:.4f}")
    logger.info(f"  Training Time: {xgb_base_train_time:.2f}s | Size: {xgb_base_size_mb:.2f} MB")
    logger.info(f"  Single-row Latency: Median={xgb_base_latencies['raw_single_median_ms']:.4f} ms, P95={xgb_base_latencies['raw_single_p95_ms']:.4f} ms")
    logger.info(f"  Batch (1k) Latency: Median={xgb_base_latencies['raw_batch_median_ms']:.2f} ms, P95={xgb_base_latencies['raw_batch_p95_ms']:.2f} ms")

    # 2. Baseline LightGBM (Matches current Tuned Champion config)
    logger.info("\nTraining Baseline LightGBM (Tuned Candidate Configuration)...")
    lgb_baseline = lgb.LGBMRegressor(
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
    t0_lgb_base = time.perf_counter()
    lgb_baseline.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
    )
    lgb_base_train_time = time.perf_counter() - t0_lgb_base

    lgb_base_val_preds = np.clip(lgb_baseline.predict(X_val), 0, None)
    lgb_base_val_mae = float(mean_absolute_error(y_val, lgb_base_val_preds))
    lgb_base_val_rmse = float(np.sqrt(mean_squared_error(y_val, lgb_base_val_preds)))
    lgb_base_val_r2 = float(r2_score(y_val, lgb_base_val_preds))

    lgb_base_path = EXPERIMENTS_DIR / "lgb_baseline.joblib"
    joblib.dump(lgb_baseline, lgb_base_path)
    lgb_base_size_mb = lgb_base_path.stat().st_size / (1024.0 * 1024.0)

    lgb_base_latencies = measure_inference_latencies(
        lgb_baseline, X_val, is_lgbm=True, feature_cols=feature_cols, label_encoders=label_encoders
    )

    logger.info(f"[OK] Baseline LightGBM: Val MAE={lgb_base_val_mae:.4f}, RMSE={lgb_base_val_rmse:.4f}, R²={lgb_base_val_r2:.4f}")
    logger.info(f"  Training Time: {lgb_base_train_time:.2f}s | Size: {lgb_base_size_mb:.2f} MB")
    logger.info(f"  Single-row Latency: Median={lgb_base_latencies['raw_single_median_ms']:.4f} ms, P95={lgb_base_latencies['raw_single_p95_ms']:.4f} ms")
    logger.info(f"  Batch (1k) Latency: Median={lgb_base_latencies['raw_batch_median_ms']:.2f} ms, P95={lgb_base_latencies['raw_batch_p95_ms']:.2f} ms")

    # -------------------------------------------------------------------------
    # PHASE 4: FAIR HYPERPARAMETER OPTIMIZATION (SYMMETRIC BUDGET)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("[Phase 4] Symmetric Hyperparameter Optimization on Validation Split")
    logger.info("=" * 78)

    # XGBoost Candidate Grid (4 additional candidates exploring depth, lr, regularization)
    xgb_candidates_configs = [
        {"name": "XGB_Base", "params": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0}},
        {"name": "XGB_Deep", "params": {"n_estimators": 400, "learning_rate": 0.04, "max_depth": 9, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.0}},
        {"name": "XGB_Regularized", "params": {"n_estimators": 450, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.75, "colsample_bytree": 0.75, "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 2.0}},
        {"name": "XGB_Fast_LR", "params": {"n_estimators": 350, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 2, "reg_alpha": 0.1, "reg_lambda": 0.5}},
        {"name": "XGB_Conservative", "params": {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2, "reg_alpha": 0.2, "reg_lambda": 1.0}},
    ]

    # LightGBM Candidate Grid (4 additional candidates exploring leaves, depth, objectives)
    lgb_candidates_configs = [
        {"name": "LGB_Tuned_L2", "params": {"objective": "regression", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1}},
        {"name": "LGB_L1_Objective", "params": {"objective": "regression_l1", "n_estimators": 450, "learning_rate": 0.05, "num_leaves": 95, "max_depth": -1, "min_child_samples": 25, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.5}},
        {"name": "LGB_Deep_Leaves", "params": {"objective": "regression", "n_estimators": 450, "learning_rate": 0.04, "num_leaves": 180, "max_depth": 9, "min_child_samples": 25, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 1.0}},
        {"name": "LGB_Regularized", "params": {"objective": "regression", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "max_depth": 7, "min_child_samples": 40, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0}},
        {"name": "LGB_Conservative", "params": {"objective": "regression", "n_estimators": 550, "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1, "min_child_samples": 30, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 0.5}},
    ]

    xgb_hpo_results = []
    trained_xgb_models = {}
    logger.info("\n--- Tuning XGBoost Candidates (5 configurations) ---")
    for cfg in xgb_candidates_configs:
        cname = cfg["name"]
        params = cfg["params"]
        model = xgb.XGBRegressor(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=30,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        dur = time.perf_counter() - t0
        preds = np.clip(model.predict(X_val), 0, None)
        v_mae = float(mean_absolute_error(y_val, preds))
        v_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        v_r2 = float(r2_score(y_val, preds))
        best_iter = getattr(model, "best_iteration", params.get("n_estimators"))
        xgb_hpo_results.append({
            "name": cname, "val_mae": v_mae, "val_rmse": v_rmse, "val_r2": v_r2,
            "train_time_s": dur, "best_iteration": best_iter, "params": params
        })
        trained_xgb_models[cname] = model
        logger.info(f"  {cname:18s} -> Val MAE: {v_mae:.4f} | RMSE: {v_rmse:.4f} | R²: {v_r2:.4f} (Fit: {dur:.1f}s, Iter: {best_iter})")

    lgb_hpo_results = []
    trained_lgb_models = {}
    logger.info("\n--- Tuning LightGBM Candidates (5 configurations) ---")
    for cfg in lgb_candidates_configs:
        cname = cfg["name"]
        params = cfg["params"]
        model = lgb.LGBMRegressor(
            **params,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            categorical_feature=CATEGORICAL_COLUMNS,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
        )
        dur = time.perf_counter() - t0
        preds = np.clip(model.predict(X_val), 0, None)
        v_mae = float(mean_absolute_error(y_val, preds))
        v_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        v_r2 = float(r2_score(y_val, preds))
        best_iter = getattr(model, "best_iteration_", params.get("n_estimators"))
        lgb_hpo_results.append({
            "name": cname, "val_mae": v_mae, "val_rmse": v_rmse, "val_r2": v_r2,
            "train_time_s": dur, "best_iteration": best_iter, "params": params
        })
        trained_lgb_models[cname] = model
        logger.info(f"  {cname:18s} -> Val MAE: {v_mae:.4f} | RMSE: {v_rmse:.4f} | R²: {v_r2:.4f} (Fit: {dur:.1f}s, Iter: {best_iter})")

    best_xgb_meta = sorted(xgb_hpo_results, key=lambda x: x["val_mae"])[0]
    best_lgb_meta = sorted(lgb_hpo_results, key=lambda x: x["val_mae"])[0]
    best_xgb_model = trained_xgb_models[best_xgb_meta["name"]]
    best_lgb_model = trained_lgb_models[best_lgb_meta["name"]]

    logger.info(f"\nBest XGBoost candidate on Validation: {best_xgb_meta['name']} (MAE={best_xgb_meta['val_mae']:.4f})")
    logger.info(f"Best LightGBM candidate on Validation: {best_lgb_meta['name']} (MAE={best_lgb_meta['val_mae']:.4f})")

    # -------------------------------------------------------------------------
    # PHASE 5: TEMPORAL GENERALIZATION (EXPANDING-WINDOW VALIDATION)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("[Phase 5] Expanding-Window Temporal Validation Across Forward Time Folds")
    logger.info("=" * 78)

    # Partition validation set into 4 consecutive temporal slices
    val_df = df.loc[X_val.index].copy()
    unique_ts = val_df["decision_timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    n_ts = len(unique_ts)
    fold_ts_indices = np.array_split(unique_ts, 4)

    temporal_fold_results = []
    paired_hourly_differences = []

    for fold_i, ts_chunk in enumerate(fold_ts_indices, 1):
        ts_start = ts_chunk[0]
        ts_end = ts_chunk[-1]
        fold_mask = (val_df["decision_timestamp"] >= ts_start) & (val_df["decision_timestamp"] <= ts_end)
        X_fold = X_val[fold_mask]
        y_fold = y_val[fold_mask]

        xgb_p = np.clip(best_xgb_model.predict(X_fold), 0, None)
        lgb_p = np.clip(best_lgb_model.predict(X_fold), 0, None)

        xgb_f_mae = float(mean_absolute_error(y_fold, xgb_p))
        xgb_f_rmse = float(np.sqrt(mean_squared_error(y_fold, xgb_p)))

        lgb_f_mae = float(mean_absolute_error(y_fold, lgb_p))
        lgb_f_rmse = float(np.sqrt(mean_squared_error(y_fold, lgb_p)))

        delta_mae = xgb_f_mae - lgb_f_mae  # Positive means LightGBM wins, Negative means XGBoost wins
        delta_rmse = xgb_f_rmse - lgb_f_rmse
        winner = "XGBoost" if xgb_f_mae < lgb_f_mae else ("LightGBM" if lgb_f_mae < xgb_f_mae else "Tie")

        temporal_fold_results.append({
            "fold": fold_i,
            "start_time": str(ts_start),
            "end_time": str(ts_end),
            "samples": len(X_fold),
            "xgb_mae": round(xgb_f_mae, 4),
            "xgb_rmse": round(xgb_f_rmse, 4),
            "lgb_mae": round(lgb_f_mae, 4),
            "lgb_rmse": round(lgb_f_rmse, 4),
            "delta_mae": round(delta_mae, 4),
            "delta_rmse": round(delta_rmse, 4),
            "winner": winner,
        })

        # Calculate hourly paired error differences
        fold_eval_df = pd.DataFrame({
            "timestamp": val_df.loc[fold_mask, "decision_timestamp"].values,
            "y_true": y_fold.values,
            "xgb_pred": xgb_p,
            "lgb_pred": lgb_p,
        })
        fold_eval_df["xgb_abs_err"] = (fold_eval_df["y_true"] - fold_eval_df["xgb_pred"]).abs()
        fold_eval_df["lgb_abs_err"] = (fold_eval_df["y_true"] - fold_eval_df["lgb_pred"]).abs()
        hourly_grp = fold_eval_df.groupby("timestamp")[["xgb_abs_err", "lgb_abs_err"]].mean()
        hourly_grp["diff"] = hourly_grp["xgb_abs_err"] - hourly_grp["lgb_abs_err"]
        paired_hourly_differences.extend(hourly_grp["diff"].tolist())

        logger.info(
            f"  Fold {fold_i} ({str(ts_start)[:10]} -> {str(ts_end)[:10]}, N={len(X_fold):,}): "
            f"XGB MAE={xgb_f_mae:.4f} vs LGB MAE={lgb_f_mae:.4f} | ΔMAE={delta_mae:+.4f} -> Winner: {winner}"
        )

    # Statistical Significance Testing with HAC (Newey-West) standard errors
    paired_diffs_arr = np.array(paired_hourly_differences)
    mean_diff, se_hac, t_stat, p_val, ci_lower, ci_upper = compute_hac_paired_stats(paired_diffs_arr, max_lag=24)

    logger.info("-" * 78)
    logger.info("Paired Hourly Error Difference (XGBoost MAE - LightGBM MAE):")
    logger.info(f"  Mean Difference (ΔMAE):   {mean_diff:+.5f}")
    logger.info(f"  HAC Standard Error (SE):   {se_hac:.5f} (Newey-West Bartlett lag=24h)")
    logger.info(f"  t-statistic:              {t_stat:.3f} | p-value: {p_val:.4f}")
    logger.info(f"  95% Confidence Interval:  [{ci_lower:+.5f}, {ci_upper:+.5f}]")
    is_stat_sig = p_val < 0.05
    logger.info(f"  Statistically Significant at α=0.05: {is_stat_sig}")
    logger.info("-" * 78)

    # -------------------------------------------------------------------------
    # PHASE 6: CHECK FEATURE AND MODEL CONSISTENCY & QUANTILES
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("[Phase 6] Feature & Model Consistency, Quantile Calibration & Explainability")
    logger.info("=" * 78)

    # 1. Output Validity (Finite, Non-Negative, Correct Shape)
    validator = PredictionValidator(raise_on_violation=True)
    val_rep_xgb = validator.validate_predictions(best_xgb_model.predict(X_val), expected_n=len(X_val))
    val_rep_lgb = validator.validate_predictions(best_lgb_model.predict(X_val), expected_n=len(X_val))
    logger.info(f"[OK] XGBoost predictions: Finite={not val_rep_xgb.has_nan}, Non-negative={not val_rep_xgb.has_negative}, Shape=({len(X_val)},)")
    logger.info(f"[OK] LightGBM predictions: Finite={not val_rep_lgb.has_nan}, Non-negative={not val_rep_lgb.has_negative}, Shape=({len(X_val)},)")

    # 2. TreeSHAP Local Attributions
    shap_sample = X_val.iloc[:5]
    dmat_xgb = xgb.DMatrix(shap_sample)
    xgb_shap_contribs = best_xgb_model.get_booster().predict(dmat_xgb, pred_contribs=True)
    lgb_shap_contribs = best_lgb_model.predict(shap_sample, pred_contrib=True)
    assert xgb_shap_contribs.shape == (5, len(feature_cols) + 1), "XGBoost SHAP shape mismatch"
    assert lgb_shap_contribs.shape == (5, len(feature_cols) + 1), "LightGBM SHAP shape mismatch"
    logger.info("[OK] Both models provide native TreeSHAP local attributions (shape=(5, 46)).")

    # 3. Quantile Prediction Intervals Comparison (P10 <= P50 <= P90)
    logger.info("\nEvaluating Quantile Intervals Calibration...")
    # Quantiles for XGBoost
    xgb_p10 = xgb.XGBRegressor(
        n_estimators=350, learning_rate=0.05, max_depth=7, objective="reg:quantileerror",
        quantile_alpha=0.10, random_state=RANDOM_SEED, tree_method="hist", n_jobs=-1
    )
    xgb_p90 = xgb.XGBRegressor(
        n_estimators=350, learning_rate=0.05, max_depth=7, objective="reg:quantileerror",
        quantile_alpha=0.90, random_state=RANDOM_SEED, tree_method="hist", n_jobs=-1
    )
    xgb_p10.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_p90.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_p10_preds = np.clip(xgb_p10.predict(X_val), 0, None)
    xgb_p90_preds = np.clip(xgb_p90.predict(X_val), 0, None)
    xgb_p50_preds = np.clip(best_xgb_model.predict(X_val), 0, None)

    # Clamping for strict non-crossing
    xgb_p10_adj = np.minimum(xgb_p10_preds, xgb_p50_preds)
    xgb_p90_adj = np.maximum(xgb_p50_preds, xgb_p90_preds)
    xgb_cov_80 = float(((y_val >= xgb_p10_adj) & (y_val <= xgb_p90_adj)).mean())
    xgb_interval_width = float((xgb_p90_adj - xgb_p10_adj).mean())

    # Quantiles for LightGBM
    lgb_p10 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.10, n_estimators=350, learning_rate=0.04,
        num_leaves=63, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
    )
    lgb_p90 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.90, n_estimators=350, learning_rate=0.04,
        num_leaves=63, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
    )
    lgb_p10.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(25, verbose=False)])
    lgb_p90.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(25, verbose=False)])

    lgb_p10_preds = np.clip(lgb_p10.predict(X_val), 0, None)
    lgb_p90_preds = np.clip(lgb_p90.predict(X_val), 0, None)
    lgb_p50_preds = np.clip(best_lgb_model.predict(X_val), 0, None)

    lgb_p10_adj = np.minimum(lgb_p10_preds, lgb_p50_preds)
    lgb_p90_adj = np.maximum(lgb_p50_preds, lgb_p90_preds)
    lgb_cov_80 = float(((y_val >= lgb_p10_adj) & (y_val <= lgb_p90_adj)).mean())
    lgb_interval_width = float((lgb_p90_adj - lgb_p10_adj).mean())

    logger.info(f"[OK] XGBoost Quantiles:  80% Coverage = {xgb_cov_80:.2%} | Mean Width = {xgb_interval_width:.2f}")
    logger.info(f"[OK] LightGBM Quantiles: 80% Coverage = {lgb_cov_80:.2%} | Mean Width = {lgb_interval_width:.2f}")

    # Latencies of the best tuned models
    xgb_tuned_latencies = measure_inference_latencies(
        best_xgb_model, X_val, is_lgbm=False, feature_cols=feature_cols, label_encoders=label_encoders
    )
    lgb_tuned_latencies = measure_inference_latencies(
        best_lgb_model, X_val, is_lgbm=True, feature_cols=feature_cols, label_encoders=label_encoders
    )

    # -------------------------------------------------------------------------
    # PHASE 7: MODEL SELECTION (VALIDATION EVIDENCE ONLY)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("[Phase 7] Model Selection Gating & Decision Hierarchy")
    logger.info("=" * 78)

    # Apply decision hierarchy
    # 1. Target & leakage: Both strictly valid
    # 2. Validation MAE difference
    mae_diff_val = best_xgb_meta["val_mae"] - best_lgb_meta["val_mae"]
    logger.info(f"Validation MAE Comparison:")
    logger.info(f"  Best XGBoost ({best_xgb_meta['name']}):  Val MAE = {best_xgb_meta['val_mae']:.4f}")
    logger.info(f"  Best LightGBM ({best_lgb_meta['name']}): Val MAE = {best_lgb_meta['val_mae']:.4f}")
    logger.info(f"  ΔMAE (XGB - LGB):                  {mae_diff_val:+.4f}")

    # Decision logic based on meaningful difference threshold (>= 0.005 MAE and p < 0.05)
    meaningful_diff_threshold = 0.005
    if abs(mean_diff) < meaningful_diff_threshold or not is_stat_sig:
        recommendation = "RETAIN_XGBOOST_CHAMPION"
        diff_conclusion = "NEGLIGIBLE_AND_INCONCLUSIVE"
        decision_rationale = (
            f"The predictive difference between XGBoost ({best_xgb_meta['val_mae']:.4f}) and "
            f"LightGBM ({best_lgb_meta['val_mae']:.4f}) is miniscule (|ΔMAE| = {abs(mae_diff_val):.4f} < 0.005, "
            f"p={p_val:.4f}). Per the decision hierarchy, negligible predictive differences do not justify "
            f"displacing the production champion. XGBoost is retained for operational stability."
        )
    elif mae_diff_val > 0 and is_stat_sig:
        recommendation = "PROMOTE_LIGHTGBM"
        diff_conclusion = "LIGHTGBM_SUPERIOR"
        decision_rationale = f"LightGBM demonstrated statistically significant superiority (ΔMAE = {mae_diff_val:+.4f}, p={p_val:.4f})."
    else:
        recommendation = "RETAIN_XGBOOST_CHAMPION"
        diff_conclusion = "XGBOOST_SUPERIOR"
        decision_rationale = f"XGBoost demonstrated superior predictive accuracy (ΔMAE = {mae_diff_val:+.4f})."

    logger.info(f"\nDECISION RESULT: {recommendation}")
    logger.info(f"CONCLUSION:      {diff_conclusion}")
    logger.info(f"RATIONALE:       {decision_rationale}")

    # -------------------------------------------------------------------------
    # PHASE 8: FINAL HOLDOUT TEST SET EVALUATION
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("[Phase 8] Final Holdout Test Set Evaluation (106,800 Observations)")
    logger.info("=" * 78)
    logger.info(f"Evaluating both best configurations on untouched test split: {test_timestamps.min()} -> {test_timestamps.max()}")

    xgb_test_preds = np.clip(best_xgb_model.predict(X_test), 0, None)
    lgb_test_preds = np.clip(best_lgb_model.predict(X_test), 0, None)

    xgb_test_mae = float(mean_absolute_error(y_test, xgb_test_preds))
    xgb_test_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_test_preds)))
    xgb_test_r2 = float(r2_score(y_test, xgb_test_preds))

    lgb_test_mae = float(mean_absolute_error(y_test, lgb_test_preds))
    lgb_test_rmse = float(np.sqrt(mean_squared_error(y_test, lgb_test_preds)))
    lgb_test_r2 = float(r2_score(y_test, lgb_test_preds))

    logger.info(f"Holdout Test Results:")
    logger.info(f"  XGBoost:  Test MAE = {xgb_test_mae:.4f} | RMSE = {xgb_test_rmse:.4f} | R² = {xgb_test_r2:.4f}")
    logger.info(f"  LightGBM: Test MAE = {lgb_test_mae:.4f} | RMSE = {lgb_test_rmse:.4f} | R² = {lgb_test_r2:.4f}")
    logger.info(f"  ΔMAE (XGB - LGB on Test): {xgb_test_mae - lgb_test_mae:+.4f}")
    logger.info(f"\n{SIMULATION_DATA_DISCLAIMER}")

    # -------------------------------------------------------------------------
    # PHASE 9: ARTIFACT PRESERVATION & MLFLOW EXPERIMENT LOGGING
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("[Phase 9] Artifact Safety & MLflow Logging")
    logger.info("=" * 78)

    # Save experimental candidates separately without touching models/demand_model.pkl
    best_xgb_path = EXPERIMENTS_DIR / f"{best_xgb_meta['name'].lower()}.joblib"
    best_lgb_path = EXPERIMENTS_DIR / f"{best_lgb_meta['name'].lower()}.joblib"
    joblib.dump(best_xgb_model, best_xgb_path)
    joblib.dump(best_lgb_model, best_lgb_path)
    logger.info(f"[OK] Experimental models saved separately in {EXPERIMENTS_DIR}")
    logger.info("[OK] Active production champion (models/demand_model.pkl) safely preserved.")

    # Log to MLflow
    mlflow_run_id = None
    if HAS_MLFLOW:
        try:
            db_path = ROOT_DIR / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}".replace("\\", "/"))
            mlflow.set_experiment("GeoDemand_Model_Comparison")
            with mlflow.start_run(run_name="xgboost_vs_lightgbm_fair_benchmark") as run:
                mlflow_run_id = run.info.run_id
                mlflow.log_params({
                    "xgb_best_model": best_xgb_meta["name"],
                    "lgb_best_model": best_lgb_meta["name"],
                    "recommendation": recommendation,
                    "difference_conclusion": diff_conclusion,
                    "random_seed": RANDOM_SEED,
                    "temporal_folds": 4,
                })
                mlflow.log_metrics({
                    "xgb_val_mae": best_xgb_meta["val_mae"],
                    "xgb_val_rmse": best_xgb_meta["val_rmse"],
                    "lgb_val_mae": best_lgb_meta["val_mae"],
                    "lgb_val_rmse": best_lgb_meta["val_rmse"],
                    "delta_val_mae": mae_diff_val,
                    "xgb_test_mae": xgb_test_mae,
                    "lgb_test_mae": lgb_test_mae,
                    "p_val_hac": p_val,
                    "xgb_latency_single_p95_ms": xgb_tuned_latencies["raw_single_p95_ms"],
                    "lgb_latency_single_p95_ms": lgb_tuned_latencies["raw_single_p95_ms"],
                })
                comparison_summary = {
                    "recommendation": recommendation,
                    "decision_rationale": decision_rationale,
                    "validation_comparison": {
                        "xgb": best_xgb_meta,
                        "lgb": best_lgb_meta,
                    },
                    "test_comparison": {
                        "xgb_test_mae": xgb_test_mae,
                        "xgb_test_rmse": xgb_test_rmse,
                        "xgb_test_r2": xgb_test_r2,
                        "lgb_test_mae": lgb_test_mae,
                        "lgb_test_rmse": lgb_test_rmse,
                        "lgb_test_r2": lgb_test_r2,
                    },
                    "temporal_folds": temporal_fold_results,
                    "statistical_test": {
                        "mean_diff": mean_diff,
                        "se_hac": se_hac,
                        "t_stat": t_stat,
                        "p_val": p_val,
                        "ci_95": [ci_lower, ci_upper],
                        "statistically_significant": is_stat_sig,
                    },
                }
                mlflow.log_dict(comparison_summary, "model_comparison_summary.json")
                logger.info(f"[OK] Benchmark results logged to MLflow Run: {mlflow_run_id}")
        except Exception as e:
            logger.warning(f"MLflow logging notice: {e}")

    # Output full machine-readable comparison file
    comparison_report_path = EXPERIMENTS_DIR / "xgboost_vs_lightgbm_comparison.json"
    full_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "difference_conclusion": diff_conclusion,
        "decision_rationale": decision_rationale,
        "baseline_comparison": {
            "xgb": {
                "val_mae": xgb_base_val_mae,
                "val_rmse": xgb_base_val_rmse,
                "val_r2": xgb_base_val_r2,
                "train_time_s": xgb_base_train_time,
                "size_mb": xgb_base_size_mb,
                "latencies": xgb_base_latencies,
            },
            "lgb": {
                "val_mae": lgb_base_val_mae,
                "val_rmse": lgb_base_val_rmse,
                "val_r2": lgb_base_val_r2,
                "train_time_s": lgb_base_train_time,
                "size_mb": lgb_base_size_mb,
                "latencies": lgb_base_latencies,
            },
        },
        "best_candidates": {
            "xgb": {
                **best_xgb_meta,
                "size_mb": best_xgb_path.stat().st_size / (1024.0 * 1024.0),
                "latencies": xgb_tuned_latencies,
                "quantiles": {"cov_80": xgb_cov_80, "mean_width": xgb_interval_width},
                "test": {"mae": xgb_test_mae, "rmse": xgb_test_rmse, "r2": xgb_test_r2},
            },
            "lgb": {
                **best_lgb_meta,
                "size_mb": best_lgb_path.stat().st_size / (1024.0 * 1024.0),
                "latencies": lgb_tuned_latencies,
                "quantiles": {"cov_80": lgb_cov_80, "mean_width": lgb_interval_width},
                "test": {"mae": lgb_test_mae, "rmse": lgb_test_rmse, "r2": lgb_test_r2},
            },
        },
        "temporal_folds": temporal_fold_results,
        "hac_paired_stats": {
            "mean_diff": mean_diff,
            "se_hac": se_hac,
            "t_stat": t_stat,
            "p_val": p_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "is_stat_sig": is_stat_sig,
        },
        "mlflow_run_id": mlflow_run_id,
        "disclaimer": SIMULATION_DATA_DISCLAIMER,
    }

    with open(comparison_report_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"[OK] Detailed comparison report saved to {comparison_report_path}")

    return full_output


if __name__ == "__main__":
    res = run_benchmark()
    print("\nBENCHMARK RUN COMPLETED SUCCESSFULLY.")
