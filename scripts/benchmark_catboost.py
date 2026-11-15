"""
GeoDemand AI — CatBoost Candidate Benchmark Script (Phase 7)

Member 3 — Machine Learning & MLOps

Executes rigorous candidate model benchmarking:
- Same 70/15/15 chronological split
- Preprocessing and encoders fit on training set only
- Validation set evaluation (MAE, RMSE, R²) without touching test set for selection
- Inference latency measurement (per-sample latency in milliseconds)
- MLflow experiment logging
- Preservation of XGBoost champion unless CatBoost achieves meaningful improvement
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.geodemand.models.train_demand_model import (
    load_and_validate,
    prepare_features,
    chronological_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark_catboost")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def run_catboost_benchmark() -> Dict[str, Any]:
    if not HAS_CATBOOST:
        logger.warning("CatBoost is not installed — skipping CatBoost benchmark.")
        return {"status": "skipped", "reason": "catboost_not_installed"}

    logger.info("=" * 70)
    logger.info("GeoDemand Step 4 Phase 7: CatBoost Benchmark Evaluation")
    logger.info("=" * 70)

    # 1. Load data
    logger.info("Loading canonical feature store...")
    df = load_and_validate()

    # 2. Extract features and encoders strictly on train mask
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_mask = pd.Series([False] * n)
    train_mask.iloc[:train_end] = True

    X, y, feature_cols, label_encoders = prepare_features(df, train_mask=train_mask)
    timestamps = df["decision_timestamp"]

    # 3. Chronological split
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y, timestamps)
    logger.info(f"Split sizes: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    # Load champion metrics for comparison
    meta_path = ROOT_DIR / "models" / "demand_model_meta.json"
    with open(meta_path) as f:
        champion_meta = json.load(f)
    champ_val_mae = champion_meta.get("metrics", {}).get("val_mae", 0.7738)
    champ_val_rmse = champion_meta.get("metrics", {}).get("val_rmse", 1.0045)
    logger.info(f"Current Champion (XGBoost) Validation Metrics: MAE={champ_val_mae:.4f}, RMSE={champ_val_rmse:.4f}")

    # 4. Train CatBoost candidate on Train split only
    logger.info("Training CatBoost candidate model on Train split...")
    model_cb = cb.CatBoostRegressor(
        iterations=400,
        learning_rate=0.08,
        depth=6,
        loss_function="MAE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100,
        thread_count=-1,
    )

    t0_train = time.perf_counter()
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
    train_time = time.perf_counter() - t0_train
    logger.info(f"✓ CatBoost training finished in {train_time:.2f} seconds.")

    # 5. Validation metrics (DO NOT USE TEST SET FOR MODEL SELECTION)
    val_preds = np.clip(model_cb.predict(X_val), 0, None)
    val_mae = float(mean_absolute_error(y_val, val_preds))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_r2 = float(r2_score(y_val, val_preds))
    logger.info(f"CatBoost Validation Results: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}")

    # 6. Measure inference latency consistently (per-sample latency)
    logger.info("Benchmarking inference latency...")
    warmup_batch = X_val.iloc[:100]
    _ = model_cb.predict(warmup_batch)

    latencies = []
    sample_batch = X_val.iloc[:500]
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model_cb.predict(sample_batch)
        dt = (time.perf_counter() - t0) * 1000.0 / len(sample_batch)  # ms per sample
        latencies.append(dt)

    median_latency_ms = float(np.median(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    logger.info(f"CatBoost Latency: Median={median_latency_ms:.4f} ms/sample, P95={p95_latency_ms:.4f} ms/sample")

    # 7. Comparison with Champion
    mae_diff = val_mae - champ_val_mae
    rmse_diff = val_rmse - champ_val_rmse
    mae_improvement_pct = ((champ_val_mae - val_mae) / champ_val_mae) * 100.0

    logger.info("-" * 50)
    logger.info("Champion vs Candidate Comparison (Validation Set):")
    logger.info(f"  Champion (XGBoost):  MAE = {champ_val_mae:.4f} | RMSE = {champ_val_rmse:.4f}")
    logger.info(f"  Candidate (CatBoost): MAE = {val_mae:.4f} | RMSE = {val_rmse:.4f}")
    logger.info(f"  Delta MAE:            {mae_diff:+.4f} ({mae_improvement_pct:+.2f}%)")
    logger.info("-" * 50)

    # Meaningful improvement criterion: >= 1.0% improvement in validation MAE
    meaningful_improvement = val_mae < (champ_val_mae - 0.01)
    if meaningful_improvement:
        decision = "RECOMMEND_REPLACEMENT"
        logger.info("Decision: CatBoost demonstrates meaningful improvement over champion.")
    else:
        decision = "RETAIN_CHAMPION"
        logger.info("Decision: RETAIN XGBOOST CHAMPION. CatBoost does not show worthwhile improvement.")

    # 8. Log results to MLflow
    if HAS_MLFLOW:
        try:
            db_path = ROOT_DIR / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}".replace("\\", "/"))
            mlflow.set_experiment("GeoDemand")
            with mlflow.start_run(run_name="catboost_benchmark_candidate"):
                mlflow.log_params({
                    "model_type": "catboost",
                    "iterations": 400,
                    "depth": 6,
                    "learning_rate": 0.08,
                    "split_type": "chronological_70_15_15",
                    "evaluation_set": "validation_only",
                    "benchmark_decision": decision,
                })
                mlflow.log_metrics({
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                    "train_time_sec": train_time,
                    "latency_median_ms": median_latency_ms,
                    "latency_p95_ms": p95_latency_ms,
                    "delta_val_mae_vs_champ": mae_diff,
                })
            logger.info("✓ CatBoost benchmark logged to MLflow successfully.")
        except Exception as e:
            logger.warning(f"MLflow benchmark logging note: {e}")

    results = {
        "candidate": "CatBoost",
        "champion": "XGBoost",
        "validation_metrics": {
            "catboost_val_mae": round(val_mae, 4),
            "catboost_val_rmse": round(val_rmse, 4),
            "catboost_val_r2": round(val_r2, 4),
            "champion_val_mae": round(champ_val_mae, 4),
            "champion_val_rmse": round(champ_val_rmse, 4),
        },
        "latency_ms_per_sample": {
            "median": round(median_latency_ms, 4),
            "p95": round(p95_latency_ms, 4),
        },
        "decision": decision,
        "champion_retained": decision == "RETAIN_CHAMPION",
    }
    return results


if __name__ == "__main__":
    res = run_catboost_benchmark()
    print("\nBENCHMARK SUMMARY:")
    print(json.dumps(res, indent=2))
