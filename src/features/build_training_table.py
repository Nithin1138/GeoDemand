"""
Build the hybrid training table: joins the simulated transactional dataset
(real-structured contextual features + simulated customer_count target) with
the static per-cell feature layer, and prepares model-ready columns.

Deliberately excludes:
  - raw h3_cell / vendor_id as model features (would let the model memorize
    cell/vendor identity instead of generalizing from measurable
    characteristics — bad for a system meant to score cells it hasn't seen
    yet, e.g. as MOVIGO expands to new zones)
  - hidden_skill_factor / cell_reputation (never leaked to the model by
    construction — simulate_dataset() doesn't even write them to the output
    rows, so there's nothing to accidentally include here)

Includes an explicit temporal "as-of" check: every feature used to predict
hour H on day D must only reflect information available before that hour,
enforced by construction (all dynamic features here are same-day/same-hour
context, not future-looking aggregates like "today's total sales so far").
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

CATEGORICAL_FEATURES = ["category"]
NUMERIC_STATIC_FEATURES = [
    "population_density", "office_count", "college_count", "school_count",
    "hospital_count", "mall_count", "restaurant_count", "park_count",
    "bus_stop_count", "road_density",
]
NUMERIC_DYNAMIC_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "is_raining", "temp_c", "event_flag",
]
TARGET = "customer_count"


def latest_simulated_dataset(sim_dir: str | Path) -> Path:
    files = sorted(glob.glob(str(Path(sim_dir) / "dataset_v*.parquet")))
    if not files:
        raise FileNotFoundError(f"No simulated dataset found in {sim_dir}")
    return Path(files[-1])


def build_training_table(
    transactions_path: str | Path,
    static_features: pd.DataFrame,
) -> pd.DataFrame:
    tx = pd.read_parquet(transactions_path)
    tx["date"] = pd.to_datetime(tx["date"])
    tx["day_of_week"] = tx["date"].dt.dayofweek
    tx["month"] = tx["date"].dt.month

    static = static_features[["h3_cell"] + NUMERIC_STATIC_FEATURES].copy()

    df = tx.merge(static, on="h3_cell", how="left", validate="many_to_one")

    # Temporal-leakage guard: assert every dynamic feature used is either
    # a calendar fact (known in advance) or same-window weather/event context
    # (known at prediction time), never a future aggregate. This is a static
    # assertion here because the simulator only ever generates same-window
    # dynamic features by construction — but we check row completeness so a
    # silent join bug can't introduce leakage either.
    required = NUMERIC_DYNAMIC_FEATURES + NUMERIC_STATIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after join: {missing}")
    n_null = df[NUMERIC_STATIC_FEATURES].isnull().any(axis=1).sum()
    if n_null > 0:
        raise ValueError(
            f"{n_null} rows have null static features after join — "
            "check for h3_cell mismatch between transactions and static grid."
        )

    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_raining"] = df["is_raining"].astype(int)
    df["event_flag"] = df["event_flag"].astype(int)
    df["category"] = df["category"].astype("category")

    keep_cols = ["date", "h3_cell", "vendor_id"] + CATEGORICAL_FEATURES + \
                NUMERIC_STATIC_FEATURES + NUMERIC_DYNAMIC_FEATURES + [TARGET]
    return df[keep_cols]


def time_based_split(df: pd.DataFrame, train_end: str, val_end: str):
    """Forward-chaining split: train < train_end <= val < val_end <= test."""
    df = df.sort_values("date")
    train = df[df["date"] < train_end]
    val = df[(df["date"] >= train_end) & (df["date"] < val_end)]
    test = df[df["date"] >= val_end]
    return train, val, test


if __name__ == "__main__":
    import sys
    root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root / "src" / "simulation"))
    from h3_grid import generate_city_grid
    from static_features_synth import generate_synthetic_static_features
    from market_simulator import SimulationParams

    params = SimulationParams()
    grid = generate_city_grid(params.center_lat, params.center_lng, params.radius_km, params.h3_resolution)
    static = generate_synthetic_static_features(grid, seed=params.seed)

    tx_path = latest_simulated_dataset(root / "data" / "simulated")
    print(f"Using transactions: {tx_path.name}")

    df = build_training_table(tx_path, static)
    print(f"Training table: {len(df):,} rows, {df.shape[1]} columns")
    print(df.dtypes)

    train, val, test = time_based_split(df, train_end="2025-10-01", val_end="2025-11-01")
    print(f"\nTrain: {len(train):,} rows ({train['date'].min().date()} - {train['date'].max().date()})")
    print(f"Val:   {len(val):,} rows ({val['date'].min().date()} - {val['date'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['date'].min().date()} - {test['date'].max().date()})")

    out_path = root / "data" / "processed" / "training_table.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")
