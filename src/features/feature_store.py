"""
Dataset 10 — Feature Store (feature_store.parquet)

Single responsibility: THE dataset used to train the ML model. Everything
before this point (h3_cells, static_features, weather_hourly, calendar,
events, competition, vendor_profiles, historical_transactions) has its own
single responsibility and its own versioning; this module's only job is
to join them correctly and enforce the contract that makes the join safe
to train on:

  - Exactly ONE target column: expected_customer_count.
  - NO leakage columns: hidden_skill_factor, popularity_score, quality_score,
    repeat_customer_rate, cell_reputation are vendor/cell latent factors
    used to GENERATE the target in market_simulation_v2.py — they are
    dropped here by construction, not just "not selected", so there's
    nothing to accidentally leak even if someone edits the column list.
  - NO raw identity columns as features (h3_cell_id, vendor_id) — kept for
    joins/grouping but excluded from the trainable feature set, so the
    model has to generalize from measurable characteristics rather than
    memorizing which cell/vendor produced which number.
  - Explicit temporal safety: every dynamic feature joined here is
    same-hour context (weather/calendar at the prediction timestamp) or
    known-in-advance (holidays, static POIs) — never a future aggregate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns that exist upstream ONLY to generate the simulated target and
# must never reach the trainable feature set.
LEAKAGE_COLUMNS = [
    "hidden_skill_factor", "popularity_score", "quality_score",
    "repeat_customer_rate", "cell_reputation",
]

STATIC_FEATURE_COLUMNS = [
    "population_density", "road_density", "office_count", "school_count",
    "college_count", "hospital_count", "restaurant_count", "mall_count",
    "park_count", "bus_stop_count", "railway_station_count",
    "metro_station_count", "parking_count", "building_density",
    "residential_ratio", "commercial_ratio", "industrial_ratio",
]

CALENDAR_FEATURE_COLUMNS = [
    "hour", "weekday", "month", "season", "quarter",
    "is_weekend", "is_holiday", "school_vacation",
]

WEATHER_FEATURE_COLUMNS = [
    "temperature", "humidity", "rainfall", "wind_speed",
    "pressure", "cloud_cover", "visibility", "weather_condition", "uv_index",
]

VENDOR_FEATURE_COLUMNS = [
    "vendor_category", "inventory_capacity", "preparation_time",
]
# average_order_value already exists on historical_transactions (per-transaction,
# vendor-specific) — pulling it again from vendor_profiles would collide on merge.

TARGET = "expected_customer_count"


def build_feature_store(
    historical_transactions: pd.DataFrame,
    static_features: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    calendar: pd.DataFrame,
    competition: pd.DataFrame,
    vendor_profiles: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constructs the feature store under the next-hour prediction contract:
      - Features are joined strictly as-of decision_timestamp (t).
      - Target expected_customer_count reflects customer arrivals during target_timestamp (t + 1h).
      - Latent variables (LEAKAGE_COLUMNS) are strictly dropped.
    """
    tx = historical_transactions.copy()
    tx["date"] = tx["decision_timestamp"].dt.floor("D")

    weather = weather_hourly[["timestamp"] + WEATHER_FEATURE_COLUMNS].drop_duplicates(
        subset=["timestamp"]
    ).rename(columns={"timestamp": "decision_timestamp"})
    cal = calendar[["timestamp"] + CALENDAR_FEATURE_COLUMNS].drop_duplicates(
        subset=["timestamp"]
    ).rename(columns={"timestamp": "decision_timestamp"})

    df = tx.merge(
        static_features[["h3_cell_id"] + STATIC_FEATURE_COLUMNS], on="h3_cell_id", how="left", validate="many_to_one"
    ).merge(
        weather, on="decision_timestamp", how="left", validate="many_to_one"
    ).merge(
        cal, on="decision_timestamp", how="left", validate="many_to_one"
    ).merge(
        competition[["date", "h3_cell_id", "competition_score"]], on=["date", "h3_cell_id"], how="left", validate="many_to_one"
    ).merge(
        vendor_profiles[["vendor_id"] + VENDOR_FEATURE_COLUMNS], on="vendor_id", how="left", validate="many_to_one"
    )

    event_lookup = events.groupby([events["timestamp"].dt.floor("D"), "h3_cell_id"])["event_importance"].max()
    df["event_importance"] = df.apply(
        lambda r: event_lookup.get((r["date"], r["h3_cell_id"]), 0.0), axis=1
    )

    # --- explicit leakage guard: assert none of these ever made it into the join ---
    leaked = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leaked:
        raise ValueError(f"Leakage columns present in feature store, must fix join: {leaked}")

    # --- explicit null guard: a join mismatch would silently produce NaN features ---
    check_cols = STATIC_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS + CALENDAR_FEATURE_COLUMNS
    n_null = df[check_cols].isnull().any(axis=1).sum()
    if n_null > 0:
        raise ValueError(
            f"{n_null} rows have null features after join — check for h3_cell_id/decision_timestamp "
            "mismatches between historical_transactions and the upstream datasets."
        )

    feature_cols = (
        ["decision_timestamp", "target_timestamp", "h3_cell_id", "vendor_id"]  # kept for grouping/audit, excluded from training below
        + STATIC_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS + CALENDAR_FEATURE_COLUMNS
        + ["competition_score", "event_importance"] + VENDOR_FEATURE_COLUMNS + ["average_order_value", TARGET]
    )
    return df[feature_cols]


def trainable_columns(feature_store: pd.DataFrame) -> list[str]:
    """The actual feature list to hand to a model — excludes identity/grouping columns and the target."""
    exclude = {"decision_timestamp", "target_timestamp", "timestamp", "date", "h3_cell_id", "vendor_id", TARGET}
    return [c for c in feature_store.columns if c not in exclude]


def time_based_split(df: pd.DataFrame, train_end: str, val_end: str):
    """Forward-chaining split: train < train_end <= val < val_end <= test. See Section 5.11 of the master doc."""
    split_col = "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"
    df = df.sort_values(split_col)
    train = df[df[split_col] < train_end]
    val = df[(df[split_col] >= train_end) & (df[split_col] < val_end)]
    test = df[df[split_col] >= val_end]
    return train, val, test


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    processed = root / "data" / "processed"

    tx = pd.read_parquet(processed / "historical_transactions.parquet")
    static = pd.read_parquet(processed / "static_features.parquet")
    weather = pd.read_parquet(processed / "weather_hourly.parquet")
    cal = pd.read_parquet(processed / "calendar.parquet")
    comp = pd.read_parquet(processed / "competition.parquet")
    vendors = pd.read_parquet(processed / "vendor_profiles.parquet")
    events = pd.read_parquet(processed / "events.parquet")

    fs = build_feature_store(tx, static, weather, cal, comp, vendors, events)
    out = processed / "feature_store.parquet"
    fs.to_parquet(out, index=False)

    print(f"feature_store.parquet: {len(fs):,} rows, {fs.shape[1]} columns -> {out}")
    print(f"Trainable feature columns ({len(trainable_columns(fs))}): {trainable_columns(fs)}")

    train, val, test = time_based_split(fs, train_end="2025-10-01", val_end="2025-11-01")
    print(f"\nTrain: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
