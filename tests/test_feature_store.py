"""
Test Suite for Feature Store Contract, 40 Features, and Temporal Split (Member 2).
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.features.feature_store import (
    LEAKAGE_COLUMNS,
    TARGET,
    trainable_columns,
    time_based_split,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@pytest.fixture(scope="module")
def feature_store_df():
    fs_path = PROCESSED_DIR / "feature_store.parquet"
    assert fs_path.exists(), f"Feature store not found at {fs_path}"
    return pd.read_parquet(fs_path)


def test_feature_store_has_40_trainable_features(feature_store_df):
    """The production LightGBM model requires exactly 40 numeric/categorical features."""
    train_cols = trainable_columns(feature_store_df)
    assert len(train_cols) == 40, f"Expected exactly 40 trainable features, found {len(train_cols)}: {train_cols}"


def test_feature_store_single_ml_target_only(feature_store_df):
    """The ML target is strictly 'expected_customer_count'. Business metrics are forbidden."""
    assert TARGET in feature_store_df.columns
    assert TARGET == "expected_customer_count"

    forbidden_targets = ["revenue", "profit", "ingredient_cost", "fuel_cost", "operating_cost", "decision"]
    train_cols = trainable_columns(feature_store_df)
    for forbidden in forbidden_targets:
        assert forbidden not in train_cols, f"Forbidden business metric {forbidden} leaked into model features!"


def test_feature_store_zero_null_values(feature_store_df):
    """Every single row and feature must be fully populated with 0 nulls."""
    total_nulls = int(feature_store_df.isnull().sum().sum())
    assert total_nulls == 0, f"Feature store contains {total_nulls} null values!"


def test_feature_store_identity_stripping(feature_store_df):
    """Raw identifiers and timestamps must be stripped from trainable features."""
    train_cols = trainable_columns(feature_store_df)
    forbidden_ids = {
        "h3_cell_id", "vendor_id", "decision_timestamp",
        "target_timestamp", "timestamp", "date", TARGET,
    }
    overlap = set(train_cols).intersection(forbidden_ids)
    assert not overlap, f"Forbidden identifiers found in trainable feature set: {overlap}"


def test_feature_store_zero_latent_simulation_leakage(feature_store_df):
    """No hidden simulation factors may exist anywhere in feature_store.parquet."""
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in feature_store_df.columns, (
            f"Hidden simulation factor {leak_col} leaked into feature store!"
        )


def test_feature_store_chronological_time_based_split(feature_store_df):
    """Chronological splitting must ensure Train < Val < Test with 0 overlap."""
    train, val, test = time_based_split(
        feature_store_df, train_end="2025-09-01", val_end="2025-11-01"
    )

    assert len(train) > 0, "Train split empty"
    assert len(val) > 0, "Validation split empty"
    assert len(test) > 0, "Test split empty"
    assert len(train) + len(val) + len(test) == len(feature_store_df)

    # Monotonic timestamps
    assert train["decision_timestamp"].max() < val["decision_timestamp"].min()
    assert val["decision_timestamp"].max() < test["decision_timestamp"].min()
