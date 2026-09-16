"""
Comprehensive Test Suite for GeoDemand AI Data Pipeline (v2.1 Next-Hour Contract).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
import pytest

from src.features.feature_store import LEAKAGE_COLUMNS, trainable_columns
from src.validation.qualitative_checks import run_all_checks

PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@pytest.fixture(scope="module")
def datasets():
    assert PROCESSED_DIR.exists(), f"Processed directory {PROCESSED_DIR} does not exist"
    return {
        "h3_cells": pd.read_parquet(PROCESSED_DIR / "h3_cells.parquet"),
        "static_features": pd.read_parquet(PROCESSED_DIR / "static_features.parquet"),
        "weather_hourly": pd.read_parquet(PROCESSED_DIR / "weather_hourly.parquet"),
        "calendar": pd.read_parquet(PROCESSED_DIR / "calendar.parquet"),
        "events": pd.read_parquet(PROCESSED_DIR / "events.parquet"),
        "competition": pd.read_parquet(PROCESSED_DIR / "competition.parquet"),
        "vendor_profiles": pd.read_parquet(PROCESSED_DIR / "vendor_profiles.parquet"),
        "products": pd.read_parquet(PROCESSED_DIR / "products.parquet"),
        "historical_transactions": pd.read_parquet(PROCESSED_DIR / "historical_transactions.parquet"),
        "feature_store": pd.read_parquet(PROCESSED_DIR / "feature_store.parquet"),
    }


def test_required_files_exist():
    expected_files = [
        "h3_cells.parquet", "static_features.parquet", "weather_hourly.parquet",
        "calendar.parquet", "events.parquet", "competition.parquet",
        "vendor_profiles.parquet", "products.parquet",
        "historical_transactions.parquet", "feature_store.parquet",
    ]
    for fname in expected_files:
        p = PROCESSED_DIR / fname
        assert p.exists() and p.stat().st_size > 0, f"File {fname} missing or empty"


def test_next_hour_temporal_contract(datasets):
    tx = datasets["historical_transactions"]
    assert "decision_timestamp" in tx.columns, "decision_timestamp missing from historical_transactions"
    assert "target_timestamp" in tx.columns, "target_timestamp missing from historical_transactions"
    assert "expected_customer_count" in tx.columns, "expected_customer_count target missing"

    delta = tx["target_timestamp"] - tx["decision_timestamp"]
    expected_delta = pd.Timedelta(hours=1)
    assert (delta == expected_delta).all(), "All transactions must exhibit exact 1-hour lag (decision @ t -> target @ t+1)"


def test_anti_circularity_protection(datasets):
    fs = datasets["feature_store"]
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in fs.columns, f"Latent simulation factor {leak_col} leaked into feature store!"

    train_cols = trainable_columns(fs)
    forbidden_train_cols = {
        "decision_timestamp", "target_timestamp", "timestamp", "h3_cell_id", "vendor_id", "date",
        "expected_customer_count", "customer_count",
    }
    overlap = set(train_cols).intersection(forbidden_train_cols)
    assert not overlap, f"Forbidden identifiers / target found in trainable columns: {overlap}"
    assert len(train_cols) >= 30, f"Expected 30+ trainable features, found {len(train_cols)}"


def test_feature_store_completeness_and_no_nulls(datasets):
    fs = datasets["feature_store"]
    assert len(fs) > 500_000, f"Expected 500k+ rows, got {len(fs)}"
    null_counts = fs.isnull().sum()
    assert (null_counts == 0).all(), f"Feature store contains null values:\n{null_counts[null_counts > 0]}"


def test_qualitative_business_checks_pass(datasets):
    tx_sample = datasets["historical_transactions"].sample(n=min(50_000, len(datasets["historical_transactions"])), random_state=42)
    results = run_all_checks(
        tx=tx_sample,
        static_features=datasets["static_features"],
        weather_hourly=datasets["weather_hourly"],
        calendar=datasets["calendar"],
        vendor_profiles=datasets["vendor_profiles"],
    )
    failed = results[results["passed"].str.contains("FAIL")]
    assert len(failed) == 0, f"Qualitative validation suite failed on checks:\n{failed}"
