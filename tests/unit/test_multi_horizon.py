r"""
GeoDemand AI — Unit Tests for Multi-Horizon & Autoregressive Forecasting (Phase 8 & Phase 13.1)

Member 2 — Machine Learning & MLOps

Validates:
1. Anti-circularity and zero temporal leakage in lag features (t-1h, t-2h, t-24h)
2. Shifted rolling window statistics (3h, 6h, 24h mean, std, min, max) excluding decision-time target
3. Historical demand baselines with split-protection and expanding windows
4. Sliding-window sequence inputs generation (past, future, static tensors) for deep models
5. Automated time alignment audit (monotonicity, frequency, and target offset verification)
6. Automated zero-leakage verification (prohibited column checks and future permutation invariance)
7. Reproducible forward-chaining multi-horizon chronological splits with horizon embargo
8. Full end-to-end multi-horizon dataset builder
9. Recursive Autoregressive LightGBM multi-step forward prediction ($H \in \{1, 2, 4, 8\}$)
10. Comparative benchmark execution across horizons
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.geodemand.models.multi_horizon import (
    DEFAULT_HORIZONS,
    AdvancedForecastingBenchmark,
    LeakageAuditReport,
    MultiHorizonFeatureEngine,
    MultiHorizonSplits,
    RecursiveAutoregressiveForecaster,
    SlidingWindowData,
    TimeAlignmentReport,
)


@pytest.fixture
def sample_timeseries_df():
    """Generates synthetic hourly timeseries DataFrame with group entity keys."""
    n = 200
    df = pd.DataFrame({
        "decision_timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "h3_cell_id": ["cell_a"] * 100 + ["cell_b"] * 100,
        "vendor_id": ["v_01"] * 100 + ["v_02"] * 100,
        "vendor_category": ["food"] * 200,
        "hour": [t % 24 for t in range(n)],
        "weekday": [((t // 24) % 7) for t in range(n)],
        "day_of_week": [((t // 24) % 7) for t in range(n)],
        "hour_sin": np.sin(2 * np.pi * np.array([t % 24 for t in range(n)]) / 24.0),
        "hour_cos": np.cos(2 * np.pi * np.array([t % 24 for t in range(n)]) / 24.0),
        "temperature": 25.0 + 5.0 * np.sin(np.linspace(0, 10, n)),
        "population_density": [1200.0] * 200,
        "commercial_ratio": [0.45] * 200,
        "expected_customer_count": np.clip(10 + 5 * np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 1, n), 0, 50),
    })
    return df


def test_lag_features_zero_leakage(sample_timeseries_df):
    """Test lag features only use strictly past observations (no leakage) with entity grouping."""
    engine = MultiHorizonFeatureEngine()
    enriched = engine.create_lag_features(
        sample_timeseries_df,
        target_col="expected_customer_count",
        group_cols=["h3_cell_id", "vendor_id"],
        lags=(1, 2, 24),
    )

    assert "expected_customer_count_lag_1h" in enriched.columns
    assert "expected_customer_count_lag_2h" in enriched.columns
    assert "expected_customer_count_lag_24h" in enriched.columns

    # Verify lag 1 is strictly equal to previous row's target within cell_a
    for idx in range(1, 20):
        prev_actual = sample_timeseries_df.loc[idx - 1, "expected_customer_count"]
        lag1_val = enriched.loc[idx, "expected_customer_count_lag_1h"]
        assert np.isclose(prev_actual, lag1_val)

    # Verify entity boundary isolation: first row of cell_b does NOT leak last row of cell_a
    cell_b_start_idx = 100
    assert enriched.loc[cell_b_start_idx, "expected_customer_count_lag_1h"] != sample_timeseries_df.loc[99, "expected_customer_count"]


def test_rolling_window_features_zero_leakage(sample_timeseries_df):
    """Test rolling window statistics strictly exclude decision-time target."""
    engine = MultiHorizonFeatureEngine()
    enriched = engine.create_rolling_window_features(
        sample_timeseries_df,
        target_col="expected_customer_count",
        group_cols=["h3_cell_id", "vendor_id"],
        windows=(3, 6),
    )

    assert "expected_customer_count_rolling_mean_3h" in enriched.columns
    assert "expected_customer_count_rolling_std_3h" in enriched.columns
    assert "expected_customer_count_rolling_min_3h" in enriched.columns
    assert "expected_customer_count_rolling_max_3h" in enriched.columns

    # Verify rolling mean for row 3 uses rows 0, 1, 2 (shifted by 1, excluding row 3)
    expected_mean_row3 = sample_timeseries_df.loc[0:2, "expected_customer_count"].mean()
    actual_mean_row3 = enriched.loc[3, "expected_customer_count_rolling_mean_3h"]
    assert np.isclose(expected_mean_row3, actual_mean_row3)


def test_historical_demand_baselines_with_split_protection(sample_timeseries_df):
    """Test historical demand baselines computed on training partition do not leak test demand."""
    engine = MultiHorizonFeatureEngine()

    train_cutoff = "2026-01-04"
    enriched = engine.create_historical_demand_features(
        sample_timeseries_df,
        target_col="expected_customer_count",
        group_cols=["h3_cell_id", "vendor_id"],
        train_end=train_cutoff,
        timestamp_col="decision_timestamp",
        cell_col="h3_cell_id",
        category_col="vendor_category",
    )

    assert "expected_customer_count_hist_cell_mean" in enriched.columns
    assert "expected_customer_count_hist_category_mean" in enriched.columns
    assert "expected_customer_count_hist_cell_dow_hour_mean" in enriched.columns

    # Verify that cell_mean is constant across evaluation split and matches training slice mean
    train_mask = pd.to_datetime(sample_timeseries_df["decision_timestamp"]) < pd.to_datetime(train_cutoff)
    cell_a_train_mean = sample_timeseries_df[train_mask & (sample_timeseries_df["h3_cell_id"] == "cell_a")]["expected_customer_count"].mean()
    cell_a_enriched = enriched[enriched["h3_cell_id"] == "cell_a"]
    assert np.allclose(cell_a_enriched["expected_customer_count_hist_cell_mean"], cell_a_train_mean)


def test_sliding_window_sequences_generation(sample_timeseries_df):
    """Test generation of 3D sliding-window tensor sequences for deep learning models."""
    engine = MultiHorizonFeatureEngine()
    sw_data = engine.prepare_sliding_window_sequences(
        sample_timeseries_df,
        lookback_window=12,
        forecast_horizons=(1, 2, 4),
        past_feature_cols=["expected_customer_count", "temperature"],
        future_feature_cols=["hour_sin", "hour_cos"],
        static_feature_cols=["population_density", "commercial_ratio"],
        group_cols=["h3_cell_id", "vendor_id"],
        stride=1,
    )

    assert isinstance(sw_data, SlidingWindowData)
    assert sw_data.past_sequences.ndim == 3
    assert sw_data.past_sequences.shape[1] == 12  # lookback
    assert sw_data.past_sequences.shape[2] == 2   # 2 past features
    assert sw_data.future_sequences.shape[1] == 4  # max horizon
    assert sw_data.future_sequences.shape[2] == 2  # 2 future features
    assert sw_data.targets.shape[1] == 3          # 3 horizons (1, 2, 4)
    assert len(sw_data.decision_timestamps) == sw_data.targets.shape[0]


def test_time_alignment_verification_valid_and_invalid(sample_timeseries_df):
    """Test automated audit of temporal monotonicity, continuity, and target offset alignment."""
    engine = MultiHorizonFeatureEngine()
    mh_df = engine.build_multi_horizon_dataset(
        sample_timeseries_df,
        horizons=(1, 2, 4),
        group_cols=["h3_cell_id", "vendor_id"],
    )

    # Valid case
    valid_report = engine.verify_time_alignment(
        mh_df,
        timestamp_col="decision_timestamp",
        group_cols=["h3_cell_id", "vendor_id"],
        horizons=(1, 2, 4),
    )
    assert isinstance(valid_report, TimeAlignmentReport)
    assert valid_report.is_valid
    assert valid_report.monotonic_ascending
    assert valid_report.duplicate_timestamps == 0
    assert valid_report.target_offsets_valid
    assert len(valid_report.violations) == 0

    # Invalid case: introduce inverted timestamps and duplicate
    corrupted = mh_df.copy()
    corrupted.loc[5, "decision_timestamp"] = corrupted.loc[2, "decision_timestamp"]
    invalid_report = engine.verify_time_alignment(
        corrupted,
        timestamp_col="decision_timestamp",
        group_cols=["h3_cell_id", "vendor_id"],
        horizons=(1, 2, 4),
    )
    assert not invalid_report.is_valid
    assert len(invalid_report.violations) > 0


def test_zero_future_leakage_audit(sample_timeseries_df):
    """Test automated detection of prohibited latent variables and target leakage."""
    engine = MultiHorizonFeatureEngine()
    clean_cols = ["hour", "hour_sin", "hour_cos", "temperature", "population_density", "expected_customer_count_lag_1h"]

    # 1. Clean feature set
    report_clean = engine.verify_zero_future_leakage(
        sample_timeseries_df,
        feature_cols=clean_cols,
        horizons=(1, 2, 4),
    )
    assert isinstance(report_clean, LeakageAuditReport)
    assert not report_clean.has_leakage
    assert len(report_clean.blocked_features_found) == 0
    assert report_clean.future_permutation_invariance

    # 2. Corrupted with prohibited column
    leak_cols = clean_cols + ["hidden_skill_factor"]
    report_leak = engine.verify_zero_future_leakage(
        sample_timeseries_df,
        feature_cols=leak_cols,
        horizons=(1, 2, 4),
    )
    assert report_leak.has_leakage
    assert "hidden_skill_factor" in report_leak.blocked_features_found

    # 3. Direct target in feature list
    direct_target_cols = clean_cols + ["expected_customer_count"]
    report_target_leak = engine.verify_zero_future_leakage(
        sample_timeseries_df,
        feature_cols=direct_target_cols,
        horizons=(1, 2, 4),
    )
    assert report_target_leak.has_leakage


def test_reproducible_multi_horizon_splits_with_embargo(sample_timeseries_df):
    """Test forward-chaining splits with horizon embargo purging between partitions."""
    engine = MultiHorizonFeatureEngine()
    mh_df = engine.build_multi_horizon_dataset(
        sample_timeseries_df,
        horizons=(1, 2, 4, 8),
        group_cols=["h3_cell_id", "vendor_id"],
    )

    train_end = "2026-01-04"
    val_end = "2026-01-07"
    embargo_hours = 8

    splits = engine.create_multi_horizon_splits(
        mh_df,
        train_end=train_end,
        val_end=val_end,
        embargo_hours=embargo_hours,
        horizons=(1, 2, 4, 8),
    )

    assert isinstance(splits, MultiHorizonSplits)
    assert len(splits.train_df) > 0
    assert len(splits.val_df) > 0
    assert len(splits.test_df) > 0

    # Ensure train max timestamp is strictly before train_end - embargo_hours
    max_train_ts = pd.to_datetime(splits.train_df["decision_timestamp"].max())
    assert max_train_ts < pd.to_datetime(train_end) - pd.Timedelta(hours=embargo_hours)

    # Ensure metadata is fully populated and reproducible
    assert splits.summary_metadata["embargo_hours"] == embargo_hours
    assert "sha256" in splits.summary_metadata


def test_end_to_end_build_multi_horizon_dataset(sample_timeseries_df):
    """Test complete dataset builder with forward targets, lags, rolling stats, and baselines."""
    engine = MultiHorizonFeatureEngine()
    horizons = (1, 2, 4, 8)
    mh_df = engine.build_multi_horizon_dataset(
        sample_timeseries_df,
        horizons=horizons,
        group_cols=["h3_cell_id", "vendor_id"],
        lags=(1, 2, 24),
        rolling_windows=(3, 6, 24),
        include_historical_baselines=True,
        drop_tail_na=True,
    )

    # Forward targets verified
    for h in horizons:
        assert f"expected_customer_count_h{h}h" in mh_df.columns
        assert f"target_timestamp_h{h}h" in mh_df.columns

    # Lags verified
    for l in (1, 2, 24):
        assert f"expected_customer_count_lag_{l}h" in mh_df.columns

    # Rolling stats verified
    for w in (3, 6, 24):
        assert f"expected_customer_count_rolling_mean_{w}h" in mh_df.columns
        assert f"expected_customer_count_rolling_std_{w}h" in mh_df.columns

    # No NaN in final forward targets due to drop_tail_na
    assert not mh_df[f"expected_customer_count_h8h"].isna().any()


def test_recursive_forecaster_multi_horizon(sample_timeseries_df):
    """Test recursive autoregressive forecasting produces valid positive multi-horizon predictions."""
    engine = MultiHorizonFeatureEngine()
    enriched = engine.generate_lag_features(sample_timeseries_df, lags=(1, 2))

    feature_cols = ["hour", "hour_sin", "hour_cos", "temperature", "expected_customer_count_lag_1h"]
    X = enriched[feature_cols]
    y = enriched["expected_customer_count"]

    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_val = X.iloc[150:180]

    forecaster = RecursiveAutoregressiveForecaster()
    forecaster.fit(X_train, y_train, feature_cols=feature_cols)

    horizons = [1, 2, 4, 8]
    preds = forecaster.predict_multi_horizon(X_val, horizons=horizons)

    assert set(preds.keys()) == set(horizons)
    for h, arr in preds.items():
        assert len(arr) == len(X_val)
        assert np.all(arr >= 0.0)  # Nonnegative customer count


def test_advanced_forecasting_benchmark(sample_timeseries_df):
    """Test multi-horizon benchmark report generation and champion retention."""
    engine = MultiHorizonFeatureEngine()
    enriched = engine.generate_lag_features(sample_timeseries_df, lags=(1, 2))

    feature_cols = ["hour", "hour_sin", "hour_cos", "temperature", "expected_customer_count_lag_1h"]
    X_train = enriched[feature_cols].iloc[:140]
    y_train = enriched["expected_customer_count"].iloc[:140]
    X_val = enriched[feature_cols].iloc[140:170]
    y_val = enriched["expected_customer_count"].iloc[140:170]

    benchmark = AdvancedForecastingBenchmark()
    report = benchmark.run_benchmark(X_train, y_train, X_val, y_val, horizons=[1, 2, 4])

    assert "Recursive_Autoregressive_LightGBM" in report.results_by_model
    assert "Deep_Temporal_Sequence_Benchmark" in report.results_by_model
    assert report.champion_retained
    assert "RETAIN_PRODUCTION_CHAMPION" in report.recommendation
