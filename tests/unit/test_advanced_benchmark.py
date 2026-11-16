r"""
GeoDemand AI — Unit Tests for Advanced ML Model Benchmark (Phase 13.2)

Member 2 — Machine Learning & MLOps

Validates:
1. ApprovedLightGBMBaseline fit, multi-horizon prediction, and non-crossing quantiles
2. RecursiveAutoregressiveLightGBM forward multi-step trajectory propagation
3. TemporalFusionTransformerExperiment (TFT) VSN, GRN, and multi-head attention
4. NHiTSExperiment hierarchical multi-rate pooling and synthesis
5. PatchTSTExperiment subseries patching and channel attention
6. Multi-model benchmark execution across identical splits
7. Champion selection logic and rollback artifact preservation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.geodemand.models.advanced_models import (
    ApprovedLightGBMBaseline,
    NHiTSExperiment,
    PatchTSTExperiment,
    RecursiveAutoregressiveLightGBM,
    TemporalFusionTransformerExperiment,
)
from src.geodemand.models.benchmark_runner import AdvancedModelBenchmarkRunner

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def synthetic_benchmark_data():
    """Generates synthetic multi-horizon dataframe for unit tests."""
    n = 300
    horizons = [1, 2, 4, 8]
    base_t = pd.date_range("2026-01-01", periods=n, freq="h")
    y_raw = np.clip(12.0 + 6.0 * np.sin(np.linspace(0, 15, n)) + np.random.normal(0, 1.2, n), 0.0, 50.0)

    df = pd.DataFrame({
        "decision_timestamp": base_t,
        "hour": [t % 24 for t in range(n)],
        "hour_sin": np.sin(2 * np.pi * np.array([t % 24 for t in range(n)]) / 24.0),
        "hour_cos": np.cos(2 * np.pi * np.array([t % 24 for t in range(n)]) / 24.0),
        "temperature": 27.0 + 4.0 * np.cos(np.linspace(0, 10, n)),
        "population_density": [1500.0] * n,
        "expected_customer_count_lag_1h": np.roll(y_raw, 1),
        "expected_customer_count": y_raw,
    })

    y_val_horizons = {}
    for h in horizons:
        y_val_horizons[h] = pd.Series(np.clip(y_raw * (1.0 + 0.04 * (h - 1)), 0.0, 60.0))

    feature_cols = ["hour", "hour_sin", "hour_cos", "temperature", "population_density", "expected_customer_count_lag_1h"]
    X = df[feature_cols]
    y = df["expected_customer_count"]

    X_train, y_train = X.iloc[:200], y.iloc[:200]
    X_val = X.iloc[200:]
    y_val_h = {h: s.iloc[200:] for h, s in y_val_horizons.items()}

    return X_train, y_train, X_val, y_val_h, horizons


def test_approved_lightgbm_baseline_fit_and_predict(synthetic_benchmark_data):
    """Test ApprovedLightGBMBaseline produces non-negative, non-crossing quantile forecasts."""
    X_train, y_train, X_val, _, horizons = synthetic_benchmark_data
    model = ApprovedLightGBMBaseline(random_state=42)
    model.fit(X_train, y_train, horizons=horizons)

    preds = model.predict_horizons(X_val, horizons=horizons)
    assert set(preds.keys()) == {"p10", "p50", "p90"}

    for h in horizons:
        p10 = preds["p10"][h]
        p50 = preds["p50"][h]
        p90 = preds["p90"][h]

        assert len(p10) == len(X_val)
        assert np.all(p10 >= 0.0)
        assert np.all(p10 <= p50)
        assert np.all(p50 <= p90)


def test_recursive_autoregressive_lightgbm_multi_horizon(synthetic_benchmark_data):
    """Test RecursiveAutoregressiveLightGBM recursive forward trajectory rollout."""
    X_train, y_train, X_val, _, horizons = synthetic_benchmark_data
    model = RecursiveAutoregressiveLightGBM(random_state=42)
    model.fit(X_train, y_train, horizons=horizons)

    preds = model.predict_horizons(X_val, horizons=horizons)
    assert set(preds.keys()) == {"p10", "p50", "p90"}

    for h in horizons:
        p10 = preds["p10"][h]
        p50 = preds["p50"][h]
        p90 = preds["p90"][h]

        assert np.all(p10 >= 0.0)
        assert np.all(p10 <= p50)
        assert np.all(p50 <= p90)


def test_tft_experiment_architecture(synthetic_benchmark_data):
    """Test TemporalFusionTransformerExperiment representations and quantile calibration."""
    X_train, y_train, X_val, _, horizons = synthetic_benchmark_data
    model = TemporalFusionTransformerExperiment(hidden_dim=16, random_state=42)
    model.fit(X_train, y_train, horizons=horizons)

    preds = model.predict_horizons(X_val, horizons=horizons)
    assert set(preds.keys()) == {"p10", "p50", "p90"}

    for h in horizons:
        p10 = preds["p10"][h]
        p50 = preds["p50"][h]
        p90 = preds["p90"][h]

        assert len(p50) == len(X_val)
        assert np.all(p10 >= 0.0)
        assert np.all(p10 <= p50)
        assert np.all(p50 <= p90)


def test_nhits_experiment_architecture(synthetic_benchmark_data):
    """Test NHiTSExperiment multi-rate hierarchical pooling and synthesis."""
    X_train, y_train, X_val, _, horizons = synthetic_benchmark_data
    model = NHiTSExperiment(hidden_dim=16, pooling_rates=(1, 2), random_state=42)
    model.fit(X_train, y_train, horizons=horizons)

    preds = model.predict_horizons(X_val, horizons=horizons)
    assert set(preds.keys()) == {"p10", "p50", "p90"}

    for h in horizons:
        p10 = preds["p10"][h]
        p50 = preds["p50"][h]
        p90 = preds["p90"][h]

        assert np.all(p10 >= 0.0)
        assert np.all(p10 <= p50)
        assert np.all(p50 <= p90)


def test_patchtst_experiment_architecture(synthetic_benchmark_data):
    """Test PatchTSTExperiment subseries patching and channel attention."""
    X_train, y_train, X_val, _, horizons = synthetic_benchmark_data
    model = PatchTSTExperiment(patch_size=2, hidden_dim=16, random_state=42)
    model.fit(X_train, y_train, horizons=horizons)

    preds = model.predict_horizons(X_val, horizons=horizons)
    assert set(preds.keys()) == {"p10", "p50", "p90"}

    for h in horizons:
        p10 = preds["p10"][h]
        p50 = preds["p50"][h]
        p90 = preds["p90"][h]

        assert np.all(p10 >= 0.0)
        assert np.all(p10 <= p50)
        assert np.all(p50 <= p90)


def test_advanced_model_benchmark_runner_execution(synthetic_benchmark_data):
    """Test full benchmark runner compares all 5 models and produces valid metrics."""
    X_train, y_train, X_val, y_val_h, horizons = synthetic_benchmark_data
    runner = AdvancedModelBenchmarkRunner(horizons=horizons, latency_budget_ms=3.0)

    report = runner.run_benchmark(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val_horizons=y_val_h,
        dataset_version="test-dataset-v1",
    )

    assert len(report.models_evaluated) == 5
    assert "Approved_LightGBM_v2.1.0" in report.model_summaries
    assert "Recursive_Autoregressive_LightGBM" in report.model_summaries
    assert "Temporal_Fusion_Transformer_TFT" in report.model_summaries
    assert "N_HiTS_Hierarchical_Synthesis" in report.model_summaries
    assert "PatchTST_Patch_Transformer" in report.model_summaries

    for name, summary in report.model_summaries.items():
        assert summary.h1_mae > 0.0
        assert summary.h8_mae > 0.0
        assert summary.mean_latency_ms >= 0.0
        assert summary.peak_memory_mb > 0.0
        for h in horizons:
            res = summary.results_by_horizon[h]
            assert res.mae > 0.0
            assert res.rmse > 0.0
            assert 0.0 <= res.quantile_coverage_80 <= 1.0

    assert report.champion_retained
    assert report.selected_production_champion == "Approved_LightGBM_v2.1.0"
    assert report.selected_shift_planner == "Recursive_Autoregressive_LightGBM"


def test_champion_rollback_preservation():
    """Test that champion artifact backup exists for immediate rollback."""
    backup_path = ROOT_DIR / "models" / "demand_model_champion_v2.1.0.pkl"
    assert backup_path.exists(), "Rollback artifact backup must exist."


def test_benchmark_report_artifacts_generation():
    """Test that JSON, comparison markdown, and champion selection markdown were generated."""
    json_path = ROOT_DIR / "models" / "multi_horizon_benchmark.json"
    comp_md = ROOT_DIR / "docs" / "MULTI_HORIZON_BENCHMARK_REPORT.md"
    champ_md = ROOT_DIR / "docs" / "CHAMPION_SELECTION_REPORT.md"

    assert json_path.exists()
    assert comp_md.exists()
    assert champ_md.exists()

    with open(comp_md, encoding="utf-8") as f:
        comp_text = f.read()
    assert "Multi-Horizon Performance Comparison Table" in comp_text
    assert "Approved LightGBM Baseline (v2.1.0)" in comp_text

    with open(champ_md, encoding="utf-8") as f:
        champ_text = f.read()
    assert "Production Champion Selection Decision" in champ_text
    assert "RETAINED" in champ_text
