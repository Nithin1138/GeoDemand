"""
GeoDemand AI — Unit Tests for Multi-City Model Generalization (Phase 9)

Member 2 — Machine Learning & MLOps

Validates:
1. Multi-city dataset generation and bounding box compliance
2. Evaluation metrics generation and category-wise decomposition
3. Cross-city transfer benchmark across Vijayawada, Hyderabad, Bengaluru, Visakhapatnam
4. Coordinate-to-city resolution and unsupported location rejection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.multicity import MultiCityRegistry
from src.geodemand.models.multicity_eval import (
    MultiCityEvaluator,
    MultiCityGeneralizationReport,
)


class MockModel:
    """Mock regression model for multi-city evaluation tests."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Simple prediction based on office count and hour if available
        preds = np.ones(len(X)) * 12.0
        if "office_count" in X.columns:
            preds += X["office_count"] * 0.5
        return preds


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def sample_feature_df():
    n = 100
    return pd.DataFrame({
        "hour": np.random.randint(0, 24, n),
        "office_count": np.random.randint(1, 15, n),
        "restaurant_count": np.random.randint(2, 25, n),
        "vendor_category": np.random.choice(["food", "chai", "snacks", "fruits"], n),
        "expected_customer_count": np.random.randint(5, 30, n),
    })


@pytest.fixture
def evaluator():
    return MultiCityEvaluator()


def test_city_eval_data_generation(evaluator, sample_feature_df):
    """Test generated city data adheres to city bounding box constraints."""
    hyd_df = evaluator.generate_synthetic_city_eval_data(sample_feature_df, "hyderabad", n_samples=50)
    assert len(hyd_df) == 50
    assert "city_id" in hyd_df.columns
    assert (hyd_df["city_id"] == "hyderabad").all()

    # Coordinates within Hyderabad bounding box: lat ~17.3-17.55, lng ~78.25-78.60
    assert (hyd_df["latitude"] >= 17.30).all() and (hyd_df["latitude"] <= 17.55).all()
    assert (hyd_df["longitude"] >= 78.25).all() and (hyd_df["longitude"] <= 78.60).all()


def test_evaluate_model_on_city(evaluator, mock_model, sample_feature_df):
    """Test city evaluation computes valid MAE, RMSE, and category breakdown."""
    city_df = evaluator.generate_synthetic_city_eval_data(sample_feature_df, "vijayawada", n_samples=80)
    metrics = evaluator.evaluate_model_on_city(
        mock_model,
        city_df,
        feature_cols=["hour", "office_count", "restaurant_count"],
        city_id="vijayawada",
    )

    assert metrics.city_id == "vijayawada"
    assert metrics.sample_size == 80
    assert metrics.mae >= 0.0
    assert metrics.rmse >= metrics.mae
    assert metrics.calibration_factor > 0.0
    assert "food" in metrics.category_mae or len(metrics.category_mae) > 0


def test_cross_city_transfer_benchmark(evaluator, mock_model, sample_feature_df):
    """Test cross-city transfer experiment evaluates all four cities."""
    report = evaluator.run_cross_city_transfer_benchmark(
        mock_model,
        sample_feature_df,
        feature_cols=["hour", "office_count", "restaurant_count"],
        train_city_id="vijayawada",
    )

    assert isinstance(report, MultiCityGeneralizationReport)
    assert set(report.cities_evaluated) == {"vijayawada", "hyderabad", "bengaluru", "visakhapatnam"}
    assert len(report.cross_city_transfer) == 3  # Hyderabad, Bengaluru, Visakhapatnam transferred from Vijayawada
    assert "vijayawada" in report.city_calibration_factors
    assert "hyderabad" in report.city_calibration_factors
    assert "conclusion" in report.to_dict()


def test_resolve_model_for_location(evaluator):
    """Test coordinate resolution to correct city and unsupported region rejection."""
    # Benz Circle, Vijayawada
    v_city, v_calib = evaluator.resolve_model_for_location(16.5062, 80.6480)
    assert v_city == "vijayawada"
    assert v_calib == 1.00

    # HITEC City, Hyderabad
    h_city, h_calib = evaluator.resolve_model_for_location(17.4435, 78.3772)
    assert h_city == "hyderabad"
    assert h_calib > 1.00

    # Koramangala, Bengaluru
    b_city, b_calib = evaluator.resolve_model_for_location(12.9352, 77.6245)
    assert b_city == "bengaluru"

    # Beach Road, Visakhapatnam
    viz_city, viz_calib = evaluator.resolve_model_for_location(17.7231, 83.3245)
    assert viz_city == "visakhapatnam"

    # Unsupported location (e.g. New Delhi: 28.6139, 77.2090)
    with pytest.raises(ValueError, match="outside all supported metropolitan boundaries"):
        evaluator.resolve_model_for_location(28.6139, 77.2090)
