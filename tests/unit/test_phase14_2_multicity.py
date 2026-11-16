"""
GeoDemand AI — Phase 14.2 Multi-City Model Generalization Unit Tests

Tests all 12 tasks of Phase 14.2:
1. Evaluate Vijayawada model on Hyderabad data
2. Evaluate Vijayawada model on Bengaluru data
3. Evaluate Vijayawada model on Visakhapatnam data
4. Compare city-specific localized models
5. Compare pooled multi-city models against localized baselines
6. Perform train-one-city / test-another-city experiments (4x4 transfer matrix)
7. Evaluate category-wise demand performance across cities
8. Identify non-generalizable vs generalizable spatial/temporal features
9. Evaluate city-specific calibration factors and verify error reduction
10. Define and test model selection by city (CityModelSelector)
11. Prevent incorrect model-to-city mapping (CityBoundaryError on out-of-bounds coords)
12. Verify generalization limitations documentation
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.multicity import MultiCityRegistry
from src.geodemand.models.multicity_eval import (
    CityBoundaryError,
    CityEvaluationMetrics,
    CityModelSelector,
    CitySelectionResult,
    CrossCityTransferResult,
    MultiCityEvaluator,
    MultiCityGeneralizationReport,
    NonGeneralizableFeaturesReport,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Fixture providing a deterministic sample feature dataset with spatial and category features."""
    rng = np.random.RandomState(42)
    n = 300
    office = rng.randint(0, 30, n)
    restaurant = rng.randint(0, 40, n)
    target = 5 + (office * 0.4) + (restaurant * 0.3) + rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "latitude": rng.uniform(16.48, 16.55, n),
        "longitude": rng.uniform(80.60, 80.68, n),
        "office_count": office,
        "restaurant_count": restaurant,
        "population_density": rng.uniform(5000, 15000, n),
        "road_density": rng.uniform(10, 50, n),
        "commercial_ratio": rng.uniform(0.1, 0.8, n),
        "temperature": rng.uniform(25.0, 38.0, n),
        "rainfall": rng.exponential(1.0, n),
        "sin_hour": np.sin(2 * np.pi * rng.randint(0, 24, n) / 24.0),
        "cos_hour": np.cos(2 * np.pi * rng.randint(0, 24, n) / 24.0),
        "is_weekend": rng.choice([0, 1], n),
        "vendor_category": rng.choice(["food", "tea", "fruits", "apparel"], n),
        "expected_customer_count": np.clip(np.round(target), 1, 60).astype(int),
    })


@pytest.fixture
def feature_cols() -> list:
    return [
        "office_count",
        "restaurant_count",
        "population_density",
        "road_density",
        "commercial_ratio",
        "temperature",
        "rainfall",
        "sin_hour",
        "cos_hour",
        "is_weekend",
        "vendor_category",
    ]


@pytest.fixture
def evaluator() -> MultiCityEvaluator:
    return MultiCityEvaluator()


# ==============================================================================
# Task 1: Evaluate Vijayawada model on Hyderabad data
# ==============================================================================
def test_task1_evaluate_vijayawada_model_on_hyderabad(evaluator, sample_feature_data, feature_cols):
    v_data = evaluator.generate_synthetic_city_eval_data(sample_feature_data, "vijayawada", n_samples=200, random_seed=42)
    h_data = evaluator.generate_synthetic_city_eval_data(sample_feature_data, "hyderabad", n_samples=200, random_seed=42)

    v_model = evaluator.train_city_model(v_data, feature_cols)
    h_eval = evaluator.evaluate_model_on_city(v_model, h_data, feature_cols, "hyderabad")

    assert h_eval.city_id == "hyderabad"
    assert h_eval.sample_size == 200
    assert h_eval.mae > 0
    assert h_eval.rmse >= h_eval.mae
    assert h_eval.calibration_factor > 1.0  # Hyderabad scale is higher (~1.25x)


# ==============================================================================
# Task 2: Evaluate Vijayawada model on Bengaluru data
# ==============================================================================
def test_task2_evaluate_vijayawada_model_on_bengaluru(evaluator, sample_feature_data, feature_cols):
    v_data = evaluator.generate_synthetic_city_eval_data(sample_feature_data, "vijayawada", n_samples=200, random_seed=42)
    b_data = evaluator.generate_synthetic_city_eval_data(sample_feature_data, "bengaluru", n_samples=200, random_seed=42)

    v_model = evaluator.train_city_model(v_data, feature_cols)
    b_eval = evaluator.evaluate_model_on_city(v_model, b_data, feature_cols, "bengaluru")

    assert b_eval.city_id == "bengaluru"
    assert b_eval.sample_size == 200
    assert b_eval.mae > 0
    assert b_eval.calibration_factor > 1.1  # Bengaluru scale is highest (~1.35x)


# ==============================================================================
# Task 3: Evaluate Vijayawada model on Visakhapatnam data
# ==============================================================================
def test_task3_evaluate_vijayawada_model_on_visakhapatnam(evaluator, sample_feature_data, feature_cols):
    v_data = evaluator.generate_synthetic_city_eval_data(sample_feature_data, "vijayawada", n_samples=200, random_seed=42)
    viz_data = evaluator.generate_synthetic_city_eval_data(sample_feature_data, "visakhapatnam", n_samples=200, random_seed=42)

    v_model = evaluator.train_city_model(v_data, feature_cols)
    viz_eval = evaluator.evaluate_model_on_city(v_model, viz_data, feature_cols, "visakhapatnam")

    assert viz_eval.city_id == "visakhapatnam"
    assert viz_eval.sample_size == 200
    assert viz_eval.mae > 0
    # Both are Tier-2 AP cities; calibration factor should be close to 1.05
    assert 0.95 <= viz_eval.calibration_factor <= 1.20


# ==============================================================================
# Task 4: Compare city-specific models
# ==============================================================================
def test_task4_compare_city_specific_models(evaluator, sample_feature_data, feature_cols):
    cities = ["vijayawada", "hyderabad", "bengaluru", "visakhapatnam"]
    city_models = {}
    city_data = {}

    for c in cities:
        city_data[c] = evaluator.generate_synthetic_city_eval_data(sample_feature_data, c, n_samples=150, random_seed=42)
        city_models[c] = evaluator.train_city_model(city_data[c], feature_cols)

    in_domain_maes = {}
    for c in cities:
        metrics = evaluator.evaluate_model_on_city(city_models[c], city_data[c], feature_cols, c)
        in_domain_maes[c] = metrics.mae
        assert metrics.mae > 0

    assert len(in_domain_maes) == 4
    # All in-domain models should achieve reasonable MAE (< 5.0)
    for c, mae in in_domain_maes.items():
        assert mae < 5.0, f"In-domain model for {c} had high MAE: {mae}"


# ==============================================================================
# Task 5: Compare pooled multi-city models
# ==============================================================================
def test_task5_compare_pooled_multi_city_models(evaluator, sample_feature_data, feature_cols):
    cities = ["vijayawada", "hyderabad", "bengaluru", "visakhapatnam"]
    datasets = [evaluator.generate_synthetic_city_eval_data(sample_feature_data, c, n_samples=150, random_seed=42) for c in cities]
    pooled_df = pd.concat(datasets, ignore_index=True)

    pooled_model = evaluator.train_city_model(pooled_df, feature_cols)

    pooled_maes = []
    for i, c in enumerate(cities):
        p_eval = evaluator.evaluate_model_on_city(pooled_model, datasets[i], feature_cols, c)
        pooled_maes.append(p_eval.mae)

    avg_pooled_mae = float(np.mean(pooled_maes))
    assert avg_pooled_mae > 0
    assert avg_pooled_mae < 6.0


# ==============================================================================
# Task 6: Perform train-one-city / test-another-city experiments (4x4 matrix)
# ==============================================================================
def test_task6_train_one_test_another_transfer_matrix(evaluator, sample_feature_data, feature_cols):
    report = evaluator.run_full_generalization_benchmark(sample_feature_data, feature_cols)

    assert isinstance(report, MultiCityGeneralizationReport)
    assert len(report.cities_evaluated) == 4
    matrix = report.cross_city_transfer_matrix

    # Check 4x4 matrix existence
    for train_c in report.cities_evaluated:
        assert train_c in matrix
        for test_c in report.cities_evaluated:
            assert test_c in matrix[train_c]
            assert matrix[train_c][test_c] > 0

    # In-domain is generally better or comparable to cross-city transfer without calibration
    v_to_v = matrix["vijayawada"]["vijayawada"]
    v_to_b = matrix["vijayawada"]["bengaluru"]
    assert v_to_b > v_to_v, "Uncalibrated zero-shot transfer to Bengaluru should have higher MAE than in-domain Vijayawada"


# ==============================================================================
# Task 7: Evaluate category-wise performance across cities
# ==============================================================================
def test_task7_evaluate_category_wise_performance(evaluator, sample_feature_data, feature_cols):
    report = evaluator.run_full_generalization_benchmark(sample_feature_data, feature_cols)
    cat_perf = report.category_performance_by_city

    assert len(cat_perf) == 4
    for city, cats in cat_perf.items():
        assert len(cats) > 0, f"City {city} should have category breakdowns"
        for cat_name, cat_mae in cats.items():
            assert cat_mae >= 0


# ==============================================================================
# Task 8: Identify non-generalizable features
# ==============================================================================
def test_task8_identify_non_generalizable_features(evaluator, feature_cols):
    report = evaluator.identify_non_generalizable_features(feature_cols)

    assert isinstance(report, NonGeneralizableFeaturesReport)
    non_gen_names = [f["feature"] for f in report.non_generalizable_features]
    gen_names = [f["feature"] for f in report.generalizable_features]

    assert "latitude" in non_gen_names
    assert "longitude" in non_gen_names
    assert "h3_cell_id" in non_gen_names
    assert any("hour" in name for name in gen_names)
    assert any("temperature" in name for name in gen_names)
    assert any("rainfall" in name for name in gen_names)


# ==============================================================================
# Task 9: Evaluate city-specific calibration
# ==============================================================================
def test_task9_evaluate_city_specific_calibration(evaluator, sample_feature_data, feature_cols):
    report = evaluator.run_full_generalization_benchmark(sample_feature_data, feature_cols)

    # Check transfer results with calibration
    assert len(report.cross_city_transfer_results) == 3  # Vijayawada -> Hyderabad, Bengaluru, Visakhapatnam

    for res in report.cross_city_transfer_results:
        assert isinstance(res, CrossCityTransferResult)
        assert res.calibrated_transfer_mae <= res.transfer_mae, (
            f"Calibration should reduce or preserve MAE: {res.calibrated_transfer_mae} vs {res.transfer_mae}"
        )
        assert res.calibration_recovery_pct >= 0.0


# ==============================================================================
# Task 10: Define model selection by city (CityModelSelector)
# ==============================================================================
def test_task10_model_selection_by_city():
    selector = CityModelSelector(strict_bounds=True)

    # Vijayawada Benz Circle
    res_v = selector.resolve_model_for_coordinate(16.5062, 80.6480)
    assert isinstance(res_v, CitySelectionResult)
    assert res_v.city_id == "vijayawada"
    assert res_v.state_code == "AP"
    assert res_v.calibration_factor == 1.00
    assert res_v.within_registered_boundary is True

    # Hyderabad HITEC City
    res_h = selector.resolve_model_for_coordinate(17.4435, 78.3772)
    assert res_h.city_id == "hyderabad"
    assert res_h.state_code == "TS"
    assert res_h.calibration_factor == 1.25

    # Bengaluru Indiranagar
    res_b = selector.resolve_model_for_coordinate(12.9784, 77.6408)
    assert res_b.city_id == "bengaluru"
    assert res_b.state_code == "KA"
    assert res_b.calibration_factor == 1.35

    # Visakhapatnam Beach Road
    res_viz = selector.resolve_model_for_coordinate(17.7126, 83.3197)
    assert res_viz.city_id == "visakhapatnam"
    assert res_viz.state_code == "AP"
    assert res_viz.calibration_factor == 1.05


# ==============================================================================
# Task 11: Prevent incorrect model-to-city mapping
# ==============================================================================
def test_task11_prevent_incorrect_model_to_city_mapping():
    selector = CityModelSelector(strict_bounds=True)

    # Coordinates in Mumbai (Maharashtra) outside AP/TS/KA boundaries
    with pytest.raises(CityBoundaryError) as exc_info:
        selector.resolve_model_for_coordinate(18.9438, 72.8232)
    assert "fall outside all supported metropolitan boundaries" in str(exc_info.value)

    # Coordinates in New Delhi
    with pytest.raises(CityBoundaryError):
        selector.resolve_model_for_coordinate(28.6139, 77.2090)

    # Non-strict mode should provide fallback
    selector_lenient = CityModelSelector(strict_bounds=False, default_city_id="vijayawada")
    fallback_res = selector_lenient.resolve_model_for_coordinate(18.9438, 72.8232)
    assert fallback_res.city_id == "vijayawada"
    assert fallback_res.within_registered_boundary is False
    assert fallback_res.model_type == "lightgbm_default_fallback"


# ==============================================================================
# Task 12: Document generalization limitations
# ==============================================================================
def test_task12_document_generalization_limitations(evaluator, sample_feature_data, feature_cols):
    report = evaluator.run_full_generalization_benchmark(sample_feature_data, feature_cols)

    assert len(report.generalization_limitations) >= 4
    limitations_text = " ".join(report.generalization_limitations).lower()
    assert "scale disparity" in limitations_text
    assert "microclimate" in limitations_text
    assert "diurnal" in limitations_text
    assert "simulated data" in limitations_text

    # Verify deliverable documents exist
    assert (ROOT_DIR / "docs" / "CROSS_CITY_MODEL_EVALUATION.md").exists()
    assert (ROOT_DIR / "docs" / "CITY_MODEL_STRATEGY.md").exists()
    assert (ROOT_DIR / "models" / "multi_city_benchmark.json").exists()
