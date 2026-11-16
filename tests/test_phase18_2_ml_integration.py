"""
GeoDemand AI — Root Dispatcher for Phase 18.2 ML Integration Tests
"""

from tests.unit.test_phase18_2_ml_integration import (
    predictor,
    sample_feature_dict,
    test_task1_production_model_loading,
    test_task2_model_schema_across_cities,
    test_task3_multi_hour_predictions,
    test_task4_quantile_ordering,
    test_task5_shap_explanations,
    test_task6_drift_detection,
    test_task7_retraining_workflow,
    test_task8_promotion_and_rollback,
    test_task9_actual_data_evaluation,
)

__all__ = [
    "predictor",
    "sample_feature_dict",
    "test_task1_production_model_loading",
    "test_task2_model_schema_across_cities",
    "test_task3_multi_hour_predictions",
    "test_task4_quantile_ordering",
    "test_task5_shap_explanations",
    "test_task6_drift_detection",
    "test_task7_retraining_workflow",
    "test_task8_promotion_and_rollback",
    "test_task9_actual_data_evaluation",
]
