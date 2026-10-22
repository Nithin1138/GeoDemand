"""
GeoDemand AI — ML Monitoring Test Suite (Member 3)

Dispatches to tests/unit/test_ml_monitoring.py
"""

from tests.unit.test_ml_monitoring import (
    detector,
    synthetic_reference_df,
    test_no_drift_identical_distributions,
    test_continuous_drift_detection_ks_and_psi,
    test_categorical_drift_detection,
    test_schema_missing_and_unexpected_features,
    test_schema_invalid_numerical_values,
    test_prediction_quality_nan_and_infinite,
    test_prediction_quality_output_shapes,
    test_prediction_quality_quantile_ordering,
    test_performance_metric_calculations,
    test_performance_monitor_slice_breakdowns,
    test_monitoring_small_datasets,
    test_monitoring_missing_values,
    test_practical_significance_filtering,
    test_multiple_comparison_corrections,
    test_mlflow_monitoring_integration,
)

__all__ = [
    "test_no_drift_identical_distributions",
    "test_continuous_drift_detection_ks_and_psi",
    "test_categorical_drift_detection",
    "test_schema_missing_and_unexpected_features",
    "test_schema_invalid_numerical_values",
    "test_prediction_quality_nan_and_infinite",
    "test_prediction_quality_output_shapes",
    "test_prediction_quality_quantile_ordering",
    "test_performance_metric_calculations",
    "test_performance_monitor_slice_breakdowns",
    "test_monitoring_small_datasets",
    "test_monitoring_missing_values",
    "test_practical_significance_filtering",
    "test_multiple_comparison_corrections",
    "test_mlflow_monitoring_integration",
]
