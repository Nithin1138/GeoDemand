"""
GeoDemand AI — Monitoring & Model Quality Automation Package

Member 3 — Machine Learning & MLOps

Exports:
- Feature Drift Detection (KS test, PSI, Categorical distribution checks, Schema validation)
- Prediction Quality Checks (Contract enforcement, NaN/Inf checks, Monotonicity)
- Delayed Performance Monitoring (MAE, RMSE, R², Bias, Interval coverage, Slices)
- MLflow Monitoring Logger
"""

from src.geodemand.monitoring.drift import (
    DriftReport,
    DriftThresholds,
    FeatureDriftDetector,
    FeatureDriftResult,
    ReferenceDistributionProfile,
    SchemaValidationResult,
)
from src.geodemand.monitoring.mlflow_logger import (
    MLflowMonitoringLogger,
)
from src.geodemand.monitoring.performance import (
    ErrorDistribution,
    PerformanceMonitor,
    PerformanceReport,
    SIMULATION_DATA_DISCLAIMER,
    SlicePerformance,
)
from src.geodemand.monitoring.prediction_quality import (
    PredictionQualityError,
    PredictionQualityReport,
    PredictionValidator,
)

__all__ = [
    # Drift
    "DriftReport",
    "DriftThresholds",
    "FeatureDriftDetector",
    "FeatureDriftResult",
    "ReferenceDistributionProfile",
    "SchemaValidationResult",
    # Prediction Quality
    "PredictionQualityError",
    "PredictionQualityReport",
    "PredictionValidator",
    # Performance
    "PerformanceMonitor",
    "PerformanceReport",
    "SlicePerformance",
    "ErrorDistribution",
    "SIMULATION_DATA_DISCLAIMER",
    # MLflow
    "MLflowMonitoringLogger",
]
