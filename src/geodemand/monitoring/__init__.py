"""
GeoDemand AI — Monitoring & Model Quality Automation Package

Member 3 — Machine Learning & MLOps

Exports:
- Feature Drift Detection (KS test, PSI, Categorical distribution checks, Schema validation)
- Prediction Quality Checks (Contract enforcement, NaN/Inf checks, Monotonicity)
- Delayed Performance Monitoring (MAE, RMSE, R², Bias, Interval coverage, Slices)
- MLflow Monitoring Logger
"""

from src.geodemand.monitoring.ct_pipeline import (
    ContinuousTrainingPipeline,
    RetrainingResult,
    RetrainingTrigger,
    TrainingDataEligibilityReport,
)
from src.geodemand.monitoring.drift import (
    DriftReport,
    DriftThresholds,
    FeatureDriftDetector,
    FeatureDriftResult,
    ReferenceDistributionProfile,
    SchemaValidationResult,
    calculate_psi,
    calculate_wasserstein_distance,
)
from src.geodemand.monitoring.evidently_adapter import (
    EvidentlyMonitoringAdapter,
    FallbackEvidentlyReport,
    HAS_EVIDENTLY,
)
from src.geodemand.monitoring.mlflow_logger import (
    MLflowMonitoringLogger,
)
from src.geodemand.monitoring.model_health import (
    ModelHealthProvider,
    ModelHealthReport,
)
from src.geodemand.monitoring.operational_monitor import (
    InferenceLatencyMonitor,
    LatencyMetrics,
    ModelLoadingHealth,
    ModelLoadingMonitor,
    ModelRollbackTracker,
    OperationalHealthSummary,
    OperationalMLMonitor,
    QuantileCoverageMonitor,
    QuantileCoverageReport,
    VersionAndRollbackStatus,
)
from src.geodemand.monitoring.performance import (
    ErrorDistribution,
    PerformanceMonitor,
    PerformanceReport,
    RollingPerformanceAssessment,
    SIMULATION_DATA_DISCLAIMER,
    SlicePerformance,
)
from src.geodemand.monitoring.prediction_drift import (
    PredictionDriftDetector,
    PredictionDriftReport,
    PredictionDriftResult,
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
    "calculate_psi",
    "calculate_wasserstein_distance",
    # Prediction Drift
    "PredictionDriftDetector",
    "PredictionDriftReport",
    "PredictionDriftResult",
    # Evidently Adapter
    "EvidentlyMonitoringAdapter",
    "FallbackEvidentlyReport",
    "HAS_EVIDENTLY",
    # Model Health
    "ModelHealthProvider",
    "ModelHealthReport",
    # Prediction Quality
    "PredictionQualityError",
    "PredictionQualityReport",
    "PredictionValidator",
    # Performance
    "PerformanceMonitor",
    "PerformanceReport",
    "SlicePerformance",
    "RollingPerformanceAssessment",
    "ErrorDistribution",
    "SIMULATION_DATA_DISCLAIMER",
    # MLflow
    "MLflowMonitoringLogger",
    # Continuous Training
    "ContinuousTrainingPipeline",
    "RetrainingResult",
    "RetrainingTrigger",
    "TrainingDataEligibilityReport",
    # Operational Monitoring (Phase 17.2)
    "OperationalMLMonitor",
    "OperationalHealthSummary",
    "ModelLoadingMonitor",
    "ModelLoadingHealth",
    "InferenceLatencyMonitor",
    "LatencyMetrics",
    "QuantileCoverageMonitor",
    "QuantileCoverageReport",
    "ModelRollbackTracker",
    "VersionAndRollbackStatus",
]
