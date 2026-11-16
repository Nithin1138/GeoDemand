"""
GeoDemand AI — Models Module (Member 3)

Exports inference, promotion gating, and on-demand lazy training utilities.
"""

from src.geodemand.models.advanced_models import (
    AdvancedForecastingModel,
    ApprovedLightGBMBaseline,
    NHiTSExperiment,
    PatchTSTExperiment,
    RecursiveAutoregressiveLightGBM,
    TemporalFusionTransformerExperiment,
)
from src.geodemand.models.benchmark_runner import (
    AdvancedBenchmarkReport,
    AdvancedModelBenchmarkRunner,
    HorizonResult,
    ModelBenchmarkSummary,
)
from src.geodemand.models.inference import DemandModelPredictor
from src.geodemand.models.multi_horizon import (
    AdvancedForecastingBenchmark,
    HorizonMetrics,
    LeakageAuditReport,
    MultiHorizonBenchmarkReport,
    MultiHorizonFeatureEngine,
    MultiHorizonSplits,
    RecursiveAutoregressiveForecaster,
    SlidingWindowData,
    TimeAlignmentReport,
)
from src.geodemand.models.multicity_eval import (
    CityEvaluationMetrics,
    CrossCityTransferResult,
    MultiCityEvaluator,
    MultiCityGeneralizationReport,
)
from src.geodemand.models.promotion import (
    GateCheckResult,
    ModelPromotionGate,
    ModelPromotionPolicy,
    PromotionEvaluationReport,
)

__all__ = [
    "DemandModelPredictor",
    "ModelPromotionGate",
    "ModelPromotionPolicy",
    "GateCheckResult",
    "PromotionEvaluationReport",
    "MultiHorizonFeatureEngine",
    "RecursiveAutoregressiveForecaster",
    "AdvancedForecastingBenchmark",
    "MultiHorizonBenchmarkReport",
    "HorizonMetrics",
    "TimeAlignmentReport",
    "LeakageAuditReport",
    "SlidingWindowData",
    "MultiHorizonSplits",
    "ApprovedLightGBMBaseline",
    "RecursiveAutoregressiveLightGBM",
    "TemporalFusionTransformerExperiment",
    "NHiTSExperiment",
    "PatchTSTExperiment",
    "AdvancedModelBenchmarkRunner",
    "AdvancedBenchmarkReport",
    "ModelBenchmarkSummary",
    "HorizonResult",
    "MultiCityEvaluator",
    "MultiCityGeneralizationReport",
    "CityEvaluationMetrics",
    "CrossCityTransferResult",
    "train_pipeline",
    "load_and_validate",
    "prepare_features",
    "chronological_split",
    "evaluate",
]


def __getattr__(name: str):
    if name in {"train_pipeline", "load_and_validate", "prepare_features", "chronological_split", "evaluate"}:
        import src.geodemand.models.train_demand_model as tdm
        return getattr(tdm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
