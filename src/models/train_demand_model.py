"""
GeoDemand AI — ML Model Training, Quantile Calibration & Explainability Pipeline

Member 3 — Machine Learning & MLOps

Entrypoint proxying to src.geodemand.models.train_demand_model.
"""

from src.geodemand.models.train_demand_model import (
    ARTIFACT_DIR,
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    FEATURE_STORE_PATH,
    FORBIDDEN_COLUMNS,
    MODELS_DIR,
    RANDOM_SEED,
    ROOT,
    TARGET_COL,
    chronological_split,
    evaluate,
    load_and_validate,
    prepare_features,
    train_pipeline,
)

__all__ = [
    "ARTIFACT_DIR",
    "CATEGORICAL_COLUMNS",
    "DROP_COLUMNS",
    "FEATURE_STORE_PATH",
    "FORBIDDEN_COLUMNS",
    "MODELS_DIR",
    "RANDOM_SEED",
    "ROOT",
    "TARGET_COL",
    "chronological_split",
    "evaluate",
    "load_and_validate",
    "prepare_features",
    "train_pipeline",
]

if __name__ == "__main__":
    train_pipeline()
