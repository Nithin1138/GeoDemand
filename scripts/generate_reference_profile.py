"""
GeoDemand AI — Generate Reference Distribution Profile Artifact

Member 3 — Machine Learning & MLOps

Generates the canonical training reference distribution profile from the 70%
chronological training split and serializes it to models/reference_distributions.json.
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.geodemand.models.train_demand_model import (
    add_cyclical_features,
    load_and_validate,
    prepare_features,
    chronological_split,
)
from src.geodemand.monitoring.drift import ReferenceDistributionProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_reference_profile")


def main():
    logger.info("Loading feature store dataset...")
    df = load_and_validate()
    n = len(df)
    train_end = int(n * 0.70)
    train_cutoff_ts = df["decision_timestamp"].iloc[train_end]

    # Chronological training split (raw features before label encoding to preserve category names)
    train_df = df[df["decision_timestamp"] < train_cutoff_ts].copy()
    train_df = add_cyclical_features(train_df)

    # Filter to model feature columns
    schema_path = ROOT_DIR / "models" / "lgbm_v1" / "feature_schema.json"
    import json
    with open(schema_path) as f:
        schema = json.load(f)
    feature_cols = schema["feature_columns"]
    cat_cols = schema["categorical_columns"]

    train_features_df = train_df[feature_cols].copy()
    logger.info(f"Extracting reference statistics for {len(train_features_df):,} training rows...")

    profile = ReferenceDistributionProfile.from_dataframe(
        df=train_features_df,
        dataset_id=f"feature_store_train_split_70pct_v2.2",
        categorical_columns=cat_cols,
        sample_size_for_ks=5000,
        random_seed=42,
    )

    out_path_1 = ROOT_DIR / "models" / "reference_distributions.json"
    out_path_2 = ROOT_DIR / "models" / "lgbm_v1" / "reference_distributions.json"
    profile.save(out_path_1)
    profile.save(out_path_2)

    logger.info(f"✓ Reference profile successfully saved to {out_path_1} and {out_path_2}")


if __name__ == "__main__":
    main()
