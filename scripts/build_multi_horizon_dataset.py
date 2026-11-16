r"""
GeoDemand AI — Chronological Multi-Horizon Dataset Builder (Phase 13.1)

Member 2 — Machine Learning & MLOps

Builds:
1. Multi-horizon chronological training dataset with forward targets (h=1, 2, 4, 8)
2. Strictly past lag features (t-1h, t-2h, t-24h)
3. Shifted rolling window statistics (3h, 6h, 24h mean, std, min, max)
4. Historical demand baselines with anti-leakage protection
5. Time alignment audit
6. Zero future-leakage verification
7. Chronological forward-chaining splits with horizon embargo
8. Exports feature specification and split metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.geodemand.models.multi_horizon import (
    DEFAULT_HORIZONS,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    MultiHorizonFeatureEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_multi_horizon_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build chronological multi-horizon feature store.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "feature_store.parquet"),
        help="Path to input canonical feature store parquet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "multi_horizon_feature_store.parquet"),
        help="Path to output multi-horizon parquet.",
    )
    parser.add_argument(
        "--meta-output",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "multi_horizon_splits_meta.json"),
        help="Path to save split metadata.",
    )
    parser.add_argument(
        "--spec-output",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "multi_horizon_feature_spec.json"),
        help="Path to save feature specification.",
    )
    parser.add_argument("--train-end", type=str, default="2025-10-01", help="Train split cutoff date.")
    parser.add_argument("--val-end", type=str, default="2025-11-01", help="Validation split cutoff date.")
    parser.add_argument("--embargo-hours", type=int, default=8, help="Embargo buffer hours before val/test boundaries.")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for rapid validation.")
    parser.add_argument("--skip-save-parquet", action="store_true", help="Skip saving large parquet file.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input feature store not found at: {input_path}")
        return 1

    logger.info(f"Loading feature store from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded raw feature store with shape {df.shape}")

    if args.sample_size and len(df) > args.sample_size:
        logger.info(f"Sampling {args.sample_size} rows chronologically for demonstration...")
        df = df.sort_values("decision_timestamp").head(args.sample_size).reset_index(drop=True)

    group_cols = [c for c in ["h3_cell_id", "vendor_id"] if c in df.columns]
    target_col = "expected_customer_count"
    timestamp_col = "decision_timestamp"

    # Step 1: Build Multi-Horizon Dataset
    logger.info("Building multi-horizon dataset with forward targets, lags, and rolling statistics...")
    engine = MultiHorizonFeatureEngine()
    mh_df = engine.build_multi_horizon_dataset(
        df=df,
        horizons=tuple(DEFAULT_HORIZONS),
        target_col=target_col,
        group_cols=group_cols,
        lags=(1, 2, 24),
        rolling_windows=(3, 6, 24),
        include_historical_baselines=True,
        train_end=args.train_end,
        timestamp_col=timestamp_col,
        drop_tail_na=True,
    )
    logger.info(f"Enriched multi-horizon dataset shape: {mh_df.shape}")

    # Step 2: Time Alignment Audit
    logger.info("Auditing temporal alignment...")
    alignment_report = engine.verify_time_alignment(
        df=mh_df,
        timestamp_col=timestamp_col,
        group_cols=group_cols,
        horizons=tuple(DEFAULT_HORIZONS),
        target_col=target_col,
    )
    logger.info(f"Time Alignment Audit Result: is_valid={alignment_report.is_valid}, violations={len(alignment_report.violations)}")
    if not alignment_report.is_valid:
        logger.warning(f"Time alignment violations: {alignment_report.violations}")

    # Step 3: Zero-Leakage Audit
    logger.info("Verifying zero future-data leakage...")
    leakage_report = engine.verify_zero_future_leakage(
        df=mh_df,
        target_col=target_col,
        horizons=tuple(DEFAULT_HORIZONS),
        timestamp_col=timestamp_col,
        group_cols=group_cols,
    )
    logger.info(f"Zero Leakage Audit Result: has_leakage={leakage_report.has_leakage}, violations={len(leakage_report.violations)}")
    if leakage_report.has_leakage:
        logger.error(f"Leakage violations: {leakage_report.violations}")
        return 1

    # Step 4: Chronological Forward-Chaining Splits with Horizon Embargo
    logger.info(f"Generating chronological splits (train < {args.train_end}, val < {args.val_end}, embargo={args.embargo_hours}h)...")
    splits = engine.create_multi_horizon_splits(
        df=mh_df,
        train_end=args.train_end,
        val_end=args.val_end,
        embargo_hours=args.embargo_hours,
        timestamp_col=timestamp_col,
        target_col=target_col,
        horizons=tuple(DEFAULT_HORIZONS),
    )
    logger.info(
        f"Splits generated: Train={splits.summary_metadata['train_samples']:,} | "
        f"Val={splits.summary_metadata['val_samples']:,} | "
        f"Test={splits.summary_metadata['test_samples']:,}"
    )

    # Step 5: Save Split Metadata and Feature Specification
    meta_path = Path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(splits.to_dict(), f, indent=2)
    logger.info(f"Saved multi-horizon split metadata to {meta_path}")

    # Construct feature specification
    spec_path = Path(args.spec_output)
    feature_columns = [
        c for c in mh_df.columns
        if c not in [timestamp_col, "target_timestamp", target_col, *group_cols]
        and not any(c.startswith(f"{target_col}_h") for _ in [1])
        and not c.startswith("target_timestamp_h")
    ]
    target_columns = [f"{target_col}_h{h}h" for h in DEFAULT_HORIZONS]

    feature_spec = {
        "version": "v2.2-chronological-multihorizon",
        "description": "Multi-horizon feature store specification with leak-free lags, rolling stats, and historical baselines",
        "horizons_hours": DEFAULT_HORIZONS,
        "primary_target": target_col,
        "horizon_targets": target_columns,
        "n_trainable_features": len(feature_columns),
        "feature_columns": sorted(feature_columns),
        "lag_features": [c for c in feature_columns if "_lag_" in c],
        "rolling_features": [c for c in feature_columns if "_rolling_" in c],
        "historical_demand_baselines": [c for c in feature_columns if "_hist_" in c],
        "time_alignment_audit": alignment_report.to_dict(),
        "leakage_audit": leakage_report.to_dict(),
    }
    with open(spec_path, "w") as f:
        json.dump(feature_spec, f, indent=2)
    logger.info(f"Saved multi-horizon feature specification to {spec_path}")

    # Step 6: Save Multi-Horizon Parquet
    if not args.skip_save_parquet:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mh_df.to_parquet(out_path, index=False)
        logger.info(f"Saved full multi-horizon dataset ({len(mh_df):,} rows, {mh_df.shape[1]} cols) to {out_path}")

    print("\n" + "=" * 70)
    print("PHASE 13.1 MULTI-HORIZON DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total Rows Processed:   {len(mh_df):,}")
    print(f"Total Columns:          {mh_df.shape[1]}")
    print(f"Trainable Features:     {len(feature_columns)}")
    print(f"Forward Horizons:       {DEFAULT_HORIZONS} hours")
    print(f"Train Split:            {splits.summary_metadata['train_samples']:,} samples ({splits.summary_metadata['train_range']})")
    print(f"Val Split:              {splits.summary_metadata['val_samples']:,} samples ({splits.summary_metadata['val_range']})")
    print(f"Test Split:             {splits.summary_metadata['test_samples']:,} samples ({splits.summary_metadata['test_range']})")
    print(f"Embargo Buffer:         {args.embargo_hours} hours purged between partitions")
    print(f"Time Alignment Valid:   {alignment_report.is_valid}")
    print(f"Leakage Free:           {not leakage_report.has_leakage}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
