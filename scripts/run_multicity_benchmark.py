r"""
GeoDemand AI — Multi-City Model Generalization Benchmark CLI (Phase 14.2)

Member 2 — Machine Learning & MLOps

Executes full multi-city evaluation across 4 Indian metropolitan areas:
1. Vijayawada (Tier-2 Baseline Hub)
2. Hyderabad (Tier-1 IT / Commercial Hub)
3. Bengaluru (Tier-1 Tech & Startup Hub)
4. Visakhapatnam (Tier-2 Coastal Industrial Hub)

Produces:
- 4x4 Train-One-City / Test-Another-City Transfer Matrix
- City-Specific vs Pooled Multi-City Model Performance
- Category-Wise Demand Breakdown across All Cities
- Non-Generalizable vs Universal Features Report
- Post-Inference City Calibration Scaling Factors
- Coordinate Routing Verification via CityModelSelector
- Artifacts: models/multi_city_benchmark.json & docs/CROSS_CITY_MODEL_EVALUATION.md
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

from src.geodemand.models.multicity_eval import (
    CityBoundaryError,
    CityModelSelector,
    MultiCityEvaluator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_multicity_benchmark")

# Forbidden columns to exclude from training
FORBIDDEN_COLUMNS = {
    "vendor_id",
    "h3_cell_id",
    "decision_timestamp",
    "target_timestamp",
    "customer_count",
    "expected_customer_count",
    "date",
    "timestamp",
    "profit",
    "revenue",
    "decision",
    "demand_score",
    "recommendation_score",
    "operating_cost",
    "fuel_cost",
    "city_id",
    "latitude",
    "longitude",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Multi-City Model Generalization Benchmark (Phase 14.2).")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT_DIR / "data" / "processed" / "feature_store.parquet"),
        help="Path to canonical feature store parquet.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of rows to sample from base feature store.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "models" / "multi_city_benchmark.json"),
        help="Output JSON benchmark path.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Feature store not found at {input_path}")
        return 1

    logger.info(f"Loading feature store from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from feature store.")

    if args.sample_size and len(df) > args.sample_size:
        logger.info(f"Sampling {args.sample_size:,} records for benchmark execution...")
        df = df.sample(args.sample_size, random_state=42).reset_index(drop=True)

    # Filter features: exclude identity, spatial coordinates, target, and forbidden columns
    feature_cols = [c for c in df.columns if c not in FORBIDDEN_COLUMNS]
    logger.info(f"Using {len(feature_cols)} generalizable features: {feature_cols}")

    evaluator = MultiCityEvaluator()
    logger.info("Executing Full Multi-City Generalization Benchmark...")
    report = evaluator.run_full_generalization_benchmark(df, feature_cols)

    # Save benchmark JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report.to_json(indent=2))
    logger.info(f"Saved benchmark results to {out_path}")

    # Display 4x4 Transfer Matrix
    cities = report.cities_evaluated
    print("\n" + "=" * 90)
    print("PHASE 14.2: MULTI-CITY MODEL GENERALIZATION & CROSS-CITY TRANSFER MATRIX")
    print("=" * 90)
    matrix_title = "Train City \\ Test City"
    header = f"{matrix_title:<24} | " + " | ".join(f"{c.capitalize():<12}" for c in cities)
    print(header)
    print("-" * 90)
    for train_c in cities:
        row_str = f"{train_c.capitalize():<24} | "
        cols = []
        for test_c in cities:
            mae_val = report.cross_city_transfer_matrix[train_c][test_c]
            diag = " *" if train_c == test_c else "  "
            cols.append(f"{mae_val:.4f}{diag}".ljust(12))
        row_str += " | ".join(cols)
        print(row_str)
    print("(* in-domain baseline)")
    print("=" * 90)

    # Display Calibration & Recovery Summary
    print("\n" + "=" * 90)
    print("ZERO-SHOT TRANSFER PENALTY & POST-INFERENCE CALIBRATION RECOVERY (TRAIN: VIJAYAWADA)")
    print("=" * 90)
    print(f"{'Target City':<16} | {'Baseline MAE':<12} | {'Transfer MAE':<12} | {'Penalty':<10} | {'Calib MAE':<12} | {'Recovery':<10}")
    print("-" * 90)
    for res in report.cross_city_transfer_results:
        print(
            f"{res.test_city.capitalize():<16} | "
            f"{res.in_domain_baseline_mae:<12.4f} | "
            f"{res.transfer_mae:<12.4f} | "
            f"+{res.transfer_penalty_pct:>6.1f}%   | "
            f"{res.calibrated_transfer_mae:<12.4f} | "
            f"{res.calibration_recovery_pct:>6.1f}%"
        )
    print("=" * 90)

    # Display Pooled vs Localized Comparison
    print("\n" + "=" * 90)
    print("POOLED MULTI-CITY VS LOCALIZED CITY-SPECIFIC ARCHITECTURE")
    print("=" * 90)
    comp = report.city_specific_vs_pooled_comparison
    print(f"Average Localized Model MAE:        {comp['average_localized_model_mae']:.4f}")
    print(f"Average Pooled Multi-City Model MAE:{comp['average_pooled_model_mae']:.4f}")
    print(f"Average Zero-Shot Transfer MAE:     {comp['average_zero_shot_transfer_mae']:.4f}")
    print(f"Average Calibrated Transfer MAE:    {comp['average_calibrated_transfer_mae']:.4f}")
    print(f"Calibration Recovery Benefit:       {comp['calibration_benefit_pct']:.1f}%")
    print("=" * 90)

    # Display Coordinate-Based Model Routing Demonstration
    print("\n" + "=" * 90)
    print("COORDINATE ROUTING & BOUNDARY ENFORCEMENT DEMONSTRATION")
    print("=" * 90)
    selector = CityModelSelector(strict_bounds=True)
    test_coords = [
        ("Vijayawada (Benz Circle)", 16.5062, 80.6480),
        ("Hyderabad (HITEC City)", 17.4435, 78.3772),
        ("Bengaluru (Indiranagar)", 12.9784, 77.6408),
        ("Visakhapatnam (Beach Road)", 17.7126, 83.3197),
    ]
    for label, lat, lng in test_coords:
        res = selector.resolve_model_for_coordinate(lat, lng)
        print(f"✓ {label:<28} -> City: {res.city_name:<14} | State: {res.state_code} | Scale: {res.calibration_factor}x | Model: {res.model_type}")

    # Demonstrate Boundary Violation Rejection
    out_of_bounds = ("Mumbai (Marine Drive)", 18.9438, 72.8232)
    try:
        selector.resolve_model_for_coordinate(out_of_bounds[1], out_of_bounds[2])
        print(f"✗ Boundary validation failed: allowed out-of-bounds coords for {out_of_bounds[0]}")
    except CityBoundaryError as e:
        print(f"✓ Boundary Safety Check: Rejected out-of-bounds ({out_of_bounds[1]}, {out_of_bounds[2]}) -> {e}")
    print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
