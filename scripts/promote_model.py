"""
GeoDemand AI — Model Promotion & Rollback CLI Script (Step 5)

Member 3 — Machine Learning & MLOps

Usage:
    python scripts/promote_model.py --check-champion
    python scripts/promote_model.py --evaluate-candidate <bundle_path>
    python scripts/promote_model.py --evaluate-candidate <bundle_path> --promote
    python scripts/promote_model.py --evaluate-candidate <bundle_path> --sample-size 5000
    python scripts/promote_model.py --rollback
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd

from src.geodemand.models.promotion import (
    ModelPromotionGate,
    ModelPromotionPolicy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("promote_model")


def main():
    parser = argparse.ArgumentParser(description="GeoDemand Model Promotion and Rollback CLI")
    parser.add_argument("--check-champion", action="store_true", help="Audit current champion metadata and status")
    parser.add_argument("--evaluate-candidate", type=str, default=None, help="Path to candidate bundle joblib/pkl file")
    parser.add_argument("--promote", action="store_true", help="Promote candidate if it passes all gates")
    parser.add_argument("--rollback", action="store_true", help="Rollback production champion to backed-up snapshot")
    parser.add_argument("--backup", action="store_true", help="Create an explicit snapshot backup of the current champion")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for quick validation check")
    args = parser.parse_args()

    gate = ModelPromotionGate()

    if args.backup:
        backup_path = gate.backup_current_champion()
        print(f"Champion backup snapshot created at: {backup_path}")
        return

    if args.rollback:
        success = gate.rollback_to_backup()
        if success:
            print("[OK] Production champion rolled back successfully.")
        else:
            print("[FAIL] Rollback failed. Check logs for details.")
            sys.exit(1)
        return

    if args.check_champion:
        meta = gate.load_champion_metadata()
        print("\n=== Current Active Champion ===")
        print(json.dumps(meta, indent=2))
        return

    if args.evaluate_candidate:
        cand_path = Path(args.evaluate_candidate)
        if not cand_path.exists():
            logger.error(f"Candidate file not found at {cand_path}")
            sys.exit(1)

        logger.info(f"Loading candidate bundle from {cand_path}...")
        candidate_bundle = joblib.load(cand_path)

        logger.info("Loading validation split for candidate gating...")
        from src.geodemand.models.train_demand_model import (
            chronological_split,
            load_and_validate,
            prepare_features,
        )
        df = load_and_validate()
        if args.sample_size and len(df) > args.sample_size:
            logger.info(f"Subsampling to {args.sample_size} rows for quick audit...")
            df = df.head(args.sample_size).copy()

        n = len(df)
        train_end = int(n * 0.70)
        train_mask = pd.Series([False] * n)
        train_mask.iloc[:train_end] = True

        X, y, _, _ = prepare_features(df, train_mask=train_mask)
        _, _, X_val, y_val, _, _ = chronological_split(X, y, df["decision_timestamp"])

        report = gate.evaluate_candidate(candidate_bundle, X_val, y_val)
        print("\n" + report.summary() + "\n")

        # Log report to MLflow audit trail if available
        gate.log_promotion_to_mlflow(report)

        if args.promote:
            if report.passed_all_gates:
                logger.info("Candidate approved! Executing atomic promotion...")
                success = gate.promote_candidate_bundle(
                    candidate_bundle=candidate_bundle,
                    candidate_meta=report.to_dict(),
                )
                if success:
                    print("[OK] Candidate promoted to champion successfully.")
                else:
                    print("[FAIL] Promotion failed during execution.")
                    sys.exit(1)
            else:
                print("[FAIL] Promotion aborted: Candidate failed required gating checks.")
                sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
