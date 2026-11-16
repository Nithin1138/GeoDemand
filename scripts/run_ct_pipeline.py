r"""
GeoDemand AI — Continuous Training Pipeline CLI (Phase 13.3)

Member 2 — Machine Learning & MLOps

Executes:
1. Scheduled or triggered continuous training cycles
2. Data eligibility verification
3. Candidate model training and evaluation against active champion
4. Promotion gating, staging, and approval
5. Atomic rollback execution
6. Audit log recording and zero-downtime hot-reloading
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.geodemand.monitoring.ct_pipeline import (
    ContinuousTrainingPipeline,
    RetrainingTrigger,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_ct_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description="GeoDemand AI Continuous Training Pipeline CLI")
    parser.add_argument(
        "--trigger",
        type=str,
        choices=["SCHEDULED", "DRIFT_TRIGGERED", "PERFORMANCE_DEGRADATION", "MANUAL"],
        default="MANUAL",
        help="Event trigger type.",
    )
    parser.add_argument("--min-samples", type=int, default=500, help="Minimum new sample threshold.")
    parser.add_argument("--require-approval", action="store_true", help="Stage candidate for manual approval.")
    parser.add_argument("--approve-token", type=str, default=None, help="Approval token to promote a staged model.")
    parser.add_argument("--rollback", action="store_true", help="Roll back to previous champion snapshot.")
    parser.add_argument("--force", action="store_true", help="Force retraining ignoring data volume check.")

    args = parser.parse_args()

    pipeline = ContinuousTrainingPipeline(
        min_new_samples=args.min_samples,
        require_manual_approval=args.require_approval,
    )

    # 1. Rollback Action
    if args.rollback:
        logger.info("Executing rollback to previous production champion...")
        try:
            rb_res = pipeline.rollback_to_previous_champion()
            print("\n" + "=" * 70)
            print("ROLLBACK EXECUTION RESULT")
            print("=" * 70)
            print(f"Status:   {'SUCCESS' if rb_res.get('success') else 'FAILED'}")
            print(f"Message:  {rb_res.get('message')}")
            print("=" * 70 + "\n")
            return 0 if rb_res.get("success") else 1
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return 1

    # 2. Operator Approval Action
    if args.approve_token:
        logger.info(f"Approving and promoting candidate with token '{args.approve_token}'...")
        try:
            res = pipeline.approve_and_promote(args.approve_token)
            print("\n" + "=" * 70)
            print("MANUAL APPROVAL & PROMOTION RESULT")
            print("=" * 70)
            print(f"Candidate ID: {res.candidate_id}")
            print(f"Status:       {res.status}")
            print(f"Message:      {res.message}")
            print("=" * 70 + "\n")
            return 0
        except Exception as e:
            logger.error(f"Approval promotion failed: {e}")
            return 1

    # 3. Continuous Retraining Cycle
    trigger_enum = RetrainingTrigger(args.trigger)
    logger.info(f"Starting CT cycle with trigger: {trigger_enum.value}...")
    res = pipeline.run_retraining_cycle(trigger=trigger_enum, force=args.force)

    print("\n" + "=" * 80)
    print("CONTINUOUS TRAINING (CT) CYCLE EXECUTION RESULT")
    print("=" * 80)
    print(f"Cycle ID:           {res.cycle_id}")
    print(f"Trigger:            {res.trigger}")
    print(f"Status:             {res.status}")
    print(f"Champion ID:        {res.champion_id} (Val MAE: {res.champion_val_mae:.4f})")
    print(f"Candidate ID:       {res.candidate_id or 'None'} (Val MAE: {res.candidate_val_mae or 0.0:.4f})")
    print(f"Approval Token:     {res.approval_token or 'None'}")
    print(f"Rollback Available: {res.rollback_available}")
    print(f"Summary Message:    {res.message}")
    if res.gate_checks:
        print("\nPromotion Gate Checks:")
        for check in res.gate_checks:
            passed_sym = "[PASS]" if check.get("passed") else "[FAIL]"
            print(f"  {passed_sym} {check.get('name')}: {check.get('details')}")
    print("=" * 80 + "\n")

    return 0 if res.status in ["COMPLETED_PROMOTED", "COMPLETED_STAGED_AWAITING_APPROVAL", "SKIPPED_INSUFFICIENT_DATA"] else 1


if __name__ == "__main__":
    sys.exit(main())
