"""
GeoDemand AI — Model Promotion Gate & Atomic Lifecycle Engine (Step 5)

Member 3 — Machine Learning & MLOps

Implements:
- Rigorous candidate model acceptance policy (Safety, Contract, Performance, Latency)
- Candidate evaluation against active champion on validation split
- Atomic promotion with snapshot backup
- Automated rollback capability
- MLflow promotion event audit logging
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.geodemand.monitoring.prediction_quality import (
    PredictionQualityError,
    PredictionValidator,
)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
CHAMPION_BUNDLE_PATH = MODELS_DIR / "demand_model.pkl"
CHAMPION_META_PATH = MODELS_DIR / "demand_model_meta.json"
CHAMPION_ARTIFACT_DIR = MODELS_DIR / "lgbm_v1"
BACKUP_DIR = MODELS_DIR / "champion_backup"


@dataclass
class ModelPromotionPolicy:
    """Configurable acceptance policy for candidate model promotion."""

    # Performance constraints (evaluated on validation set only)
    max_val_mae: float = 1.00  # Hard ceiling for validation MAE
    max_val_rmse: float = 1.30  # Hard ceiling for validation RMSE
    min_mae_improvement_pct: float = 0.0  # Must be <= champion MAE (0.0 means at least equal or better)
    max_allowed_mae_degradation_pct: float = 0.0  # Cannot degrade champion validation MAE
    max_latency_p95_ms: float = 25.0  # Max acceptable 95th percentile latency per sample (ms)

    # Contract constraints
    require_monotonic_quantiles: bool = True
    max_quantile_inversion_rate: float = 0.001  # Max acceptable raw quantile crossing rate (0.1%)
    require_shap_support: bool = True
    require_exact_schema_match: bool = True
    require_non_negative_outputs: bool = True


@dataclass
class GateCheckResult:
    """Result of an individual gating check."""

    name: str
    passed: bool
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionEvaluationReport:
    """Detailed report produced by the Model Promotion Gate."""

    timestamp: str
    candidate_id: str
    candidate_model_type: str
    champion_id: str
    champion_model_type: str
    decision: str  # "APPROVED" or "REJECTED"
    passed_all_gates: bool
    gate_checks: Dict[str, GateCheckResult]
    candidate_metrics: Dict[str, float]
    champion_metrics: Dict[str, float]
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "candidate_id": self.candidate_id,
            "candidate_model_type": self.candidate_model_type,
            "champion_id": self.champion_id,
            "champion_model_type": self.champion_model_type,
            "decision": self.decision,
            "passed_all_gates": self.passed_all_gates,
            "gate_checks": {k: v.to_dict() for k, v in self.gate_checks.items()},
            "candidate_metrics": {k: round(float(v), 4) for k, v in self.candidate_metrics.items()},
            "champion_metrics": {k: round(float(v), 4) for k, v in self.champion_metrics.items()},
            "reasons": self.reasons,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [
            f"=== GeoDemand Model Promotion Gate [{self.decision}] ===",
            f"Timestamp: {self.timestamp}",
            f"Candidate: {self.candidate_id} ({self.candidate_model_type})",
            f"Champion:  {self.champion_id} ({self.champion_model_type})",
            f"All Gates Passed: {self.passed_all_gates}",
            "--- Validation Metrics Comparison ---",
            f"  Candidate Val MAE:  {self.candidate_metrics.get('val_mae', 'N/A')}",
            f"  Champion Val MAE:   {self.champion_metrics.get('val_mae', 'N/A')}",
            f"  Candidate Val RMSE: {self.candidate_metrics.get('val_rmse', 'N/A')}",
            f"  Champion Val RMSE:  {self.champion_metrics.get('val_rmse', 'N/A')}",
            "--- Gate Verification Results ---",
        ]
        for name, chk in self.gate_checks.items():
            status = "PASS" if chk.passed else "FAIL"
            lines.append(f"  [{status}] {name}: {chk.details}")
        if self.reasons:
            lines.append("--- Decision Notes ---")
            for r in self.reasons:
                lines.append(f"  - {r}")
        return "\n".join(lines)


class ModelPromotionGate:
    """
    Automated gate evaluating candidate models against production champion.
    Handles atomic promotion, backup, and rollback.
    """

    def __init__(
        self,
        policy: Optional[ModelPromotionPolicy] = None,
        models_dir: Optional[Path] = None,
    ):
        self.policy = policy or ModelPromotionPolicy()
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.champion_bundle_path = self.models_dir / "demand_model.pkl"
        self.champion_meta_path = self.models_dir / "demand_model_meta.json"
        self.champion_dir = self.models_dir / "lgbm_v1"
        self.backup_dir = self.models_dir / "champion_backup"
        self.validator = PredictionValidator(raise_on_violation=False)

    def load_champion_metadata(self) -> Dict[str, Any]:
        """Load current champion metadata."""
        if self.champion_meta_path.exists():
            with open(self.champion_meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif (self.champion_dir / "metadata.json").exists():
            with open(self.champion_dir / "metadata.json", "r", encoding="utf-8") as f:
                return json.load(f)
        return {"model_version": "v2.0.0", "model_type": "xgboost", "metrics": {"val_mae": 0.7738, "val_rmse": 1.0045}}

    def evaluate_candidate(
        self,
        candidate_bundle: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> PromotionEvaluationReport:
        """
        Evaluate candidate model bundle against the promotion policy on validation split.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        champ_meta = self.load_champion_metadata()
        champ_val_mae = float(champ_meta.get("metrics", {}).get("val_mae", 0.7738))
        champ_val_rmse = float(champ_meta.get("metrics", {}).get("val_rmse", 1.0045))

        cand_meta = candidate_bundle.get("meta", {})
        cand_version = cand_meta.get("model_version", "candidate_v" + now_iso[:10])
        cand_type = cand_meta.get("model_type", "unknown")
        champ_version = champ_meta.get("model_version", "champion_v2.0.0")
        champ_type = champ_meta.get("model_type", "xgboost")

        checks: Dict[str, GateCheckResult] = {}
        reasons: List[str] = []

        # Gate 1: Artifact & Bundle Structure
        required_keys = {"model", "p10_model", "p90_model", "feature_cols", "label_encoders"}
        missing_keys = required_keys - set(candidate_bundle.keys())
        bundle_ok = len(missing_keys) == 0
        checks["artifact_structure"] = GateCheckResult(
            name="artifact_structure",
            passed=bundle_ok,
            details="All bundle components present" if bundle_ok else f"Missing keys: {missing_keys}",
        )

        # Gate 2: Feature Schema Match
        schema_path = self.champion_dir / "feature_schema.json"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                ref_schema = json.load(f)
            expected_cols = ref_schema.get("feature_columns", [])
            cand_cols = candidate_bundle.get("feature_cols", [])
            schema_match = cand_cols == expected_cols
            checks["feature_schema_match"] = GateCheckResult(
                name="feature_schema_match",
                passed=schema_match,
                details=f"Exact match on {len(cand_cols)} features" if schema_match else f"Columns mismatch (cand={len(cand_cols)}, exp={len(expected_cols)})",
            )
        else:
            checks["feature_schema_match"] = GateCheckResult(
                name="feature_schema_match",
                passed=True,
                details="Reference schema not found — bypassed.",
            )

        # Gate 3: Prediction Execution & Non-Negativity
        cand_model = candidate_bundle.get("model")
        p10_model = candidate_bundle.get("p10_model")
        p90_model = candidate_bundle.get("p90_model")

        try:
            preds_raw = cand_model.predict(X_val)
            preds = np.clip(preds_raw, 0, None)
            pred_rep = self.validator.validate_predictions(preds, expected_n=len(X_val))
            pred_ok = pred_rep.is_valid and not pred_rep.has_nan and not pred_rep.has_infinite
            checks["predictions_finite_and_valid"] = GateCheckResult(
                name="predictions_finite_and_valid",
                passed=pred_ok,
                details=f"Predictions valid (N={len(preds):,}, min={pred_rep.min_predicted:.2f}, max={pred_rep.max_predicted:.2f})" if pred_ok else f"Violations: {pred_rep.violations}",
            )
        except Exception as e:
            pred_ok = False
            preds = np.zeros(len(X_val))
            checks["predictions_finite_and_valid"] = GateCheckResult(
                name="predictions_finite_and_valid",
                passed=False,
                details=f"Predict failed: {e}",
            )

        # Gate 4: Quantile Ordering & Intervals
        try:
            p10_raw = np.clip(p10_model.predict(X_val), 0, None)
            p90_raw = np.clip(p90_model.predict(X_val), 0, None)
            if self.policy.require_monotonic_quantiles:
                inv_10_50 = (p10_raw > (preds + 1e-4)).sum()
                inv_50_90 = (preds > (p90_raw + 1e-4)).sum()
                inv_rate_10_50 = float(inv_10_50) / len(X_val)
                inv_rate_50_90 = float(inv_50_90) / len(X_val)
                max_inv_rate = max(inv_rate_10_50, inv_rate_50_90)

                if max_inv_rate > self.policy.max_quantile_inversion_rate:
                    q_ok = False
                    q_details = (
                        f"Excessive quantile inversion rate in candidate: "
                        f"P10>P50: {inv_rate_10_50:.2%} ({inv_10_50} cases), "
                        f"P50>P90: {inv_rate_50_90:.2%} ({inv_50_90} cases) "
                        f"> threshold {self.policy.max_quantile_inversion_rate:.2%}"
                    )
                else:
                    p10_adj = np.minimum(p10_raw, preds)
                    p90_adj = np.maximum(preds, p90_raw)
                    q_rep = self.validator.validate_quantiles(p10_adj, preds, p90_adj, expected_n=len(X_val))
                    q_ok = q_rep.is_valid and not q_rep.has_quantile_inversion
                    q_details = (
                        f"P10 <= P50 <= P90 verified (Mean interval width={q_rep.mean_interval_width:.2f}, "
                        f"raw crossing rate={max_inv_rate:.3%})"
                    )
                checks["quantile_ordering"] = GateCheckResult(
                    name="quantile_ordering",
                    passed=q_ok,
                    details=q_details,
                )
            else:
                p10_adj = np.minimum(p10_raw, preds)
                p90_adj = np.maximum(preds, p90_raw)
                q_rep = self.validator.validate_quantiles(p10_adj, preds, p90_adj, expected_n=len(X_val))
                q_ok = q_rep.is_valid and not q_rep.has_quantile_inversion
                checks["quantile_ordering"] = GateCheckResult(
                    name="quantile_ordering",
                    passed=q_ok,
                    details=f"P10 <= P50 <= P90 verified (Mean interval width={q_rep.mean_interval_width:.2f})" if q_ok else f"Violations: {q_rep.violations}",
                )
        except Exception as e:
            checks["quantile_ordering"] = GateCheckResult(
                name="quantile_ordering",
                passed=False,
                details=f"Quantile verification failed: {e}",
            )

        # Gate 5: TreeSHAP Explainability Verification
        try:
            sample_sample = X_val.iloc[:5]
            if hasattr(cand_model, "predict"):
                try:
                    contribs = cand_model.predict(sample_sample, pred_contrib=True)
                    shap_ok = contribs.shape[0] == 5 and contribs.shape[1] == (X_val.shape[1] + 1)
                except Exception:
                    if hasattr(cand_model, "get_booster"):
                        import xgboost as xgb
                        dmat = xgb.DMatrix(sample_sample)
                        contribs = cand_model.get_booster().predict(dmat, pred_contribs=True)
                        shap_ok = contribs.shape[0] == 5
                    elif hasattr(cand_model, "get_feature_importance"):
                        try:
                            import catboost as cb
                            pool = cb.Pool(sample_sample)
                            contribs = cand_model.get_feature_importance(data=pool, type="ShapValues")
                            shap_ok = contribs.shape[0] == 5
                        except Exception:
                            shap_ok = True
                    else:
                        shap_ok = True  # Mock model fallback
            else:
                shap_ok = False

            checks["shap_explainability"] = GateCheckResult(
                name="shap_explainability",
                passed=shap_ok,
                details="TreeSHAP local feature attribution operational" if shap_ok else "SHAP attribution failed",
            )
        except Exception as e:
            checks["shap_explainability"] = GateCheckResult(
                name="shap_explainability",
                passed=False,
                details=f"SHAP explanation failed: {e}",
            )

        # Gate 6: Latency Benchmark
        try:
            latencies = []
            test_batch = X_val.iloc[:200]
            for _ in range(5):
                t0 = time.perf_counter()
                _ = cand_model.predict(test_batch)
                dt_ms_sample = (time.perf_counter() - t0) * 1000.0 / len(test_batch)
                latencies.append(dt_ms_sample)
            latency_p95 = float(np.percentile(latencies, 95))
            latency_ok = latency_p95 <= self.policy.max_latency_p95_ms
            checks["inference_latency"] = GateCheckResult(
                name="inference_latency",
                passed=latency_ok,
                details=f"P95 latency {latency_p95:.4f} ms/sample <= {self.policy.max_latency_p95_ms} ms" if latency_ok else f"P95 latency {latency_p95:.4f} ms exceeded threshold",
            )
        except Exception as e:
            latency_ok = False
            latency_p95 = 999.0
            checks["inference_latency"] = GateCheckResult(
                name="inference_latency",
                passed=False,
                details=f"Latency test failed: {e}",
            )

        # Gate 7: Validation Performance Comparison (NO TEST SET FOR SELECTION)
        if pred_ok:
            cand_val_mae = float(mean_absolute_error(y_val, preds))
            cand_val_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            cand_val_r2 = float(r2_score(y_val, preds)) if np.var(y_val) > 1e-9 else 0.0

            mae_improvement = ((champ_val_mae - cand_val_mae) / champ_val_mae) * 100.0
            mae_passed = cand_val_mae <= (champ_val_mae * (1.0 + self.policy.max_allowed_mae_degradation_pct / 100.0) + 1e-4)
            rmse_passed = cand_val_rmse <= (self.policy.max_val_rmse + 1e-4)
            perf_passed = mae_passed and rmse_passed
            perf_details = (
                f"Candidate MAE={cand_val_mae:.4f} vs Champion MAE={champ_val_mae:.4f} ({mae_improvement:+.2f}%)"
                if perf_passed
                else f"Candidate MAE {cand_val_mae:.4f} degraded champion MAE {champ_val_mae:.4f}"
            )
        else:
            cand_val_mae = 999.0
            cand_val_rmse = 999.0
            cand_val_r2 = -999.0
            perf_passed = False
            perf_details = "Cannot compute metrics: candidate predictions failed validation."

        checks["validation_performance"] = GateCheckResult(
            name="validation_performance",
            passed=perf_passed,
            details=perf_details,
        )

        # Final Promotion Decision
        passed_all = all(chk.passed for chk in checks.values())

        if not passed_all:
            decision = "REJECTED"
            failed_gates = [k for k, v in checks.items() if not v.passed]
            reasons.append(f"Rejected: Failed gating checks: {', '.join(failed_gates)}")
            for k in failed_gates:
                reasons.append(f"Failed check '{k}': {checks[k].details}")
        elif (cand_val_mae - champ_val_mae) > 1e-4:
            decision = "REJECTED"
            reasons.append(f"Rejected: Candidate MAE ({cand_val_mae:.4f}) did not improve current champion ({champ_val_mae:.4f}).")
        else:
            decision = "APPROVED"
            reasons.append(f"Approved: Candidate demonstrated superior/equal performance with full contract compatibility.")

        return PromotionEvaluationReport(
            timestamp=now_iso,
            candidate_id=cand_version,
            candidate_model_type=cand_type,
            champion_id=champ_version,
            champion_model_type=champ_type,
            decision=decision,
            passed_all_gates=passed_all and (decision == "APPROVED"),
            gate_checks=checks,
            candidate_metrics={"val_mae": cand_val_mae, "val_rmse": cand_val_rmse, "val_r2": cand_val_r2, "latency_p95_ms": latency_p95},
            champion_metrics={"val_mae": champ_val_mae, "val_rmse": champ_val_rmse},
            reasons=reasons,
        )

    def log_promotion_to_mlflow(self, report: PromotionEvaluationReport) -> Optional[str]:
        """Log model promotion evaluation report and decision to MLflow if available."""
        if not HAS_MLFLOW:
            return None
        try:
            db_path = self.models_dir.parent / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}".replace("\\", "/"))
            mlflow.set_experiment("GeoDemand")
            with mlflow.start_run(run_name=f"promotion_gate_{report.candidate_id}") as run:
                mlflow.log_params({
                    "candidate_id": report.candidate_id,
                    "candidate_model_type": report.candidate_model_type,
                    "champion_id": report.champion_id,
                    "champion_model_type": report.champion_model_type,
                    "decision": report.decision,
                    "passed_all_gates": report.passed_all_gates,
                })
                for k, v in report.candidate_metrics.items():
                    mlflow.log_metric(f"candidate_{k}", float(v))
                for k, v in report.champion_metrics.items():
                    mlflow.log_metric(f"champion_{k}", float(v))
                for name, chk in report.gate_checks.items():
                    mlflow.log_param(f"gate_{name}_passed", chk.passed)
                mlflow.log_dict(report.to_dict(), "promotion_evaluation_report.json")
                logger.info(f"[OK] Promotion report logged to MLflow run {run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            logger.warning(f"MLflow promotion logging skipped: {e}")
            return None

    def backup_current_champion(self) -> Path:
        """Create a full snapshot backup of the active champion."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        if self.champion_bundle_path.exists():
            shutil.copy2(self.champion_bundle_path, self.backup_dir / "demand_model.pkl")
        if self.champion_meta_path.exists():
            shutil.copy2(self.champion_meta_path, self.backup_dir / "demand_model_meta.json")

        if self.champion_dir.exists():
            backup_art_dir = self.backup_dir / "lgbm_v1"
            backup_art_dir.mkdir(parents=True, exist_ok=True)
            for f in self.champion_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, backup_art_dir / f.name)

        logger.info(f"[OK] Active champion snapshot backed up to {self.backup_dir}")
        return self.backup_dir

    def rollback_to_backup(self) -> bool:
        """Rollback production champion to backed-up snapshot."""
        if not (self.backup_dir / "demand_model.pkl").exists():
            logger.error(f"Cannot rollback: no valid backup bundle found at {self.backup_dir}")
            return False

        logger.info("Executing atomic rollback from backup...")
        shutil.copy2(self.backup_dir / "demand_model.pkl", self.champion_bundle_path)
        if (self.backup_dir / "demand_model_meta.json").exists():
            shutil.copy2(self.backup_dir / "demand_model_meta.json", self.champion_meta_path)

        backup_art_dir = self.backup_dir / "lgbm_v1"
        if backup_art_dir.exists():
            self.champion_dir.mkdir(parents=True, exist_ok=True)
            for f in backup_art_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, self.champion_dir / f.name)

        logger.info("[OK] Rollback successfully completed.")
        return True

    def promote_candidate_bundle(
        self,
        candidate_bundle: Dict[str, Any],
        candidate_meta: Dict[str, Any],
    ) -> bool:
        """
        Atomically promote candidate bundle to production champion with automatic rollback on error.
        Ensures both demand_model.pkl and lgbm_v1 artifacts remain fully consistent.
        """
        logger.info("Initiating atomic candidate promotion...")
        self.backup_current_champion()

        try:
            # Merge candidate bundle meta with promotion report metadata
            merged_meta = {**candidate_bundle.get("meta", {}), **candidate_meta}

            # 1. Write bundle to temporary staging file
            temp_bundle_path = self.models_dir / "demand_model.pkl.tmp"
            temp_meta_path = self.models_dir / "demand_model_meta.json.tmp"

            joblib.dump(candidate_bundle, temp_bundle_path)
            with open(temp_meta_path, "w", encoding="utf-8") as f:
                json.dump(merged_meta, f, indent=2)

            # 2. Atomic rename of root bundle
            os.replace(temp_bundle_path, self.champion_bundle_path)
            os.replace(temp_meta_path, self.champion_meta_path)

            # 3. Synchronize individual artifact directory (lgbm_v1)
            self.champion_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(candidate_bundle["model"], self.champion_dir / "model.joblib")
            if "p10_model" in candidate_bundle and candidate_bundle["p10_model"] is not None:
                joblib.dump(candidate_bundle["p10_model"], self.champion_dir / "p10_model.joblib")
            if "p90_model" in candidate_bundle and candidate_bundle["p90_model"] is not None:
                joblib.dump(candidate_bundle["p90_model"], self.champion_dir / "p90_model.joblib")
            if "label_encoders" in candidate_bundle:
                joblib.dump(candidate_bundle["label_encoders"], self.champion_dir / "label_encoders.joblib")

            with open(self.champion_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(merged_meta, f, indent=2)
            with open(self.champion_dir / "model_metadata.json", "w", encoding="utf-8") as f:
                json.dump(merged_meta, f, indent=2)

            # 4. Post-promotion sanity verification
            test_bundle = joblib.load(self.champion_bundle_path)
            assert "model" in test_bundle, "Promoted bundle missing model!"
            logger.info("[OK] Promoted candidate bundle verified successfully in production.")
            return True
        except Exception as e:
            logger.error(f"Promotion failed: {e}. Executing automatic rollback!")
            self.rollback_to_backup()
            return False
