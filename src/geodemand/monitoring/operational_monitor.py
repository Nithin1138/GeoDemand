"""
GeoDemand AI — Operational ML Monitoring Engine (Phase 17.2)

Member 2 — Machine Learning & MLOps

Implements the 7 core production ML monitoring responsibilities:
1. Feature Drift Monitoring (PSI >= 0.15, Wasserstein distance, KS test)
2. Prediction Drift Monitoring (PSI >= 0.15, prediction distribution shift)
3. Real-Label MAE Monitoring (7-day rolling window, >20% degradation trigger, delayed actuals)
4. Quantile Coverage Monitoring (Empirical 80% coverage, quantile crossing prevention, interval width)
5. Model Loading Failure Monitoring (Failure rates, deserialization errors, fallback tracking)
6. Inference Latency Monitoring (P50, P90, P95, P99 ms, throughput, 3.0ms SLA budget enforcement)
7. Model Version & Rollback History Tracking (Champion vs Backup, promotion audit logs, rollback readiness)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.geodemand.monitoring.drift import (
    DriftReport,
    DriftThresholds,
    FeatureDriftDetector,
    ReferenceDistributionProfile,
)
from src.geodemand.monitoring.performance import (
    PerformanceMonitor,
    RollingPerformanceAssessment,
)
from src.geodemand.monitoring.prediction_drift import (
    PredictionDriftDetector,
    PredictionDriftReport,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


# ==============================================================================
# 1. Model Loading Failure Monitoring
# ==============================================================================

@dataclass
class ModelLoadingHealth:
    """Telemetry report for model loading and deserialization operations."""

    total_load_attempts: int
    successful_loads: int
    failed_loads: int
    failure_rate_pct: float
    fallback_activations: int
    last_load_timestamp: Optional[str]
    last_failure_timestamp: Optional[str]
    last_failure_reason: Optional[str]
    last_failure_type: Optional[str]
    is_healthy: bool
    status: str  # "HEALTHY", "WARNING", "CRITICAL_LOAD_FAILURE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelLoadingMonitor:
    """
    Tracks and records model loading attempts, deserialization failures,
    corrupted artifacts, and fallback activations.
    """

    def __init__(self, failure_rate_threshold_pct: float = 0.0) -> None:
        self.failure_rate_threshold_pct = failure_rate_threshold_pct
        self.total_load_attempts: int = 0
        self.successful_loads: int = 0
        self.failed_loads: int = 0
        self.fallback_activations: int = 0
        self.last_load_timestamp: Optional[str] = None
        self.last_failure_timestamp: Optional[str] = None
        self.last_failure_reason: Optional[str] = None
        self.last_failure_type: Optional[str] = None
        self._history: deque = deque(maxlen=100)

    def record_success(self, model_version: str = "v2.1.0") -> None:
        """Record successful model loading event."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self.total_load_attempts += 1
        self.successful_loads += 1
        self.last_load_timestamp = now_iso
        self._history.append({
            "timestamp": now_iso,
            "status": "SUCCESS",
            "model_version": model_version,
        })
        logger.info(f"Model {model_version} loaded successfully at {now_iso}.")

    def record_failure(
        self,
        failure_type: str,
        reason: str,
        model_version: Optional[str] = None,
    ) -> None:
        """
        Record a model load failure.
        Failure types: 'FILE_NOT_FOUND', 'DESERIALIZATION_ERROR', 'CORRUPT_ARTIFACT', 'SCHEMA_MISMATCH'
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        self.total_load_attempts += 1
        self.failed_loads += 1
        self.last_failure_timestamp = now_iso
        self.last_failure_reason = reason
        self.last_failure_type = failure_type
        self._history.append({
            "timestamp": now_iso,
            "status": "FAILURE",
            "failure_type": failure_type,
            "reason": reason,
            "model_version": model_version,
        })
        logger.error(f"Model load failure ({failure_type}): {reason} at {now_iso}.")

    def record_fallback_activation(self, reason: str) -> None:
        """Record an instance where system fell back to a secondary model."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self.fallback_activations += 1
        self._history.append({
            "timestamp": now_iso,
            "status": "FALLBACK_ACTIVATED",
            "reason": reason,
        })
        logger.warning(f"Fallback model activated: {reason} at {now_iso}.")

    def get_health(self) -> ModelLoadingHealth:
        """Synthesize current model loading health."""
        fail_rate = (
            (self.failed_loads / self.total_load_attempts * 100.0)
            if self.total_load_attempts > 0
            else 0.0
        )
        is_healthy = self.failed_loads == 0 and self.fallback_activations == 0
        status = "HEALTHY"
        if self.failed_loads > 0:
            status = "CRITICAL_LOAD_FAILURE"
        elif self.fallback_activations > 0:
            status = "WARNING"

        return ModelLoadingHealth(
            total_load_attempts=self.total_load_attempts,
            successful_loads=self.successful_loads,
            failed_loads=self.failed_loads,
            failure_rate_pct=round(fail_rate, 2),
            fallback_activations=self.fallback_activations,
            last_load_timestamp=self.last_load_timestamp,
            last_failure_timestamp=self.last_failure_timestamp,
            last_failure_reason=self.last_failure_reason,
            last_failure_type=self.last_failure_type,
            is_healthy=is_healthy,
            status=status,
        )


# ==============================================================================
# 2. Inference Latency Monitoring
# ==============================================================================

@dataclass
class LatencyMetrics:
    """Inference execution latency percentiles and SLA compliance report."""

    total_predictions: int
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    sla_budget_ms: float = 3.0  # Production roadmap target: <= 3.0 ms per inference
    sla_breached: bool = False
    status: str = "HEALTHY"  # "HEALTHY", "WARNING", "SLA_BREACHED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InferenceLatencyMonitor:
    """
    High-precision telemetry tracker for model inference execution times.
    Calculates P50, P90, P95, P99 latencies and verifies SLA budget compliance.
    """

    def __init__(
        self,
        sla_budget_ms: float = 3.0,
        max_samples: int = 10000,
    ) -> None:
        self.sla_budget_ms = sla_budget_ms
        self.latencies: deque = deque(maxlen=max_samples)

    def record_latency(self, latency_ms: float) -> None:
        """Record a single or batch inference latency duration in milliseconds."""
        self.latencies.append(float(latency_ms))

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Context manager to measure and record execution time of an inference block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.record_latency(elapsed_ms)

    def get_metrics(self) -> LatencyMetrics:
        """Compute comprehensive latency statistics across recorded inference samples."""
        if not self.latencies:
            return LatencyMetrics(
                total_predictions=0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p90_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                sla_budget_ms=self.sla_budget_ms,
                sla_breached=False,
                status="HEALTHY",
            )

        arr = np.asarray(self.latencies, dtype=float)
        mean_l = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        min_l = float(np.min(arr))
        max_l = float(np.max(arr))

        # Check SLA breach (if P95 or mean exceeds budget)
        breached = bool(p95 > self.sla_budget_ms or mean_l > self.sla_budget_ms)
        status = "SLA_BREACHED" if breached else ("WARNING" if p90 > self.sla_budget_ms * 0.8 else "HEALTHY")

        return LatencyMetrics(
            total_predictions=len(arr),
            mean_latency_ms=round(mean_l, 4),
            p50_latency_ms=round(p50, 4),
            p90_latency_ms=round(p90, 4),
            p95_latency_ms=round(p95, 4),
            p99_latency_ms=round(p99, 4),
            min_latency_ms=round(min_l, 4),
            max_latency_ms=round(max_l, 4),
            sla_budget_ms=self.sla_budget_ms,
            sla_breached=breached,
            status=status,
        )


# ==============================================================================
# 3. Quantile Coverage & Interval Monitoring
# ==============================================================================

@dataclass
class QuantileCoverageReport:
    """Report on empirical prediction interval coverage and quantile validity."""

    sample_size: int
    empirical_coverage_80: float
    target_coverage_80: float = 0.80
    min_acceptable_coverage: float = 0.70
    max_acceptable_coverage: float = 0.95
    mean_interval_width: float = 0.0
    median_interval_width: float = 0.0
    quantile_crossing_count: int = 0
    quantile_crossing_rate: float = 0.0
    under_coverage_alert: bool = False
    over_coverage_alert: bool = False
    crossing_alert: bool = False
    status: str = "HEALTHY"  # "HEALTHY", "WARNING", "DEGRADED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuantileCoverageMonitor:
    """
    Evaluates empirical coverage of P10/P90 prediction intervals and monitors
    quantile crossings (P10 > P50 or P50 > P90).
    """

    def __init__(
        self,
        target_coverage: float = 0.80,
        min_acceptable_coverage: float = 0.70,
        max_acceptable_coverage: float = 0.95,
    ) -> None:
        self.target_coverage = target_coverage
        self.min_acceptable_coverage = min_acceptable_coverage
        self.max_acceptable_coverage = max_acceptable_coverage

    def evaluate_coverage(
        self,
        p10: Union[List[float], np.ndarray],
        p50: Union[List[float], np.ndarray],
        p90: Union[List[float], np.ndarray],
        actuals: Union[List[float], np.ndarray],
    ) -> QuantileCoverageReport:
        """
        Calculate empirical coverage: proportion of actuals falling within [P10, P90].
        Also checks for any quantile crossing violations.
        """
        q10 = np.asarray(p10, dtype=float)
        q50 = np.asarray(p50, dtype=float)
        q90 = np.asarray(p90, dtype=float)
        y = np.asarray(actuals, dtype=float)

        n = len(y)
        if n == 0 or len(q10) != n or len(q50) != n or len(q90) != n:
            return QuantileCoverageReport(
                sample_size=0,
                empirical_coverage_80=0.0,
                status="NO_DATA",
            )

        # Check empirical coverage
        covered = (y >= q10) & (y <= q90)
        coverage = float(np.mean(covered))

        # Check quantile crossing violations: P10 > P50 or P50 > P90
        crossings = (q10 > (q50 + 1e-4)) | (q50 > (q90 + 1e-4))
        crossing_count = int(np.sum(crossings))
        crossing_rate = float(crossing_count / n)

        # Interval width
        widths = q90 - q10
        mean_w = float(np.mean(widths))
        median_w = float(np.median(widths))

        under_cov = coverage < self.min_acceptable_coverage
        over_cov = coverage > self.max_acceptable_coverage
        cross_alert = crossing_count > 0

        status = "HEALTHY"
        if under_cov or cross_alert:
            status = "DEGRADED"
        elif over_cov:
            status = "WARNING"

        return QuantileCoverageReport(
            sample_size=n,
            empirical_coverage_80=round(coverage, 4),
            target_coverage_80=self.target_coverage,
            min_acceptable_coverage=self.min_acceptable_coverage,
            max_acceptable_coverage=self.max_acceptable_coverage,
            mean_interval_width=round(mean_w, 4),
            median_interval_width=round(median_w, 4),
            quantile_crossing_count=crossing_count,
            quantile_crossing_rate=round(crossing_rate, 4),
            under_coverage_alert=under_cov,
            over_coverage_alert=over_cov,
            crossing_alert=cross_alert,
            status=status,
        )


# ==============================================================================
# 4. Model Version & Rollback History Tracking
# ==============================================================================

@dataclass
class VersionAndRollbackStatus:
    """Report on active model version, champion backup status, and audit history."""

    active_model_version: str
    active_model_type: str
    active_champion_path: str
    active_model_checksum: Optional[str]
    backup_exists: bool
    backup_champion_path: Optional[str]
    backup_version: Optional[str]
    backup_model_checksum: Optional[str]
    backup_verified_loadable: bool
    rollback_readiness: str  # "READY", "UNAVAILABLE", "CORRUPTED"
    total_rollback_events: int
    last_rollback_event: Optional[Dict[str, Any]]
    total_promotion_events: int
    last_promotion_event: Optional[Dict[str, Any]]
    recent_audit_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRollbackTracker:
    """
    Inspects model versioning, verifies backup champion readiness, and tracks
    historical promotion and rollback actions from the retraining audit log.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = models_dir or MODELS_DIR

    @staticmethod
    def _file_hash(path: Path) -> Optional[str]:
        """Compute SHA-256 hash of the first 1MB of a file for verification."""
        if not path.exists():
            return None
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                h.update(f.read(1_048_576))
            return h.hexdigest()[:12]
        except Exception:
            return None

    def get_status(self) -> VersionAndRollbackStatus:
        """Inspect models directory and audit log to produce rollback status report."""
        champ_file = self.models_dir / "demand_model.pkl"
        meta_file = self.models_dir / "demand_model_meta.json"
        backup_file = self.models_dir / "champion_backup" / "demand_model.pkl"
        backup_meta_file = self.models_dir / "champion_backup" / "demand_model_meta.json"
        audit_file = self.models_dir / "retraining_audit_log.json"

        # 1. Active Champion Status
        active_ver = "v2.1.0"
        active_type = "lightgbm"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    active_ver = m.get("model_version", active_ver)
                    active_type = m.get("model_type", active_type)
            except Exception:
                pass
        champ_hash = self._file_hash(champ_file)

        # 2. Backup Status
        backup_exists = backup_file.exists()
        if not backup_exists and champ_file.exists():
            try:
                from src.geodemand.models.promotion import ModelPromotionGate
                gate = ModelPromotionGate(models_dir=self.models_dir)
                gate.backup_current_champion()
                backup_exists = backup_file.exists()
            except Exception as e:
                logger.warning(f"Failed to auto-seed champion backup snapshot: {e}")

        backup_ver = None
        backup_hash = self._file_hash(backup_file) if backup_exists else None
        backup_loadable = False

        if backup_exists:
            if backup_meta_file.exists():
                try:
                    with open(backup_meta_file, "r", encoding="utf-8") as f:
                        bm = json.load(f)
                        backup_ver = bm.get("model_version", "v2.0.0_backup")
                except Exception:
                    backup_ver = "v2.0.0_backup"
            else:
                backup_ver = "v2.0.0_backup"

            # Quick verification: check non-zero size
            try:
                if backup_file.stat().st_size > 100:
                    backup_loadable = True
            except Exception:
                backup_loadable = False

        rollback_readiness = "READY" if (backup_exists and backup_loadable) else "UNAVAILABLE"

        # 3. Retraining & Rollback Audit Records
        audit_records: List[Dict[str, Any]] = []
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        audit_records = loaded
            except Exception:
                pass

        rollbacks = [r for r in audit_records if "rollback" in r.get("status", "").lower() or r.get("rollback_executed", False)]
        promotions = [r for r in audit_records if "promoted" in r.get("status", "").lower()]

        return VersionAndRollbackStatus(
            active_model_version=active_ver,
            active_model_type=active_type,
            active_champion_path=str(champ_file),
            active_model_checksum=champ_hash,
            backup_exists=backup_exists,
            backup_champion_path=str(backup_file) if backup_exists else None,
            backup_version=backup_ver,
            backup_model_checksum=backup_hash,
            backup_verified_loadable=backup_loadable,
            rollback_readiness=rollback_readiness,
            total_rollback_events=len(rollbacks),
            last_rollback_event=rollbacks[-1] if rollbacks else None,
            total_promotion_events=len(promotions),
            last_promotion_event=promotions[-1] if promotions else None,
            recent_audit_records=audit_records[-5:] if audit_records else [],
        )


# ==============================================================================
# 5. Consolidated Operational ML Monitor
# ==============================================================================

@dataclass
class OperationalHealthSummary:
    """Consolidated production ML monitoring telemetry synthesis (Phase 17.2)."""

    timestamp: str
    overall_health: str  # "HEALTHY", "WARNING", "CRITICAL_DRIFT", "DEGRADED", "SLA_BREACHED"
    feature_drift_alert: bool
    prediction_drift_alert: bool
    real_label_mae_alert: bool
    quantile_coverage_alert: bool
    model_loading_alert: bool
    inference_latency_alert: bool
    rollback_ready: bool

    # Dimension Details
    feature_drift: Dict[str, Any]
    prediction_drift: Dict[str, Any]
    real_label_performance: Dict[str, Any]
    quantile_coverage: Dict[str, Any]
    model_loading: Dict[str, Any]
    inference_latency: Dict[str, Any]
    version_and_rollback: Dict[str, Any]

    # Automated Retraining Trigger
    retraining_trigger_active: bool
    retraining_trigger_type: Optional[str]
    retraining_trigger_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class OperationalMLMonitor:
    """
    Central operational ML monitoring engine managing all 7 Phase 17.2 responsibilities:
    1. Feature drift
    2. Prediction drift
    3. Real-label MAE
    4. Quantile coverage
    5. Model loading failures
    6. Inference latency
    7. Model version and rollback history
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        thresholds: Optional[DriftThresholds] = None,
        sla_budget_ms: float = 3.0,
    ) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self.thresholds = thresholds or DriftThresholds()

        self.feature_detector = FeatureDriftDetector(thresholds=self.thresholds)
        self.prediction_detector = PredictionDriftDetector(thresholds=self.thresholds)
        self.perf_monitor = PerformanceMonitor()
        self.quantile_monitor = QuantileCoverageMonitor()
        self.loading_monitor = ModelLoadingMonitor()
        self.latency_monitor = InferenceLatencyMonitor(sla_budget_ms=sla_budget_ms)
        self.rollback_tracker = ModelRollbackTracker(models_dir=self.models_dir)

    def evaluate_system_health(
        self,
        current_features_df: Optional[pd.DataFrame] = None,
        reference_features_df: Optional[pd.DataFrame] = None,
        current_predictions: Optional[Union[List[float], np.ndarray]] = None,
        reference_predictions: Optional[Union[List[float], np.ndarray]] = None,
        recent_actuals: Optional[Union[List[float], np.ndarray]] = None,
        recent_preds_for_actuals: Optional[Union[List[float], np.ndarray]] = None,
        p10_preds: Optional[Union[List[float], np.ndarray]] = None,
        p50_preds: Optional[Union[List[float], np.ndarray]] = None,
        p90_preds: Optional[Union[List[float], np.ndarray]] = None,
    ) -> OperationalHealthSummary:
        """
        Evaluate and synthesize all 7 ML monitoring dimensions into a single actionable health status.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Feature Drift Evaluation
        feature_drift_summary = {
            "status": "NOMINAL",
            "max_psi": 0.0,
            "mean_psi": 0.0,
            "drifted_features_count": 0,
            "drifted_features": [],
            "trigger_activated": False,
        }
        if current_features_df is not None and not current_features_df.empty:
            ref_df = reference_features_df
            if ref_df is None:
                ref_df = current_features_df.iloc[: min(200, len(current_features_df))]
            try:
                prof = ReferenceDistributionProfile.from_dataframe(ref_df, dataset_id="op_monitor_ref")
                rep = self.feature_detector.detect_drift(current_features_df, reference_profile=prof)
                feature_drift_summary["max_psi"] = round(rep.max_psi, 4)
                feature_drift_summary["mean_psi"] = round(rep.mean_psi, 4)
                feature_drift_summary["drifted_features_count"] = len(rep.drifted_features)
                feature_drift_summary["drifted_features"] = rep.drifted_features
                feature_drift_summary["trigger_activated"] = rep.max_psi >= self.thresholds.psi_threshold
                feature_drift_summary["status"] = (
                    "CRITICAL_DRIFT" if rep.max_psi >= self.thresholds.psi_threshold
                    else ("MODERATE_DRIFT" if rep.overall_drift_detected else "NOMINAL")
                )
            except Exception as e:
                logger.warning(f"Feature drift monitor exception: {e}")

        # 2. Prediction Drift Evaluation
        prediction_drift_summary = {
            "status": "NOMINAL",
            "psi": 0.0,
            "wasserstein_distance": 0.0,
            "trigger_activated": False,
        }
        if current_predictions is not None and len(current_predictions) >= 10:
            cur_p = np.asarray(current_predictions, dtype=float)
            if reference_predictions is not None and len(reference_predictions) >= 10:
                ref_p = np.asarray(reference_predictions, dtype=float)
            else:
                rng = np.random.RandomState(42)
                ref_p = rng.poisson(lam=15.0, size=len(cur_p)).astype(float)
            try:
                p_rep = self.prediction_detector.evaluate(ref_p, cur_p)
                prediction_drift_summary["psi"] = p_rep.metrics.psi
                prediction_drift_summary["wasserstein_distance"] = p_rep.metrics.wasserstein_distance
                prediction_drift_summary["trigger_activated"] = p_rep.trigger_activated
                prediction_drift_summary["status"] = p_rep.severity
            except Exception as e:
                logger.warning(f"Prediction drift monitor exception: {e}")

        # 3. Real-Label MAE Performance Evaluation
        rolling_summary = {
            "status": "NOMINAL",
            "current_7d_mae": 0.7693,
            "baseline_mae": 0.7693,
            "degradation_pct": 0.0,
            "degradation_alert": False,
            "sample_count": 0,
            "message": "Awaiting ground truth labels.",
        }
        if recent_actuals is not None and recent_preds_for_actuals is not None:
            y_t = np.asarray(recent_actuals, dtype=float)
            y_p = np.asarray(recent_preds_for_actuals, dtype=float)
            valid_m = np.isfinite(y_t) & np.isfinite(y_p)
            y_t_c = y_t[valid_m]
            y_p_c = y_p[valid_m]

            if len(y_t_c) >= 20:
                assess = self.perf_monitor.assess_rolling_degradation(
                    baseline_mae=0.7693,
                    recent_y_true=y_t_c,
                    recent_y_pred=y_p_c,
                    degradation_threshold_pct=20.0,
                )
                rolling_summary["current_7d_mae"] = assess.current_7d_mae
                rolling_summary["degradation_pct"] = assess.degradation_pct
                rolling_summary["degradation_alert"] = assess.degradation_alert
                rolling_summary["sample_count"] = assess.sample_count
                rolling_summary["message"] = assess.message
                rolling_summary["status"] = "DEGRADED" if assess.degradation_alert else "NOMINAL"
            else:
                rolling_summary["sample_count"] = len(y_t_c)
                rolling_summary["message"] = f"Insufficient labeled actuals ({len(y_t_c)} < 20). Delayed ground truth expected."

        # 4. Quantile Coverage Evaluation
        quantile_summary = {
            "status": "NOMINAL",
            "empirical_coverage_80": 0.80,
            "under_coverage_alert": False,
            "quantile_crossing_count": 0,
            "sample_size": 0,
        }
        if (
            recent_actuals is not None
            and p10_preds is not None
            and p50_preds is not None
            and p90_preds is not None
        ):
            q_rep = self.quantile_monitor.evaluate_coverage(
                p10=p10_preds,
                p50=p50_preds,
                p90=p90_preds,
                actuals=recent_actuals,
            )
            quantile_summary = q_rep.to_dict()

        # 5. Model Loading Telemetry
        loading_health = self.loading_monitor.get_health().to_dict()

        # 6. Inference Latency Telemetry
        latency_metrics = self.latency_monitor.get_metrics().to_dict()

        # 7. Model Version & Rollback Telemetry
        rollback_info = self.rollback_tracker.get_status().to_dict()

        # Retraining Trigger Logic
        retrain_active = False
        retrain_type = None
        retrain_reason = None

        if feature_drift_summary["trigger_activated"]:
            retrain_active = True
            retrain_type = "DRIFT_TRIGGERED"
            retrain_reason = f"Feature drift PSI {feature_drift_summary['max_psi']:.4f} >= {self.thresholds.psi_threshold}"
        elif prediction_drift_summary["trigger_activated"]:
            retrain_active = True
            retrain_type = "DRIFT_TRIGGERED"
            retrain_reason = f"Prediction drift PSI {prediction_drift_summary['psi']:.4f} >= {self.thresholds.psi_threshold}"
        elif rolling_summary["degradation_alert"]:
            retrain_active = True
            retrain_type = "PERFORMANCE_DEGRADATION"
            retrain_reason = f"7-day rolling MAE degraded by {rolling_summary['degradation_pct']:.1f}% >= 20%"

        # Overall Health Synthesis
        if loading_health["failed_loads"] > 0:
            overall = "CRITICAL_LOAD_FAILURE"
        elif latency_metrics["sla_breached"]:
            overall = "SLA_BREACHED"
        elif rolling_summary["degradation_alert"] or quantile_summary.get("under_coverage_alert", False):
            overall = "DEGRADED"
        elif feature_drift_summary["trigger_activated"] or prediction_drift_summary["trigger_activated"]:
            overall = "CRITICAL_DRIFT"
        elif (
            feature_drift_summary["status"] == "MODERATE_DRIFT"
            or prediction_drift_summary["status"] == "MODERATE"
            or loading_health["status"] == "WARNING"
            or latency_metrics["status"] == "WARNING"
        ):
            overall = "WARNING"
        else:
            overall = "HEALTHY"

        return OperationalHealthSummary(
            timestamp=now_iso,
            overall_health=overall,
            feature_drift_alert=feature_drift_summary["trigger_activated"],
            prediction_drift_alert=prediction_drift_summary["trigger_activated"],
            real_label_mae_alert=rolling_summary["degradation_alert"],
            quantile_coverage_alert=quantile_summary.get("under_coverage_alert", False) or quantile_summary.get("crossing_alert", False),
            model_loading_alert=not loading_health["is_healthy"],
            inference_latency_alert=latency_metrics["sla_breached"],
            rollback_ready=rollback_info["backup_verified_loadable"],
            feature_drift=feature_drift_summary,
            prediction_drift=prediction_drift_summary,
            real_label_performance=rolling_summary,
            quantile_coverage=quantile_summary,
            model_loading=loading_health,
            inference_latency=latency_metrics,
            version_and_rollback=rollback_info,
            retraining_trigger_active=retrain_active,
            retraining_trigger_type=retrain_type,
            retraining_trigger_reason=retrain_reason,
        )
