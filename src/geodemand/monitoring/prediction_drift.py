"""
GeoDemand AI — Prediction Drift & Output Distribution Monitoring Module

Member 2 — Machine Learning & MLOps

Implements:
- Prediction distribution drift detection between reference and current serving batches
- Population Stability Index (PSI) on predicted customer counts
- Wasserstein Distance (Earth Mover's Distance) on output predictions
- Two-sample Kolmogorov-Smirnov (KS) test on prediction distributions
- Quantile shift analysis (P10, P50, P90 shifts)
- Out-of-bounds and negative prediction checks
- Roadmap drift trigger integration (PSI >= 0.15)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import stats

from src.geodemand.monitoring.drift import DriftThresholds

logger = logging.getLogger(__name__)


@dataclass
class PredictionDriftResult:
    """Individual prediction drift evaluation metrics."""

    psi: float
    wasserstein_distance: float
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    severity: str  # "NOMINAL", "MODERATE", "CRITICAL"
    mean_reference: float
    mean_current: float
    mean_shift: float
    p10_reference: float
    p10_current: float
    p50_reference: float
    p50_current: float
    p90_reference: float
    p90_current: float
    n_reference: int
    n_current: int
    negative_predictions_count: int
    zero_predictions_rate: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionDriftReport:
    """Complete prediction drift monitoring report."""

    timestamp: str
    drift_detected: bool
    trigger_activated: bool  # True if PSI >= 0.15 trigger activated
    severity: str
    metrics: PredictionDriftResult
    thresholds: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift_detected": self.drift_detected,
            "trigger_activated": self.trigger_activated,
            "severity": self.severity,
            "metrics": self.metrics.to_dict(),
            "thresholds": self.thresholds,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        m = self.metrics
        status = "CRITICAL PREDICTION DRIFT" if self.trigger_activated else ("MODERATE DRIFT" if self.drift_detected else "NOMINAL")
        lines = [
            f"=== GeoDemand Prediction Drift Report [{status}] ===",
            f"Timestamp: {self.timestamp}",
            f"Severity: {self.severity}",
            f"Retraining Trigger Active: {self.trigger_activated} (Threshold: PSI >= {self.thresholds.get('psi_threshold', 0.15)})",
            f"Prediction PSI: {m.psi:.4f}",
            f"Wasserstein Distance: {m.wasserstein_distance:.4f}",
            f"KS Statistic: {m.ks_statistic:.4f} (p-value: {m.ks_p_value:.4e})",
            f"Mean Pred Shift: {m.mean_reference:.2f} -> {m.mean_current:.2f} (diff: {m.mean_shift:+.2f})",
            f"P10 Shift: {m.p10_reference:.2f} -> {m.p10_current:.2f}",
            f"P50 Shift: {m.p50_reference:.2f} -> {m.p50_current:.2f}",
            f"P90 Shift: {m.p90_reference:.2f} -> {m.p90_current:.2f}",
            f"Sample Counts: N_ref={m.n_reference:,}, N_cur={m.n_current:,}",
            f"Zero Prediction Rate: {m.zero_predictions_rate:.2%}",
            f"Message: {m.message}",
        ]
        return "\n".join(lines)


class PredictionDriftDetector:
    """
    Monitors distribution shifts in model predictions ($y_{pred}$) between reference
    (e.g., validation/baseline period) and live inference streams.
    """

    def __init__(
        self,
        thresholds: Optional[DriftThresholds] = None,
        num_bins: int = 10,
        epsilon: float = 1e-4,
    ):
        self.thresholds = thresholds or DriftThresholds()
        self.num_bins = num_bins
        self.epsilon = epsilon

    def calculate_psi(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray,
    ) -> float:
        """
        Calculate Population Stability Index (PSI) using reference decile bin edges.
        """
        ref_clean = reference_preds[np.isfinite(reference_preds)]
        cur_clean = current_preds[np.isfinite(current_preds)]
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return 0.0

        # Bin edges from reference deciles
        deciles = np.percentile(ref_clean, np.linspace(0, 100, self.num_bins + 1))
        edges = sorted(list(set(deciles)))
        if len(edges) < 2:
            edges = [deciles[0] - 1.0, deciles[0] + 1.0]

        edges = np.array(edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf

        k_bins = len(edges) - 1
        ref_counts, _ = np.histogram(ref_clean, bins=edges)
        cur_counts, _ = np.histogram(cur_clean, bins=edges)

        ref_props = (ref_counts + self.epsilon) / (np.sum(ref_counts) + k_bins * self.epsilon)
        cur_props = (cur_counts + self.epsilon) / (np.sum(cur_counts) + k_bins * self.epsilon)

        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return float(np.maximum(0.0, psi))

    def calculate_wasserstein(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray,
    ) -> float:
        """
        Calculate 1D Wasserstein Distance (Earth Mover's Distance) between predictions.
        """
        ref_clean = reference_preds[np.isfinite(reference_preds)]
        cur_clean = current_preds[np.isfinite(current_preds)]
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return 0.0
        try:
            return float(stats.wasserstein_distance(ref_clean, cur_clean))
        except Exception:
            return 0.0

    def evaluate(
        self,
        reference_predictions: Union[List[float], np.ndarray],
        current_predictions: Union[List[float], np.ndarray],
        timestamp: Optional[str] = None,
    ) -> PredictionDriftReport:
        """
        Evaluate prediction drift between reference and current predictions.
        """
        now_iso = timestamp or datetime.now(timezone.utc).isoformat()
        ref_arr = np.asarray(reference_predictions, dtype=float)
        cur_arr = np.asarray(current_predictions, dtype=float)

        ref_clean = ref_arr[np.isfinite(ref_arr)]
        cur_clean = cur_arr[np.isfinite(cur_arr)]

        n_ref = len(ref_clean)
        n_cur = len(cur_clean)

        if n_ref == 0 or n_cur == 0:
            result = PredictionDriftResult(
                psi=0.0,
                wasserstein_distance=0.0,
                ks_statistic=0.0,
                ks_p_value=1.0,
                drift_detected=False,
                severity="NOMINAL",
                mean_reference=0.0,
                mean_current=0.0,
                mean_shift=0.0,
                p10_reference=0.0,
                p10_current=0.0,
                p50_reference=0.0,
                p50_current=0.0,
                p90_reference=0.0,
                p90_current=0.0,
                n_reference=n_ref,
                n_current=n_cur,
                negative_predictions_count=0,
                zero_predictions_rate=0.0,
                message="Insufficient valid prediction samples to evaluate drift.",
            )
            return PredictionDriftReport(
                timestamp=now_iso,
                drift_detected=False,
                trigger_activated=False,
                severity="NOMINAL",
                metrics=result,
                thresholds={"psi_threshold": self.thresholds.psi_threshold},
            )

        # 1. PSI Calculation
        psi_val = self.calculate_psi(ref_clean, cur_clean)

        # 2. Wasserstein Distance
        wd_val = self.calculate_wasserstein(ref_clean, cur_clean)

        # 3. Two-sample Kolmogorov-Smirnov test
        ks_res = stats.ks_2samp(ref_clean, cur_clean)
        ks_stat = float(ks_res.statistic)
        ks_p = float(ks_res.pvalue)

        # 4. Summary statistics & quantiles
        mean_ref = float(np.mean(ref_clean))
        mean_cur = float(np.mean(cur_clean))
        mean_shift = mean_cur - mean_ref

        p10_ref, p50_ref, p90_ref = [float(x) for x in np.percentile(ref_clean, [10, 50, 90])]
        p10_cur, p50_cur, p90_cur = [float(x) for x in np.percentile(cur_clean, [10, 50, 90])]

        neg_count = int(np.sum(cur_arr < 0))
        zero_rate = float(np.mean(cur_arr == 0)) if len(cur_arr) > 0 else 0.0

        # 5. Drift Decision & Roadmap Trigger (PSI >= 0.15)
        trigger_active = psi_val >= self.thresholds.psi_threshold
        moderate_drift = self.thresholds.psi_moderate_threshold <= psi_val < self.thresholds.psi_threshold
        ks_drift = (ks_p < self.thresholds.ks_alpha) and (ks_stat >= self.thresholds.ks_stat_threshold)

        drift_detected = trigger_active or moderate_drift or ks_drift

        if trigger_active:
            severity = "CRITICAL"
            msg = f"Critical prediction drift detected: PSI {psi_val:.4f} >= {self.thresholds.psi_threshold}. Retraining trigger ACTIVATED."
        elif moderate_drift:
            severity = "MODERATE"
            msg = f"Moderate prediction drift detected: PSI {psi_val:.4f} in [{self.thresholds.psi_moderate_threshold}, {self.thresholds.psi_threshold}). Monitor closely."
        elif ks_drift:
            severity = "MODERATE"
            msg = f"Statistically significant prediction shift detected by KS test (stat={ks_stat:.4f}, p={ks_p:.4e}, PSI={psi_val:.4f})."
        else:
            severity = "NOMINAL"
            msg = f"Prediction distribution nominal (PSI={psi_val:.4f} < {self.thresholds.psi_moderate_threshold}, Wasserstein={wd_val:.4f})."

        result = PredictionDriftResult(
            psi=round(psi_val, 4),
            wasserstein_distance=round(wd_val, 4),
            ks_statistic=round(ks_stat, 4),
            ks_p_value=round(ks_p, 6),
            drift_detected=drift_detected,
            severity=severity,
            mean_reference=round(mean_ref, 2),
            mean_current=round(mean_cur, 2),
            mean_shift=round(mean_shift, 2),
            p10_reference=round(p10_ref, 2),
            p10_current=round(p10_cur, 2),
            p50_reference=round(p50_ref, 2),
            p50_current=round(p50_cur, 2),
            p90_reference=round(p90_ref, 2),
            p90_current=round(p90_cur, 2),
            n_reference=n_ref,
            n_current=n_cur,
            negative_predictions_count=neg_count,
            zero_predictions_rate=round(zero_rate, 4),
            message=msg,
        )

        return PredictionDriftReport(
            timestamp=now_iso,
            drift_detected=drift_detected,
            trigger_activated=trigger_active,
            severity=severity,
            metrics=result,
            thresholds={
                "psi_threshold": self.thresholds.psi_threshold,
                "psi_moderate_threshold": self.thresholds.psi_moderate_threshold,
                "ks_alpha": self.thresholds.ks_alpha,
                "ks_stat_threshold": self.thresholds.ks_stat_threshold,
            },
        )
