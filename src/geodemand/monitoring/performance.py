"""
GeoDemand AI — Delayed Performance Monitoring & Ground-Truth Evaluation Module

Member 3 — Machine Learning & MLOps

Implements:
- Delayed ground-truth accuracy tracking: MAE, RMSE, R², Prediction Bias
- Full error distribution profiling (mean, std, median, min, max, percentiles)
- Uncertainty interval validation: empirical quantile coverage and interval width
- Slice breakdowns: hour of day, weekday vs weekend, time period, vendor category
- Machine-readable performance reports with simulation disclaimer
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIMULATION_DATA_DISCLAIMER = (
    "SIMULATED DATA NOTICE: The GeoDemand dataset contains simulated targets. "
    "Performance metrics evaluated on simulated targets reflect validation against the "
    "synthetic generating process and do not establish real-world accuracy."
)


@dataclass
class SlicePerformance:
    """Performance metrics for a specific cohort or temporal slice."""

    slice_dimension: str
    slice_value: str
    sample_size: int
    mae: float
    rmse: float
    r2: Optional[float]
    prediction_bias: float
    quantile_coverage_80: Optional[float] = None
    mean_interval_width: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorDistribution:
    """Detailed summary of the prediction error distribution (y_pred - y_true)."""

    mean: float
    std: float
    median: float
    min: float
    max: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceReport:
    """Comprehensive performance monitoring report on ground-truth observations."""

    timestamp: str
    sample_size: int
    mae: float
    rmse: float
    r2: Optional[float]
    prediction_bias: float  # Mean Signed Error
    error_distribution: ErrorDistribution
    quantile_coverage_80: Optional[float]
    mean_interval_width: Optional[float]
    slices: Dict[str, List[SlicePerformance]] = field(default_factory=dict)
    data_disclaimer: str = SIMULATION_DATA_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sample_size": self.sample_size,
            "mae": round(float(self.mae), 4),
            "rmse": round(float(self.rmse), 4),
            "r2": round(float(self.r2), 4) if self.r2 is not None else None,
            "prediction_bias": round(float(self.prediction_bias), 4),
            "error_distribution": self.error_distribution.to_dict(),
            "quantile_coverage_80": round(float(self.quantile_coverage_80), 4) if self.quantile_coverage_80 is not None else None,
            "mean_interval_width": round(float(self.mean_interval_width), 4) if self.mean_interval_width is not None else None,
            "slices": {k: [s.to_dict() for s in v] for k, v in self.slices.items()},
            "data_disclaimer": self.data_disclaimer,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [
            "=== GeoDemand Performance Monitoring Report ===",
            f"Timestamp: {self.timestamp}",
            f"Evaluated Observations: {self.sample_size:,}",
            f"MAE: {self.mae:.4f} customers/hour",
            f"RMSE: {self.rmse:.4f} customers/hour",
            f"R²: {self.r2:.4f}" if self.r2 is not None else "R²: N/A",
            f"Prediction Bias: {self.prediction_bias:+.4f} (Mean Signed Error)",
        ]
        if self.quantile_coverage_80 is not None:
            lines.append(f"80% Interval Coverage: {self.quantile_coverage_80:.1%} (nominal 80%)")
        if self.mean_interval_width is not None:
            lines.append(f"Mean Interval Width (P90-P10): {self.mean_interval_width:.2f}")
        lines.append(f"Disclaimer: {self.data_disclaimer}")
        return "\n".join(lines)


class PerformanceMonitor:
    """
    Evaluates predictions after actual ground-truth customer counts become available.
    """

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    @staticmethod
    def _compute_core_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        p10: Optional[np.ndarray] = None,
        p90: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute core statistical and error metrics."""
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))
        bias = float(np.mean(errors))

        # R² with safe handling of zero-variance actual targets
        var_true = float(np.var(y_true))
        if var_true > 1e-9:
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_res = np.sum(sq_errors)
            r2 = float(1.0 - (ss_res / ss_tot))
        else:
            r2 = None

        err_dist = ErrorDistribution(
            mean=float(np.mean(errors)),
            std=float(np.std(errors)),
            median=float(np.median(errors)),
            min=float(np.min(errors)),
            max=float(np.max(errors)),
            p10=float(np.percentile(errors, 10)),
            p25=float(np.percentile(errors, 25)),
            p75=float(np.percentile(errors, 75)),
            p90=float(np.percentile(errors, 90)),
            p95=float(np.percentile(errors, 95)),
        )

        coverage_80 = None
        mean_width = None
        if p10 is not None and p90 is not None:
            in_interval = (y_true >= (p10 - 1e-4)) & (y_true <= (p90 + 1e-4))
            coverage_80 = float(np.mean(in_interval))
            mean_width = float(np.mean(p90 - p10))

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "bias": bias,
            "error_distribution": err_dist,
            "coverage_80": coverage_80,
            "mean_width": mean_width,
        }

    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        y_pred: Union[np.ndarray, pd.Series, List[float]],
        p10: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
        p90: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> PerformanceReport:
        """
        Compute performance report comparing predictions against actuals, with slice breakdowns.
        """
        arr_true = np.asarray(y_true, dtype=float)
        arr_pred = np.asarray(y_pred, dtype=float)

        if len(arr_true) != len(arr_pred):
            raise ValueError(f"Length mismatch: y_true ({len(arr_true)}) vs y_pred ({len(arr_pred)})")
        if len(arr_true) == 0:
            raise ValueError("Cannot evaluate performance on empty dataset.")

        arr_p10 = np.asarray(p10, dtype=float) if p10 is not None else None
        arr_p90 = np.asarray(p90, dtype=float) if p90 is not None else None

        core = self._compute_core_metrics(arr_true, arr_pred, arr_p10, arr_p90)
        now_iso = datetime.now(timezone.utc).isoformat()

        slices_result: Dict[str, List[SlicePerformance]] = {}

        if metadata_df is not None and len(metadata_df) == len(arr_true):
            slice_candidates = ["hour", "is_weekend", "weekday", "vendor_category", "date"]
            for col in slice_candidates:
                if col in metadata_df.columns:
                    col_slices: List[SlicePerformance] = []
                    groups = metadata_df.groupby(col)
                    for val, group_indices in groups.groups.items():
                        idx = group_indices.to_numpy()
                        if len(idx) < 2:
                            continue

                        sub_true = arr_true[idx]
                        sub_pred = arr_pred[idx]
                        sub_p10 = arr_p10[idx] if arr_p10 is not None else None
                        sub_p90 = arr_p90[idx] if arr_p90 is not None else None

                        sub_core = self._compute_core_metrics(sub_true, sub_pred, sub_p10, sub_p90)
                        col_slices.append(
                            SlicePerformance(
                                slice_dimension=col,
                                slice_value=str(val),
                                sample_size=len(idx),
                                mae=round(sub_core["mae"], 4),
                                rmse=round(sub_core["rmse"], 4),
                                r2=round(sub_core["r2"], 4) if sub_core["r2"] is not None else None,
                                prediction_bias=round(sub_core["bias"], 4),
                                quantile_coverage_80=round(sub_core["coverage_80"], 4) if sub_core["coverage_80"] is not None else None,
                                mean_interval_width=round(sub_core["mean_width"], 4) if sub_core["mean_width"] is not None else None,
                            )
                        )
                    if col_slices:
                        slices_result[col] = col_slices

        return PerformanceReport(
            timestamp=now_iso,
            sample_size=len(arr_true),
            mae=core["mae"],
            rmse=core["rmse"],
            r2=core["r2"],
            prediction_bias=core["bias"],
            error_distribution=core["error_distribution"],
            quantile_coverage_80=core["coverage_80"],
            mean_interval_width=core["mean_width"],
            slices=slices_result,
        )
