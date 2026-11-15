"""
GeoDemand AI — Prediction Quality & Contract Validation Module

Member 3 — Machine Learning & MLOps

Validates model outputs before returning to callers:
- Finite outputs (strictly no NaN, +inf, -inf)
- Exact shape match with input batch size
- Customer-count domain contract: non-negative and within reasonable upper bounds
- Quantile ordering contract: P10 <= P50 <= P90
- Required input features presence and sanity
- Explicit error reporting (never silently concealing corrupted predictions)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionQualityError(ValueError):
    """Raised when model predictions violate contract or quality checks."""

    def __init__(self, message: str, violations: Optional[List[str]] = None):
        super().__init__(message)
        self.violations = violations or [message]


@dataclass
class PredictionQualityReport:
    """Detailed result of prediction quality validation."""

    is_valid: bool
    sample_size: int
    has_nan: bool = False
    has_infinite: bool = False
    has_negative: bool = False
    has_quantile_inversion: bool = False
    shape_matches: bool = True
    violations: List[str] = field(default_factory=list)
    min_predicted: Optional[float] = None
    max_predicted: Optional[float] = None
    mean_predicted: Optional[float] = None
    mean_interval_width: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PredictionValidator:
    """
    Validates model inputs and predictions according to GeoDemand contracts.
    """

    def __init__(
        self,
        max_reasonable_count: float = 50000.0,
        tolerance: float = 1e-4,
        raise_on_violation: bool = True,
    ):
        self.max_reasonable_count = max_reasonable_count
        self.tolerance = tolerance
        self.raise_on_violation = raise_on_violation

    def validate_inputs(
        self,
        features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        required_features: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Validate inference inputs before passing to model.
        """
        violations: List[str] = []

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        elif isinstance(features, pd.DataFrame):
            df = features
        else:
            violations.append(f"Invalid input type: {type(features)}. Expected DataFrame, dict, or list of dicts.")
            if self.raise_on_violation:
                raise PredictionQualityError("Input type violation", violations)
            return violations

        if required_features:
            missing = [f for f in required_features if f not in df.columns]
            if missing:
                violations.append(f"Missing required inference features: {missing}")

        # Check for unhandled NaNs or infs in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            vals = df[col].to_numpy()
            if np.isnan(vals).any():
                violations.append(f"Input feature '{col}' contains NaN values.")
            if np.isinf(vals).any():
                violations.append(f"Input feature '{col}' contains Infinite values.")

        if violations and self.raise_on_violation:
            raise PredictionQualityError(
                f"Input quality validation failed with {len(violations)} violations: {violations[0]}",
                violations=violations,
            )

        return violations

    def validate_predictions(
        self,
        predictions: Union[np.ndarray, List[float], pd.Series],
        expected_n: int,
    ) -> PredictionQualityReport:
        """
        Validate point predictions (P50 / expected customer count).
        """
        preds = np.asarray(predictions, dtype=float)
        violations: List[str] = []

        has_nan = bool(np.isnan(preds).any())
        has_inf = bool(np.isinf(preds).any())
        has_neg = bool((preds < -self.tolerance).any())
        shape_matches = bool(preds.shape == (expected_n,))

        if not shape_matches:
            violations.append(f"Prediction output shape mismatch: expected ({expected_n},), got {preds.shape}")

        if has_nan:
            n_nan = int(np.isnan(preds).sum())
            violations.append(f"Model returned {n_nan} NaN predictions.")

        if has_inf:
            n_inf = int(np.isinf(preds).sum())
            violations.append(f"Model returned {n_inf} infinite predictions.")

        if has_neg:
            n_neg = int((preds < -self.tolerance).sum())
            min_val = float(np.min(preds))
            violations.append(f"Customer-count contract violation: {n_neg} negative predictions (min={min_val:.4f}).")

        # Sanity bound check
        if len(preds) > 0 and not (has_nan or has_inf):
            max_val = float(np.max(preds))
            if max_val > self.max_reasonable_count:
                violations.append(
                    f"Extreme prediction value detected: {max_val:.2f} > reasonable upper bound {self.max_reasonable_count}"
                )

        is_valid = len(violations) == 0
        min_p = float(np.min(preds)) if len(preds) > 0 and not (has_nan or has_inf) else None
        max_p = float(np.max(preds)) if len(preds) > 0 and not (has_nan or has_inf) else None
        mean_p = float(np.mean(preds)) if len(preds) > 0 and not (has_nan or has_inf) else None

        report = PredictionQualityReport(
            is_valid=is_valid,
            sample_size=len(preds),
            has_nan=has_nan,
            has_infinite=has_inf,
            has_negative=has_neg,
            shape_matches=shape_matches,
            violations=violations,
            min_predicted=min_p,
            max_predicted=max_p,
            mean_predicted=mean_p,
        )

        if not is_valid and self.raise_on_violation:
            raise PredictionQualityError(
                f"Prediction quality validation failed: {violations[0]}",
                violations=violations,
            )

        return report

    def validate_quantiles(
        self,
        p10: Union[np.ndarray, List[float]],
        p50: Union[np.ndarray, List[float]],
        p90: Union[np.ndarray, List[float]],
        expected_n: int,
    ) -> PredictionQualityReport:
        """
        Validate quantile predictions ensuring P10 <= P50 <= P90 and valid ranges.
        """
        q10 = np.asarray(p10, dtype=float)
        q50 = np.asarray(p50, dtype=float)
        q90 = np.asarray(p90, dtype=float)

        violations: List[str] = []

        shape_matches = (
            q10.shape == (expected_n,) and q50.shape == (expected_n,) and q90.shape == (expected_n,)
        )
        if not shape_matches:
            violations.append(
                f"Quantile output shape mismatch: expected ({expected_n},), got "
                f"P10={q10.shape}, P50={q50.shape}, P90={q90.shape}"
            )

        has_nan = bool(np.isnan(q10).any() or np.isnan(q50).any() or np.isnan(q90).any())
        has_inf = bool(np.isinf(q10).any() or np.isinf(q50).any() or np.isinf(q90).any())
        has_neg = bool((q10 < -self.tolerance).any() or (q50 < -self.tolerance).any() or (q90 < -self.tolerance).any())

        if has_nan:
            violations.append("Quantile predictions contain NaN values.")
        if has_inf:
            violations.append("Quantile predictions contain Infinite values.")
        if has_neg:
            violations.append("Quantile predictions contain negative values.")

        # Quantile ordering: P10 <= P50 <= P90
        has_inversion = False
        if not (has_nan or has_inf):
            inv_10_50 = q10 > (q50 + self.tolerance)
            inv_50_90 = q50 > (q90 + self.tolerance)

            if inv_10_50.any():
                n_inv = int(inv_10_50.sum())
                violations.append(f"Quantile monotonicity violation: P10 > P50 in {n_inv} cases.")
                has_inversion = True

            if inv_50_90.any():
                n_inv = int(inv_50_90.sum())
                violations.append(f"Quantile monotonicity violation: P50 > P90 in {n_inv} cases.")
                has_inversion = True

        is_valid = len(violations) == 0
        min_p = float(np.min(q50)) if len(q50) > 0 and not (has_nan or has_inf) else None
        max_p = float(np.max(q50)) if len(q50) > 0 and not (has_nan or has_inf) else None
        mean_p = float(np.mean(q50)) if len(q50) > 0 and not (has_nan or has_inf) else None
        mean_width = float(np.mean(q90 - q10)) if len(q10) > 0 and not (has_nan or has_inf) else None

        report = PredictionQualityReport(
            is_valid=is_valid,
            sample_size=len(q50),
            has_nan=has_nan,
            has_infinite=has_inf,
            has_negative=has_neg,
            has_quantile_inversion=has_inversion,
            shape_matches=shape_matches,
            violations=violations,
            min_predicted=min_p,
            max_predicted=max_p,
            mean_predicted=mean_p,
            mean_interval_width=mean_width,
        )

        if not is_valid and self.raise_on_violation:
            raise PredictionQualityError(
                f"Quantile quality validation failed: {violations[0]}",
                violations=violations,
            )

        return report
