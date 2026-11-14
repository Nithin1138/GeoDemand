"""
GeoDemand AI — Feature Drift Detection & Distribution Monitoring Module

Member 3 — Machine Learning & MLOps

Implements:
- Kolmogorov-Smirnov (KS) two-sample test for continuous features
- Population Stability Index (PSI) with quantile/decile binning
- Categorical frequency distribution analysis (Categorical PSI & Chi-square)
- Schema validation: missing features, unexpected features, type and value checks
- Configurable practical significance filtering (combining effect size & p-values)
- Multiple comparisons adjustments (Benjamini-Hochberg FDR & Bonferroni)
- Machine-readable serialization of monitoring reports
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Known allowed metadata / identifier columns that are not model features
DEFAULT_ALLOWED_NON_FEATURE_COLS = {
    "h3_cell_id",
    "vendor_id",
    "decision_timestamp",
    "target_timestamp",
    "date",
    "timestamp",
    "expected_customer_count",
    "customer_count",
}

# Features that are strictly non-negative by business/physical logic
NON_NEGATIVE_FEATURE_PREFIXES = (
    "office_",
    "college_",
    "school_",
    "hospital_",
    "mall_",
    "restaurant_",
    "park_",
    "bus_stop_",
    "railway_station_",
    "metro_station_",
    "parking_",
    "building_density",
    "population_density",
    "road_density",
    "commercial_ratio",
    "residential_ratio",
    "industrial_ratio",
    "inventory_capacity",
    "preparation_time",
    "average_order_value",
    "humidity",
    "rainfall",
    "uv_index",
    "visibility",
    "wind_speed",
)


@dataclass
class DriftThresholds:
    """Configurable thresholds for feature drift and schema validation."""

    psi_threshold: float = 0.20  # PSI >= 0.20 indicates significant drift
    psi_moderate_threshold: float = 0.10  # 0.10 <= PSI < 0.20 indicates moderate drift
    ks_alpha: float = 0.05  # Significance level for KS test
    ks_stat_threshold: float = 0.10  # Minimum KS statistic (effect size) to avoid false alarms with large N
    chi2_alpha: float = 0.05  # Significance level for categorical Chi-square test
    multiple_comparison_method: str = "benjamini-hochberg"  # "benjamini-hochberg", "bonferroni", or "none"
    missing_value_rate_threshold: float = 0.05  # Maximum acceptable null rate per feature (5%)
    min_sample_size: int = 20  # Minimum observations required for statistical significance tests


@dataclass
class FeatureDriftResult:
    """Individual feature drift assessment result."""

    feature_name: str
    feature_type: str  # "continuous" or "categorical"
    drift_detected: bool
    stat_name: str  # "ks" or "chi2" or "psi"
    statistic: float
    p_value: Optional[float]
    p_value_adjusted: Optional[float]
    psi: float
    is_statistically_significant: bool
    is_practically_significant: bool
    effect_size: float
    threshold: float
    n_ref: int
    n_current: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchemaValidationResult:
    """Results of schema and value validity checks."""

    is_valid: bool
    missing_features: List[str] = field(default_factory=list)
    unexpected_features: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    infinite_counts: Dict[str, int] = field(default_factory=dict)
    negative_value_counts: Dict[str, int] = field(default_factory=dict)
    unseen_categories: Dict[str, List[str]] = field(default_factory=dict)
    type_mismatches: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """Machine-readable feature drift and schema monitoring report."""

    timestamp: str
    reference_dataset_id: str
    n_reference_samples: int
    n_current_samples: int
    schema_validation: SchemaValidationResult
    feature_results: Dict[str, FeatureDriftResult]
    drifted_features: List[str]
    moderate_drift_features: List[str]
    overall_drift_detected: bool
    max_psi: float
    mean_psi: float
    thresholds: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "reference_dataset_id": self.reference_dataset_id,
            "n_reference_samples": self.n_reference_samples,
            "n_current_samples": self.n_current_samples,
            "schema_validation": self.schema_validation.to_dict(),
            "feature_results": {k: v.to_dict() for k, v in self.feature_results.items()},
            "drifted_features": self.drifted_features,
            "moderate_drift_features": self.moderate_drift_features,
            "overall_drift_detected": self.overall_drift_detected,
            "max_psi": round(float(self.max_psi), 4),
            "mean_psi": round(float(self.mean_psi), 4),
            "thresholds": self.thresholds,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        status = "DRIFT DETECTED" if self.overall_drift_detected else "NO CRITICAL DRIFT"
        lines = [
            f"=== GeoDemand Feature Drift Report [{status}] ===",
            f"Timestamp: {self.timestamp}",
            f"Reference ID: {self.reference_dataset_id} (N={self.n_reference_samples:,})",
            f"Current Sample Size: N={self.n_current_samples:,}",
            f"Schema Valid: {self.schema_validation.is_valid}",
            f"Total Features Monitored: {len(self.feature_results)}",
            f"Features with Critical Drift: {len(self.drifted_features)}",
            f"Features with Moderate Drift: {len(self.moderate_drift_features)}",
            f"Max PSI: {self.max_psi:.4f} | Mean PSI: {self.mean_psi:.4f}",
        ]
        if self.drifted_features:
            lines.append(f"Critical Drift Features: {', '.join(self.drifted_features)}")
        if self.schema_validation.missing_features:
            lines.append(f"Missing Required Features: {', '.join(self.schema_validation.missing_features)}")
        if self.schema_validation.unseen_categories:
            lines.append(f"Unseen Categories: {self.schema_validation.unseen_categories}")
        return "\n".join(lines)


class ReferenceDistributionProfile:
    """
    Versioned summary profile of the training reference feature distributions.
    Stores quantiles, decile bin edges, sample summaries, and categorical proportions.
    """

    def __init__(
        self,
        dataset_id: str,
        feature_columns: List[str],
        categorical_columns: List[str],
        n_samples: int,
        continuous_stats: Dict[str, Dict[str, Any]],
        categorical_stats: Dict[str, Dict[str, Any]],
        reference_samples: Optional[Dict[str, List[float]]] = None,
    ):
        self.dataset_id = dataset_id
        self.feature_columns = sorted(feature_columns)
        self.categorical_columns = sorted(categorical_columns)
        self.continuous_columns = sorted([c for c in self.feature_columns if c not in self.categorical_columns])
        self.n_samples = n_samples
        self.continuous_stats = continuous_stats
        self.categorical_stats = categorical_stats
        self.reference_samples = reference_samples or {}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        dataset_id: str = "reference_v1",
        categorical_columns: Optional[List[str]] = None,
        sample_size_for_ks: int = 5000,
        random_seed: int = 42,
    ) -> "ReferenceDistributionProfile":
        """Compute reference distribution statistics from a DataFrame."""
        if categorical_columns is None:
            categorical_columns = [c for c in df.columns if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype)]

        feature_cols = sorted(list(df.columns))
        cat_cols = sorted(list(categorical_columns))
        cont_cols = sorted([c for c in feature_cols if c not in cat_cols])
        n_samples = len(df)

        continuous_stats: Dict[str, Dict[str, Any]] = {}
        reference_samples: Dict[str, List[float]] = {}
        rng = np.random.RandomState(random_seed)

        for col in cont_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            vals = series.astype(float).values
            # Compute decile bin edges [0, 10, ..., 100]
            deciles = np.percentile(vals, np.linspace(0, 100, 11)).tolist()
            # Ensure unique monotonically increasing bin edges
            unique_deciles = sorted(list(set(deciles)))
            if len(unique_deciles) < 2:
                # Flat distribution (e.g. constant column)
                unique_deciles = [deciles[0] - 1.0, deciles[0] + 1.0]

            # Compute empirical reference bin proportions
            edges = np.array(unique_deciles, dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            ref_counts, _ = np.histogram(vals, bins=edges)
            ref_props = (ref_counts / len(vals)).tolist()

            continuous_stats[col] = {
                "count": int(len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
                "deciles": deciles,
                "bin_edges": unique_deciles,
                "bin_proportions": ref_props,
            }

            # Subsample for fast KS test execution
            if len(vals) > sample_size_for_ks:
                sample_idx = rng.choice(len(vals), size=sample_size_for_ks, replace=False)
                reference_samples[col] = vals[sample_idx].tolist()
            else:
                reference_samples[col] = vals.tolist()

        categorical_stats: Dict[str, Dict[str, Any]] = {}
        for col in cat_cols:
            if col in df.columns:
                counts = df[col].astype(str).value_counts()
                total = counts.sum()
                categorical_stats[col] = {
                    "counts": {k: int(v) for k, v in counts.items()},
                    "proportions": {k: float(v / total) for k, v in counts.items()},
                    "unique_categories": sorted(list(counts.keys())),
                }

        return cls(
            dataset_id=dataset_id,
            feature_columns=feature_cols,
            categorical_columns=cat_cols,
            n_samples=n_samples,
            continuous_stats=continuous_stats,
            categorical_stats=categorical_stats,
            reference_samples=reference_samples,
        )

    def save(self, file_path: Union[str, Path]) -> None:
        """Serialize reference distribution profile to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dataset_id": self.dataset_id,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "continuous_columns": self.continuous_columns,
            "n_samples": self.n_samples,
            "continuous_stats": self.continuous_stats,
            "categorical_stats": self.categorical_stats,
            "reference_samples": self.reference_samples,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved reference distribution profile to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ReferenceDistributionProfile":
        """Load reference distribution profile from JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference distribution artifact not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            dataset_id=data["dataset_id"],
            feature_columns=data["feature_columns"],
            categorical_columns=data.get("categorical_columns", []),
            n_samples=data["n_samples"],
            continuous_stats=data["continuous_stats"],
            categorical_stats=data["categorical_stats"],
            reference_samples=data.get("reference_samples"),
        )


class FeatureDriftDetector:
    """
    Production-grade Feature Drift Detector and Schema Validator.
    Compares incoming feature distributions against a reference profile.
    """

    def __init__(
        self,
        reference_profile: Union[ReferenceDistributionProfile, Path, str, pd.DataFrame],
        thresholds: Optional[DriftThresholds] = None,
        categorical_columns: Optional[List[str]] = None,
        reference_dataset_id: str = "training_reference",
    ):
        self.thresholds = thresholds or DriftThresholds()

        if isinstance(reference_profile, ReferenceDistributionProfile):
            self.profile = reference_profile
        elif isinstance(reference_profile, (str, Path)):
            self.profile = ReferenceDistributionProfile.load(reference_profile)
        elif isinstance(reference_profile, pd.DataFrame):
            self.profile = ReferenceDistributionProfile.from_dataframe(
                df=reference_profile,
                dataset_id=reference_dataset_id,
                categorical_columns=categorical_columns,
            )
        else:
            raise TypeError(f"Unsupported reference_profile type: {type(reference_profile)}")

        self.feature_columns = self.profile.feature_columns
        self.categorical_columns = self.profile.categorical_columns
        self.continuous_columns = self.profile.continuous_columns

    def validate_schema(
        self,
        df: pd.DataFrame,
        allowed_extra_cols: Optional[Set[str]] = None,
    ) -> SchemaValidationResult:
        """
        Validate incoming DataFrame schema, data types, and value domain constraints.
        """
        allowed_extras = allowed_extra_cols or DEFAULT_ALLOWED_NON_FEATURE_COLS
        cols_present = set(df.columns)
        expected_cols = set(self.feature_columns)

        missing = sorted(list(expected_cols - cols_present))
        unexpected = sorted(list(cols_present - expected_cols - allowed_extras))

        null_counts: Dict[str, int] = {}
        infinite_counts: Dict[str, int] = {}
        negative_value_counts: Dict[str, int] = {}
        unseen_categories: Dict[str, List[str]] = {}
        type_mismatches: Dict[str, str] = {}
        errors: List[str] = []

        if missing:
            errors.append(f"Missing required feature columns: {missing}")

        for col in self.feature_columns:
            if col not in df.columns:
                continue

            series = df[col]
            n_null = int(series.isnull().sum())
            if n_null > 0:
                null_counts[col] = n_null
                null_rate = n_null / len(df)
                if null_rate > self.thresholds.missing_value_rate_threshold:
                    errors.append(f"Column '{col}' has null rate {null_rate:.1%} exceeding threshold {self.thresholds.missing_value_rate_threshold:.1%}")

            # Check categorical constraints
            if col in self.categorical_columns:
                known_cats = set(self.profile.categorical_stats.get(col, {}).get("unique_categories", []))
                if known_cats:
                    observed_cats = set(series.dropna().astype(str).unique())
                    unseen = sorted(list(observed_cats - known_cats))
                    if unseen:
                        unseen_categories[col] = unseen
                        errors.append(f"Column '{col}' contains unseen categorical levels: {unseen}")
            else:
                # Check continuous numerical constraints
                if pd.api.types.is_numeric_dtype(series):
                    arr = series.dropna().to_numpy()
                    n_inf = int(np.isinf(arr).sum())
                    if n_inf > 0:
                        infinite_counts[col] = n_inf
                        errors.append(f"Column '{col}' contains {n_inf} infinite values.")

                    # Domain check for non-negative features
                    if any(col.startswith(prefix) for prefix in NON_NEGATIVE_FEATURE_PREFIXES):
                        n_neg = int((arr < -1e-6).sum())
                        if n_neg > 0:
                            negative_value_counts[col] = n_neg
                            errors.append(f"Column '{col}' contains {n_neg} invalid negative values.")
                else:
                    # Non-numeric in numerical column
                    type_mismatches[col] = f"Expected numeric, got {series.dtype}"
                    errors.append(f"Column '{col}' expected numeric, got {series.dtype}")

        is_valid = len(missing) == 0 and len(errors) == 0

        return SchemaValidationResult(
            is_valid=is_valid,
            missing_features=missing,
            unexpected_features=unexpected,
            null_counts=null_counts,
            infinite_counts=infinite_counts,
            negative_value_counts=negative_value_counts,
            unseen_categories=unseen_categories,
            type_mismatches=type_mismatches,
            errors=errors,
        )

    @staticmethod
    def calculate_psi_continuous(
        bin_edges: List[float],
        actual_values: np.ndarray,
        expected_proportions: Optional[Union[List[float], np.ndarray]] = None,
        epsilon: float = 1e-4,
    ) -> float:
        """
        Calculate Population Stability Index (PSI) for continuous features using predefined bin edges.
        Uses Laplace-style epsilon smoothing to ensure numerical stability.
        """
        if len(actual_values) == 0 or len(bin_edges) < 2:
            return 0.0

        # Replace outer edges with -inf and +inf to capture all incoming points
        edges = np.array(bin_edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf

        # Bin counts for actual incoming data (cast to float to handle boolean/integer arrays cleanly)
        actual_arr = np.asarray(actual_values, dtype=float)
        actual_counts, _ = np.histogram(actual_arr, bins=edges)
        k_bins = len(actual_counts)

        # Expected proportions (using empirical reference proportions if provided, else uniform)
        if expected_proportions is None or len(expected_proportions) != k_bins:
            exp_props = np.full(k_bins, 1.0 / k_bins)
        else:
            exp_arr = np.array(expected_proportions, dtype=float)
            exp_props = (exp_arr + epsilon) / (np.sum(exp_arr) + k_bins * epsilon)

        act_props = (actual_counts + epsilon) / (np.sum(actual_counts) + k_bins * epsilon)

        # PSI = sum((actual - expected) * ln(actual / expected))
        psi_val = np.sum((act_props - exp_props) * np.log(act_props / exp_props))
        return float(np.maximum(0.0, psi_val))

    @staticmethod
    def calculate_psi_categorical(
        expected_props: Dict[str, float],
        actual_series: pd.Series,
        epsilon: float = 1e-4,
    ) -> float:
        """Calculate PSI for categorical features with smoothing across all observed categories."""
        if len(actual_series) == 0:
            return 0.0

        actual_counts = actual_series.astype(str).value_counts().to_dict()
        all_categories = sorted(list(set(expected_props.keys()) | set(actual_counts.keys())))
        k = len(all_categories)
        if k == 0:
            return 0.0

        total_act = len(actual_series)
        act_props = np.array([(actual_counts.get(c, 0) + epsilon) / (total_act + k * epsilon) for c in all_categories])
        exp_props = np.array([(expected_props.get(c, 0.0) + epsilon) / (1.0 + k * epsilon) for c in all_categories])

        psi_val = np.sum((act_props - exp_props) * np.log(act_props / exp_props))
        return float(np.maximum(0.0, psi_val))

    @staticmethod
    def adjust_p_values(
        p_values: List[float],
        method: str = "benjamini-hochberg",
    ) -> List[float]:
        """
        Adjust p-values for multiple comparisons across features:
        - Benjamini-Hochberg (FDR)
        - Bonferroni (FWER)
        - None
        """
        m = len(p_values)
        if m == 0:
            return []
        if method == "none":
            return list(p_values)
        if method == "bonferroni":
            return [min(1.0, float(p * m)) for p in p_values]

        if method == "benjamini-hochberg":
            p_arr = np.array(p_values, dtype=float)
            sorted_indices = np.argsort(p_arr)
            sorted_p = p_arr[sorted_indices]

            adj_p = np.zeros(m)
            for i in range(m):
                rank = i + 1
                adj_p[i] = min(1.0, float(sorted_p[i] * m / rank))

            # Enforce monotonicity backwards
            for i in range(m - 2, -1, -1):
                adj_p[i] = min(adj_p[i], adj_p[i + 1])

            result = np.zeros(m)
            result[sorted_indices] = adj_p
            return result.tolist()

        raise ValueError(f"Unknown multiple comparison method: {method}")

    def detect_drift(self, current_df: pd.DataFrame) -> DriftReport:
        """
        Execute comprehensive feature drift detection comparing current_df against reference.
        """
        n_current = len(current_df)
        n_ref = self.profile.n_samples
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Schema and value validation
        schema_res = self.validate_schema(current_df)

        feature_results: Dict[str, FeatureDriftResult] = {}
        critical_drift_features: List[str] = []
        moderate_drift_features: List[str] = []

        # Collect continuous features for multiple-testing adjustment
        continuous_eval_order: List[str] = []
        raw_p_values: List[float] = []

        for col in self.continuous_columns:
            if col not in current_df.columns:
                continue

            series = pd.to_numeric(current_df[col], errors="coerce").dropna()
            if len(series) < 2:
                continue

            ref_sample = self.profile.reference_samples.get(col)
            if not ref_sample or len(ref_sample) < 2:
                continue

            current_vals = series.to_numpy()
            ref_vals = np.array(ref_sample, dtype=float)

            # Two-sample KS test
            ks_res = stats.ks_2samp(ref_vals, current_vals)
            continuous_eval_order.append(col)
            raw_p_values.append(float(ks_res.pvalue))

        # Adjust p-values for multiple comparisons
        adj_p_values = self.adjust_p_values(raw_p_values, method=self.thresholds.multiple_comparison_method)
        adj_p_dict = dict(zip(continuous_eval_order, adj_p_values))

        # Evaluate continuous features
        for col in self.continuous_columns:
            if col not in current_df.columns:
                continue

            series = pd.to_numeric(current_df[col], errors="coerce").dropna()
            current_vals = series.to_numpy() if len(series) > 0 else np.array([])
            col_stats = self.profile.continuous_stats.get(col, {})
            bin_edges = col_stats.get("bin_edges", [])
            bin_props = col_stats.get("bin_proportions")

            # Compute PSI
            psi = self.calculate_psi_continuous(bin_edges, current_vals, expected_proportions=bin_props)

            # Compute KS statistic and p-values
            ref_sample = self.profile.reference_samples.get(col, [])
            if len(current_vals) >= 2 and len(ref_sample) >= 2:
                ks_res = stats.ks_2samp(np.array(ref_sample), current_vals)
                ks_stat = float(ks_res.statistic)
                p_val = float(ks_res.pvalue)
                adj_p = adj_p_dict.get(col, p_val)
            else:
                ks_stat = 0.0
                p_val = 1.0
                adj_p = 1.0

            # Practical significance & statistical significance
            stat_sig = bool(adj_p < self.thresholds.ks_alpha)
            # Require effect size (KS stat threshold or PSI threshold) so large N does not trigger false alarms
            pract_sig = bool((stat_sig and ks_stat >= self.thresholds.ks_stat_threshold) or (psi >= self.thresholds.psi_threshold))
            moderate_sig = bool((psi >= self.thresholds.psi_moderate_threshold and psi < self.thresholds.psi_threshold) or (stat_sig and ks_stat >= (self.thresholds.ks_stat_threshold / 2)))

            msg = "Stable"
            if pract_sig:
                msg = f"Critical drift detected (PSI={psi:.3f}, KS={ks_stat:.3f}, adj_p={adj_p:.4f})"
                critical_drift_features.append(col)
            elif moderate_sig:
                msg = f"Moderate drift detected (PSI={psi:.3f}, KS={ks_stat:.3f}, adj_p={adj_p:.4f})"
                moderate_drift_features.append(col)

            feature_results[col] = FeatureDriftResult(
                feature_name=col,
                feature_type="continuous",
                drift_detected=pract_sig,
                stat_name="ks",
                statistic=round(ks_stat, 4),
                p_value=round(p_val, 6) if p_val is not None else None,
                p_value_adjusted=round(adj_p, 6) if adj_p is not None else None,
                psi=round(psi, 4),
                is_statistically_significant=stat_sig,
                is_practically_significant=pract_sig,
                effect_size=round(ks_stat, 4),
                threshold=self.thresholds.psi_threshold,
                n_ref=n_ref,
                n_current=len(current_vals),
                message=msg,
            )

        # Evaluate categorical features
        for col in self.categorical_columns:
            if col not in current_df.columns:
                continue

            series = current_df[col].dropna()
            cat_stats = self.profile.categorical_stats.get(col, {})
            exp_props = cat_stats.get("proportions", {})

            psi_cat = self.calculate_psi_categorical(exp_props, series)

            # Chi-square test if sufficient counts
            chi2_stat, chi2_p = 0.0, 1.0
            obs_counts = series.astype(str).value_counts()
            known_cats = cat_stats.get("unique_categories", [])

            if len(series) >= self.thresholds.min_sample_size and len(known_cats) > 1:
                obs_vec = [obs_counts.get(c, 0) for c in known_cats]
                exp_vec = [exp_props.get(c, 0.0) * len(series) for c in known_cats]
                # Avoid zero division
                if all(e > 0 for e in exp_vec):
                    try:
                        res = stats.chisquare(f_obs=obs_vec, f_exp=exp_vec)
                        chi2_stat = float(res.statistic)
                        chi2_p = float(res.pvalue)
                    except Exception:
                        pass

            stat_sig = bool(chi2_p < self.thresholds.chi2_alpha)
            pract_sig = bool(psi_cat >= self.thresholds.psi_threshold or (stat_sig and psi_cat >= self.thresholds.psi_moderate_threshold))
            moderate_sig = bool(psi_cat >= self.thresholds.psi_moderate_threshold and psi_cat < self.thresholds.psi_threshold)

            msg = "Stable"
            if pract_sig:
                msg = f"Critical categorical drift (PSI={psi_cat:.3f}, Chi2_p={chi2_p:.4f})"
                critical_drift_features.append(col)
            elif moderate_sig:
                msg = f"Moderate categorical drift (PSI={psi_cat:.3f}, Chi2_p={chi2_p:.4f})"
                moderate_drift_features.append(col)

            feature_results[col] = FeatureDriftResult(
                feature_name=col,
                feature_type="categorical",
                drift_detected=pract_sig,
                stat_name="chi2",
                statistic=round(chi2_stat, 4),
                p_value=round(chi2_p, 6),
                p_value_adjusted=round(chi2_p, 6),
                psi=round(psi_cat, 4),
                is_statistically_significant=stat_sig,
                is_practically_significant=pract_sig,
                effect_size=round(psi_cat, 4),
                threshold=self.thresholds.psi_threshold,
                n_ref=n_ref,
                n_current=len(series),
                message=msg,
            )

        all_psi = [res.psi for res in feature_results.values()]
        max_psi = float(np.max(all_psi)) if all_psi else 0.0
        mean_psi = float(np.mean(all_psi)) if all_psi else 0.0
        overall_drift = len(critical_drift_features) > 0 or not schema_res.is_valid

        return DriftReport(
            timestamp=now_iso,
            reference_dataset_id=self.profile.dataset_id,
            n_reference_samples=n_ref,
            n_current_samples=n_current,
            schema_validation=schema_res,
            feature_results=feature_results,
            drifted_features=critical_drift_features,
            moderate_drift_features=moderate_drift_features,
            overall_drift_detected=overall_drift,
            max_psi=max_psi,
            mean_psi=mean_psi,
            thresholds={
                "psi_threshold": self.thresholds.psi_threshold,
                "psi_moderate_threshold": self.thresholds.psi_moderate_threshold,
                "ks_alpha": self.thresholds.ks_alpha,
                "ks_stat_threshold": self.thresholds.ks_stat_threshold,
                "multiple_comparison_method": self.thresholds.multiple_comparison_method,
            },
        )
