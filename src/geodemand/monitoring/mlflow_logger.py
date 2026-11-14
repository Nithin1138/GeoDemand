"""
GeoDemand AI — MLflow Monitoring Logger Module

Member 3 — Machine Learning & MLOps

Logs monitoring summaries, drift statistics, prediction quality checks,
and delayed performance metrics to MLflow SQLite tracking store.
Gracefully degrades if MLflow is not installed or tracking is unavailable.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


class MLflowMonitoringLogger:
    """
    MLflow integration logger for GeoDemand monitoring results.
    """

    def __init__(
        self,
        experiment_name: str = "GeoDemand",
        tracking_uri: Optional[str] = None,
        model_version: str = "v2.0.0",
        feature_schema_version: str = "v2.2-next-hour-forecast-cyclical",
    ):
        self.experiment_name = experiment_name
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version

        if tracking_uri:
            self.tracking_uri = tracking_uri
        else:
            db_path = ROOT_DIR / "mlflow.db"
            self.tracking_uri = f"sqlite:///{db_path.resolve()}".replace("\\", "/")

        self._configured = False

    def _configure(self) -> bool:
        """Configure MLflow connection."""
        if not HAS_MLFLOW:
            logger.info("MLflow not installed — monitoring logging is disabled.")
            return False

        if not self._configured:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._configured = True
            except Exception as e:
                logger.warning(f"Could not configure MLflow tracking URI: {e}")
                return False
        return True

    def log_drift_report(
        self,
        drift_report: Any,
        run_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log a feature drift report to MLflow.
        drift_report can be DriftReport or dict.
        """
        if not self._configure():
            return None

        report_dict = drift_report.to_dict() if hasattr(drift_report, "to_dict") else drift_report
        name = run_name or f"drift_monitoring_{report_dict.get('timestamp', '')[:10]}"

        try:
            with mlflow.start_run(run_name=name) as run:
                # Log metadata parameters
                mlflow.log_params({
                    "run_type": "feature_drift_monitoring",
                    "model_version": self.model_version,
                    "feature_schema_version": self.feature_schema_version,
                    "monitoring_timestamp": report_dict.get("timestamp"),
                    "reference_dataset_id": report_dict.get("reference_dataset_id"),
                    "n_reference_samples": report_dict.get("n_reference_samples"),
                    "n_current_samples": report_dict.get("n_current_samples"),
                    "overall_drift_detected": str(report_dict.get("overall_drift_detected")),
                    "schema_is_valid": str(report_dict.get("schema_validation", {}).get("is_valid")),
                })

                # Log top-level aggregate metrics
                mlflow.log_metrics({
                    "monitoring.max_psi": float(report_dict.get("max_psi", 0.0)),
                    "monitoring.mean_psi": float(report_dict.get("mean_psi", 0.0)),
                    "monitoring.n_drifted_features": len(report_dict.get("drifted_features", [])),
                    "monitoring.n_moderate_drift_features": len(report_dict.get("moderate_drift_features", [])),
                    "monitoring.drift_detected_flag": 1.0 if report_dict.get("overall_drift_detected") else 0.0,
                })

                # Log per-feature PSI and KS metrics
                feature_res = report_dict.get("feature_results", {})
                for feat, res in feature_res.items():
                    if "psi" in res:
                        # Clean metric name for MLflow
                        safe_name = feat.replace(" ", "_").replace("-", "_")
                        mlflow.log_metric(f"feature_psi.{safe_name}", float(res["psi"]))
                    if "statistic" in res and res.get("stat_name") == "ks":
                        safe_name = feat.replace(" ", "_").replace("-", "_")
                        mlflow.log_metric(f"feature_ks_stat.{safe_name}", float(res["statistic"]))

                # Log full machine-readable artifact
                with tempfile.TemporaryDirectory() as tmp_dir:
                    artifact_file = Path(tmp_dir) / "drift_report.json"
                    with open(artifact_file, "w", encoding="utf-8") as f:
                        json.dump(report_dict, f, indent=2)
                    mlflow.log_artifact(str(artifact_file), artifact_path="monitoring")

                logger.info(f"MLflow drift monitoring run '{name}' logged successfully (Run ID: {run.info.run_id}).")
                return run.info.run_id
        except Exception as e:
            logger.warning(f"MLflow drift report logging failed (non-fatal): {e}")
            return None

    def log_performance_report(
        self,
        perf_report: Any,
        run_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log a delayed ground-truth performance evaluation report to MLflow.
        perf_report can be PerformanceReport or dict.
        """
        if not self._configure():
            return None

        report_dict = perf_report.to_dict() if hasattr(perf_report, "to_dict") else perf_report
        name = run_name or f"performance_monitoring_{report_dict.get('timestamp', '')[:10]}"

        try:
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({
                    "run_type": "performance_monitoring",
                    "model_version": self.model_version,
                    "feature_schema_version": self.feature_schema_version,
                    "monitoring_timestamp": report_dict.get("timestamp"),
                    "evaluated_sample_size": report_dict.get("sample_size"),
                    "data_disclaimer": "SIMULATED_TARGETS",
                })

                # Overall performance metrics
                metrics_to_log: Dict[str, float] = {
                    "perf.mae": float(report_dict.get("mae", 0.0)),
                    "perf.rmse": float(report_dict.get("rmse", 0.0)),
                    "perf.bias": float(report_dict.get("prediction_bias", 0.0)),
                }
                if report_dict.get("r2") is not None:
                    metrics_to_log["perf.r2"] = float(report_dict["r2"])
                if report_dict.get("quantile_coverage_80") is not None:
                    metrics_to_log["perf.coverage_80"] = float(report_dict["quantile_coverage_80"])
                if report_dict.get("mean_interval_width") is not None:
                    metrics_to_log["perf.interval_width"] = float(report_dict["mean_interval_width"])

                # Log error distribution percentiles
                err_dist = report_dict.get("error_distribution", {})
                for k, v in err_dist.items():
                    if isinstance(v, (int, float)):
                        metrics_to_log[f"perf.error_{k}"] = float(v)

                # Log slice breakdown metrics
                slices = report_dict.get("slices", {})
                for dim, slice_list in slices.items():
                    for s in slice_list:
                        s_val = str(s.get("slice_value", "")).replace(" ", "_")
                        metrics_to_log[f"slice_{dim}_{s_val}.mae"] = float(s.get("mae", 0.0))
                        metrics_to_log[f"slice_{dim}_{s_val}.rmse"] = float(s.get("rmse", 0.0))

                mlflow.log_metrics(metrics_to_log)

                # Save JSON artifact
                with tempfile.TemporaryDirectory() as tmp_dir:
                    artifact_file = Path(tmp_dir) / "performance_report.json"
                    with open(artifact_file, "w", encoding="utf-8") as f:
                        json.dump(report_dict, f, indent=2)
                    mlflow.log_artifact(str(artifact_file), artifact_path="monitoring")

                logger.info(f"MLflow performance monitoring run '{name}' logged successfully (Run ID: {run.info.run_id}).")
                return run.info.run_id
        except Exception as e:
            logger.warning(f"MLflow performance report logging failed (non-fatal): {e}")
            return None

    def log_prediction_quality_report(
        self,
        quality_report: Any,
        run_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log prediction quality validation checks to MLflow.
        """
        if not self._configure():
            return None

        report_dict = quality_report.to_dict() if hasattr(quality_report, "to_dict") else quality_report
        name = run_name or "prediction_quality_check"

        try:
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({
                    "run_type": "prediction_quality_audit",
                    "model_version": self.model_version,
                    "sample_size": report_dict.get("sample_size"),
                    "is_valid": str(report_dict.get("is_valid")),
                })

                mlflow.log_metrics({
                    "quality.is_valid": 1.0 if report_dict.get("is_valid") else 0.0,
                    "quality.n_violations": len(report_dict.get("violations", [])),
                    "quality.has_nan": 1.0 if report_dict.get("has_nan") else 0.0,
                    "quality.has_inf": 1.0 if report_dict.get("has_infinite") else 0.0,
                    "quality.has_neg": 1.0 if report_dict.get("has_negative") else 0.0,
                    "quality.has_inversion": 1.0 if report_dict.get("has_quantile_inversion") else 0.0,
                })

                if report_dict.get("mean_predicted") is not None:
                    mlflow.log_metric("quality.mean_predicted", float(report_dict["mean_predicted"]))

                logger.info(f"MLflow prediction quality run '{name}' logged successfully (Run ID: {run.info.run_id}).")
                return run.info.run_id
        except Exception as e:
            logger.warning(f"MLflow prediction quality logging failed (non-fatal): {e}")
            return None
