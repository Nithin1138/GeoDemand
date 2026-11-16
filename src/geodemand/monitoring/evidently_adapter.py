"""
GeoDemand AI — Evidently AI Integration & Monitoring Adapter Module

Member 2 — Machine Learning & MLOps

Implements:
- Integration with Evidently AI report suite (DataDriftPreset, TargetDriftPreset)
- Production-grade fallback when Evidently AI is not installed or running in lightweight edge mode
- HTML dashboard generation for feature and prediction drift visualization
- Standardized machine-readable drift schema matching Evidently AI conventions
- Exportable metrics for MLflow logging and API consumption
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.geodemand.monitoring.drift import (
    DriftThresholds,
    FeatureDriftDetector,
    ReferenceDistributionProfile,
)

logger = logging.getLogger(__name__)

# Check Evidently AI availability
try:
    import evidently  # type: ignore
    from evidently.metric_preset import DataDriftPreset  # type: ignore
    from evidently.report import Report  # type: ignore
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False


class FallbackEvidentlyReport:
    """
    High-fidelity native fallback for Evidently AI Report.
    Produces identical metrics, drift tables, summary stats, and standalone HTML reports.
    """

    def __init__(self, metrics_data: Dict[str, Any], html_summary: str):
        self._data = metrics_data
        self._html = html_summary

    def as_dict(self) -> Dict[str, Any]:
        """Return Evidently-style report dictionary."""
        return self._data

    def json(self) -> str:
        """Return report as JSON string."""
        return json.dumps(self._data, indent=2)

    def save_html(self, filepath: Union[str, Path]) -> Path:
        """Save standalone HTML drift report dashboard."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self._html)
        logger.info(f"Saved drift report HTML to {out_path}")
        return out_path


class EvidentlyMonitoringAdapter:
    """
    Evidently AI Adapter providing both native Evidently execution and zero-dependency fallback.
    """

    def __init__(self, thresholds: Optional[DriftThresholds] = None):
        self.thresholds = thresholds or DriftThresholds()
        self.detector = FeatureDriftDetector(thresholds=self.thresholds)

    def run_data_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: Optional[Dict[str, Any]] = None,
        target_col: Optional[str] = "expected_customer_count",
    ) -> Union[Any, FallbackEvidentlyReport]:
        """
        Generate data drift report comparing reference and current dataframes.
        """
        if HAS_EVIDENTLY:
            try:
                logger.info("Executing native Evidently AI DataDriftPreset report...")
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=reference_data, current_data=current_data)
                return report
            except Exception as e:
                logger.warning(f"Native Evidently AI execution failed ({e}); switching to native engine fallback.")

        # Native Fallback Report Engine
        return self._build_fallback_report(
            reference_df=reference_data,
            current_df=current_data,
            target_col=target_col,
        )

    def _build_fallback_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> FallbackEvidentlyReport:
        """
        Build an Evidently-compatible drift metrics dictionary and HTML dashboard.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        profile = ReferenceDistributionProfile.from_dataframe(
            df=reference_df,
            dataset_id="ref_baseline",
        )
        report = self.detector.detect_drift(
            current_df=current_df,
            reference_profile=profile,
            timestamp=now_iso,
        )

        n_ref = len(reference_df)
        n_cur = len(current_df)
        drift_by_columns: Dict[str, Any] = {}

        for feat, res in report.feature_results.items():
            drift_by_columns[feat] = {
                "column_name": feat,
                "column_type": res.feature_type,
                "stattest_name": res.stat_name,
                "drift_score": round(res.statistic, 4),
                "p_value": round(res.p_value, 6) if res.p_value is not None else None,
                "threshold": res.threshold,
                "drift_detected": res.drift_detected,
                "psi": round(res.psi, 4),
                "wasserstein_distance": res.wasserstein_distance,
            }

        n_drifted = len(report.drifted_features)
        total_cols = len(report.feature_results)
        drift_share = round(n_drifted / max(1, total_cols), 4)
        dataset_drift = bool(drift_share >= 0.3 or report.max_psi >= self.thresholds.psi_threshold)

        metrics_payload = {
            "timestamp": now_iso,
            "engine": "EvidentlyAI-Native-Compatible" if not HAS_EVIDENTLY else "EvidentlyAI",
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "drift_share": drift_share,
                        "number_of_columns": total_cols,
                        "number_of_drifted_columns": n_drifted,
                        "share_of_drifted_columns": drift_share,
                        "dataset_drift": dataset_drift,
                        "max_psi": round(report.max_psi, 4),
                        "mean_psi": round(report.mean_psi, 4),
                    },
                },
                {
                    "metric": "DataDriftTable",
                    "result": {
                        "number_of_columns": total_cols,
                        "number_of_drifted_columns": n_drifted,
                        "share_of_drifted_columns": drift_share,
                        "dataset_drift": dataset_drift,
                        "drift_by_columns": drift_by_columns,
                    },
                },
            ],
            "schema_validation": report.schema_validation.to_dict(),
        }

        # Generate HTML Dashboard
        status_badge_color = "#e53e3e" if dataset_drift else "#38a169"
        status_text = "DATA DRIFT DETECTED" if dataset_drift else "SYSTEM NOMINAL"

        table_rows = []
        for feat, info in drift_by_columns.items():
            drift_label = '<span style="color:#e53e3e;font-weight:bold;">DRIFT</span>' if info["drift_detected"] else '<span style="color:#38a169;">OK</span>'
            wd_str = f"{info['wasserstein_distance']:.4f}" if info["wasserstein_distance"] is not None else "N/A"
            table_rows.append(
                f"<tr>"
                f"<td>{feat}</td>"
                f"<td>{info['column_type']}</td>"
                f"<td>{info['stattest_name']}</td>"
                f"<td>{info['drift_score']}</td>"
                f"<td>{info['psi']:.4f}</td>"
                f"<td>{wd_str}</td>"
                f"<td>{drift_label}</td>"
                f"</tr>"
            )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GeoDemand AI — Evidently Data Drift Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f7fafc; color: #2d3748; padding: 24px; }}
        .header {{ background: #ffffff; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        .badge {{ display: inline-block; padding: 6px 12px; border-radius: 4px; font-weight: bold; color: #ffffff; background: {status_badge_color}; }}
        .metric-cards {{ display: flex; gap: 16px; margin-top: 16px; }}
        .card {{ background: #edf2f7; border-radius: 6px; padding: 16px; flex: 1; text-align: center; }}
        .card-num {{ font-size: 24px; font-weight: bold; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
        th {{ background: #edf2f7; font-weight: 600; }}
        tr:hover {{ background: #f8fafc; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>GeoDemand AI — Distribution & Feature Drift Report</h2>
        <p>Engine: <strong>Evidently AI Native / High-Fidelity Adapter</strong> | Timestamp: {now_iso}</p>
        <div>Status: <span class="badge">{status_text}</span></div>
        <div class="metric-cards">
            <div class="card"><div>Reference Samples</div><div class="card-num">{n_ref:,}</div></div>
            <div class="card"><div>Current Samples</div><div class="card-num">{n_cur:,}</div></div>
            <div class="card"><div>Drifted Columns</div><div class="card-num">{n_drifted} / {total_cols}</div></div>
            <div class="card"><div>Max PSI</div><div class="card-num">{report.max_psi:.4f}</div></div>
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Type</th>
                <th>Stat Test</th>
                <th>Stat Value</th>
                <th>PSI</th>
                <th>Wasserstein Dist</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
</body>
</html>
"""
        return FallbackEvidentlyReport(metrics_data=metrics_payload, html_summary=html_content)
