"""
Data Monitoring and Quality Alerting Engine for GeoDemand AI (Phase 17.1).

Monitors:
1. Ingestion failures and error rates.
2. Data freshness against SLA thresholds.
3. Null and duplicate rates with threshold alerting.
4. External provider health, uptime, and fallback status.
5. Pipeline stage execution times and duration benchmarks.
6. Structured data-quality alerts with severity levels (INFO, WARNING, CRITICAL).
7. End-to-end data lineage and provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger("geodemand.monitoring")


# =====================================================================
# 1. Alerting Schema & Severity
# =====================================================================

class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertEvent:
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold_value: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Dispatches, logs, and manages data-quality alerts with cooldown deduplication."""

    def __init__(self, cooldown_seconds: int = 300):
        self.cooldown_seconds = cooldown_seconds
        self.alerts: List[AlertEvent] = []
        self._last_alerted: Dict[str, float] = {}
        self._alert_handlers: List[Callable[[AlertEvent], None]] = []

    def register_handler(self, handler: Callable[[AlertEvent], None]) -> None:
        self._alert_handlers.append(handler)

    def trigger(
        self,
        rule_name: str,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold_value: float,
        dataset_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AlertEvent]:
        now = time.time()
        cooldown_key = f"{rule_name}:{dataset_name or ''}"
        
        # Check cooldown to prevent duplicate alert storms
        last_time = self._last_alerted.get(cooldown_key, 0)
        if (now - last_time) < self.cooldown_seconds:
            return None

        self._last_alerted[cooldown_key] = now
        alert_id = f"alert_{int(now * 1000)}_{len(self.alerts) + 1}"
        event = AlertEvent(
            alert_id=alert_id,
            rule_name=rule_name,
            severity=severity,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=threshold_value,
            dataset_name=dataset_name,
            metadata=metadata or {},
        )
        self.alerts.append(event)
        
        # Log alert
        log_msg = f"[{severity.value}] {rule_name}: {message} (metric={metric_value}, threshold={threshold_value})"
        if severity == AlertSeverity.CRITICAL:
            logger.critical(log_msg)
        elif severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Notify registered handlers (webhooks, email, etc.)
        for handler in self._alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error executing alert handler: {e}")

        return event

    def get_active_alerts(self, min_severity: Optional[AlertSeverity] = None) -> List[AlertEvent]:
        if not min_severity:
            return list(self.alerts)
        hierarchy = {AlertSeverity.INFO: 1, AlertSeverity.WARNING: 2, AlertSeverity.CRITICAL: 3}
        min_level = hierarchy[min_severity]
        return [a for a in self.alerts if hierarchy[a.severity] >= min_level]


# =====================================================================
# 2. Ingestion Failure Monitoring
# =====================================================================

@dataclass
class IngestionAttemptRecord:
    timestamp: str
    dataset_name: str
    total_records: int
    success_count: int
    failure_count: int
    failure_rate: float
    error_reasons: List[str]


class IngestionFailureMonitor:
    """Tracks failed ingestion attempts, validation errors, and failure rates."""

    def __init__(self, alert_manager: AlertManager, max_failure_rate: float = 0.05):
        self.alert_manager = alert_manager
        self.max_failure_rate = max_failure_rate
        self.history: List[IngestionAttemptRecord] = []

    def record_attempt(
        self,
        dataset_name: str,
        total_records: int,
        failed_records: int,
        errors: Optional[List[str]] = None,
    ) -> IngestionAttemptRecord:
        errors = errors or []
        success_count = max(0, total_records - failed_records)
        failure_rate = (failed_records / total_records) if total_records > 0 else 0.0

        record = IngestionAttemptRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=dataset_name,
            total_records=total_records,
            success_count=success_count,
            failure_count=failed_records,
            failure_rate=round(failure_rate, 4),
            error_reasons=errors[:10],  # cap error reason sample
        )
        self.history.append(record)

        if failure_rate > self.max_failure_rate:
            severity = AlertSeverity.CRITICAL if failure_rate > 0.20 else AlertSeverity.WARNING
            self.alert_manager.trigger(
                rule_name="ingestion_failure_rate_exceeded",
                severity=severity,
                message=f"Dataset {dataset_name} failure rate {failure_rate:.1%} exceeded threshold {self.max_failure_rate:.1%}",
                metric_name="failure_rate",
                metric_value=failure_rate,
                threshold_value=self.max_failure_rate,
                dataset_name=dataset_name,
                metadata={"total_records": total_records, "failure_count": failed_records, "errors": errors[:5]},
            )

        return record


# =====================================================================
# 3. Data Freshness Monitoring
# =====================================================================

DEFAULT_FRESHNESS_SLAS = {
    "transactions": timedelta(minutes=15),
    "weather": timedelta(minutes=60),
    "competitors": timedelta(hours=24),
    "feature_store": timedelta(hours=2),
    "demographics": timedelta(days=365),
}


class FreshnessReport(BaseModel):
    dataset_name: str
    latest_timestamp: str
    age_seconds: float
    sla_seconds: float
    is_fresh: bool
    status: str


class DataFreshnessMonitor:
    """Monitors freshness of real-time telemetry and periodic datasets against SLAs."""

    def __init__(self, alert_manager: AlertManager, slas: Optional[Dict[str, timedelta]] = None):
        self.alert_manager = alert_manager
        self.slas = slas or DEFAULT_FRESHNESS_SLAS

    def check_freshness(
        self,
        dataset_name: str,
        latest_timestamp: Union[datetime, str],
        as_of: Optional[datetime] = None,
    ) -> FreshnessReport:
        as_of = as_of or datetime.now(timezone.utc)
        if isinstance(latest_timestamp, str):
            ts = datetime.fromisoformat(latest_timestamp.replace("Z", "+00:00"))
        else:
            ts = latest_timestamp

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        age = (as_of - ts).total_seconds()
        sla_delta = self.slas.get(dataset_name, timedelta(hours=24))
        sla_seconds = sla_delta.total_seconds()
        is_fresh = age <= sla_seconds

        report = FreshnessReport(
            dataset_name=dataset_name,
            latest_timestamp=ts.isoformat(),
            age_seconds=max(0.0, age),
            sla_seconds=sla_seconds,
            is_fresh=is_fresh,
            status="FRESH" if is_fresh else "STALE",
        )

        if not is_fresh:
            severity = AlertSeverity.CRITICAL if age > (sla_seconds * 2) else AlertSeverity.WARNING
            self.alert_manager.trigger(
                rule_name="data_staleness_sla_breached",
                severity=severity,
                message=f"Dataset {dataset_name} is stale by {int(age - sla_seconds)}s beyond SLA {int(sla_seconds)}s",
                metric_name="age_seconds",
                metric_value=age,
                threshold_value=sla_seconds,
                dataset_name=dataset_name,
            )

        return report


# =====================================================================
# 4. Null & Duplicate Rate Monitoring
# =====================================================================

@dataclass
class QualityProfileReport:
    dataset_name: str
    total_rows: int
    null_rate: float
    duplicate_rate: float
    column_null_rates: Dict[str, float]
    duplicate_count: int
    is_acceptable: bool


class QualityProfileMonitor:
    """Monitors null rates and duplicate records across datasets."""

    def __init__(
        self,
        alert_manager: AlertManager,
        max_null_rate: float = 0.01,
        max_dup_rate: float = 0.005,
    ):
        self.alert_manager = alert_manager
        self.max_null_rate = max_null_rate
        self.max_dup_rate = max_dup_rate

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        subset_keys: Optional[List[str]] = None,
    ) -> QualityProfileReport:
        total_rows = len(df)
        if total_rows == 0:
            return QualityProfileReport(
                dataset_name=dataset_name,
                total_rows=0,
                null_rate=0.0,
                duplicate_rate=0.0,
                column_null_rates={},
                duplicate_count=0,
                is_acceptable=True,
            )

        # Null analysis
        col_nulls = {col: float(df[col].isnull().mean()) for col in df.columns}
        overall_null_rate = float(df.isnull().values.mean())

        # Duplicate analysis
        dup_count = int(df.duplicated(subset=subset_keys).sum())
        dup_rate = dup_count / total_rows

        is_acceptable = (overall_null_rate <= self.max_null_rate) and (dup_rate <= self.max_dup_rate)

        # Trigger alerts if thresholds breached
        if overall_null_rate > self.max_null_rate:
            self.alert_manager.trigger(
                rule_name="excessive_null_rate",
                severity=AlertSeverity.WARNING,
                message=f"{dataset_name} overall null rate {overall_null_rate:.2%} exceeded {self.max_null_rate:.2%}",
                metric_name="null_rate",
                metric_value=overall_null_rate,
                threshold_value=self.max_null_rate,
                dataset_name=dataset_name,
            )

        if dup_rate > self.max_dup_rate:
            self.alert_manager.trigger(
                rule_name="excessive_duplicate_rate",
                severity=AlertSeverity.WARNING,
                message=f"{dataset_name} duplicate rate {dup_rate:.2%} exceeded {self.max_dup_rate:.2%}",
                metric_name="duplicate_rate",
                metric_value=dup_rate,
                threshold_value=self.max_dup_rate,
                dataset_name=dataset_name,
            )

        return QualityProfileReport(
            dataset_name=dataset_name,
            total_rows=total_rows,
            null_rate=round(overall_null_rate, 5),
            duplicate_rate=round(dup_rate, 5),
            column_null_rates={k: round(v, 5) for k, v in col_nulls.items()},
            duplicate_count=dup_count,
            is_acceptable=is_acceptable,
        )


# =====================================================================
# 5. External Provider Availability & Health Monitoring
# =====================================================================

class ProviderStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class ProviderHealthReport:
    provider_name: str
    status: ProviderStatus
    latency_ms: float
    consecutive_failures: int
    fallback_active: bool
    last_checked: str


class ProviderAvailabilityMonitor:
    """Tracks external data providers (Open-Meteo, OSM Overpass, OSRM) and fallback status."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.provider_records: Dict[str, ProviderHealthReport] = {}

    def record_status(
        self,
        provider_name: str,
        is_success: bool,
        latency_ms: float,
        fallback_used: bool = False,
    ) -> ProviderHealthReport:
        prev = self.provider_records.get(provider_name)
        consecutive_failures = 0 if is_success else ((prev.consecutive_failures + 1) if prev else 1)

        if is_success:
            status = ProviderStatus.DEGRADED if (latency_ms > 2000 or fallback_used) else ProviderStatus.HEALTHY
        else:
            status = ProviderStatus.UNAVAILABLE if consecutive_failures >= 3 else ProviderStatus.DEGRADED

        report = ProviderHealthReport(
            provider_name=provider_name,
            status=status,
            latency_ms=round(latency_ms, 2),
            consecutive_failures=consecutive_failures,
            fallback_active=fallback_used,
            last_checked=datetime.now(timezone.utc).isoformat(),
        )
        self.provider_records[provider_name] = report

        if status == ProviderStatus.UNAVAILABLE:
            self.alert_manager.trigger(
                rule_name="provider_outage_detected",
                severity=AlertSeverity.CRITICAL,
                message=f"External provider {provider_name} is UNAVAILABLE after {consecutive_failures} failures",
                metric_name="consecutive_failures",
                metric_value=float(consecutive_failures),
                threshold_value=3.0,
                dataset_name=provider_name,
                metadata={"fallback_active": fallback_used},
            )
        elif fallback_used:
            self.alert_manager.trigger(
                rule_name="provider_fallback_activated",
                severity=AlertSeverity.INFO,
                message=f"Provider {provider_name} synthetic offline fallback was activated",
                metric_name="fallback_active",
                metric_value=1.0,
                threshold_value=0.0,
                dataset_name=provider_name,
            )

        return report


# =====================================================================
# 6. Pipeline Execution Time Benchmarking
# =====================================================================

@dataclass
class PipelineStepMetric:
    step_name: str
    duration_seconds: float
    rows_processed: int
    sla_max_seconds: float
    is_within_sla: bool
    completed_at: str


class PipelineExecutionMonitor:
    """Monitors runtime of ingestion, aggregation, and feature-store stages."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.step_history: List[PipelineStepMetric] = []

    def record_step(
        self,
        step_name: str,
        duration_seconds: float,
        rows_processed: int = 0,
        sla_max_seconds: float = 60.0,
    ) -> PipelineStepMetric:
        within_sla = duration_seconds <= sla_max_seconds
        metric = PipelineStepMetric(
            step_name=step_name,
            duration_seconds=round(duration_seconds, 3),
            rows_processed=rows_processed,
            sla_max_seconds=sla_max_seconds,
            is_within_sla=within_sla,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.step_history.append(metric)

        if not within_sla:
            self.alert_manager.trigger(
                rule_name="pipeline_step_duration_sla_breach",
                severity=AlertSeverity.WARNING,
                message=f"Pipeline step {step_name} took {duration_seconds:.2f}s exceeding SLA limit of {sla_max_seconds:.2f}s",
                metric_name="duration_seconds",
                metric_value=duration_seconds,
                threshold_value=sla_max_seconds,
                dataset_name=step_name,
                metadata={"rows_processed": rows_processed},
            )

        return metric


# =====================================================================
# 7. Lineage and Provenance Tracking
# =====================================================================

@dataclass
class DatasetProvenanceRecord:
    dataset_name: str
    version_hash: str
    source_type: str  # real_telemetry, osm_overpass, open_meteo_archive, simulated
    upstream_dependencies: List[str]
    git_commit_sha: Optional[str]
    transformation_function: str
    record_count: int
    generated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLineageTracker:
    """Maintains dataset lineage, transformation provenance, and upstream dependencies."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (Path(__file__).parent.parent.parent / "data" / "processed")
        self.registry: Dict[str, DatasetProvenanceRecord] = {}

    def register(
        self,
        dataset_name: str,
        source_type: str,
        upstream_dependencies: List[str],
        transformation_function: str,
        record_count: int,
        content_bytes: Optional[bytes] = None,
        git_commit_sha: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetProvenanceRecord:
        # Generate version hash
        if content_bytes:
            v_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
        else:
            seed_str = f"{dataset_name}:{record_count}:{datetime.now(timezone.utc).date()}"
            v_hash = hashlib.sha256(seed_str.encode()).hexdigest()[:16]

        record = DatasetProvenanceRecord(
            dataset_name=dataset_name,
            version_hash=v_hash,
            source_type=source_type,
            upstream_dependencies=upstream_dependencies,
            git_commit_sha=git_commit_sha or "HEAD",
            transformation_function=transformation_function,
            record_count=record_count,
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self.registry[dataset_name] = record
        return record

    def export_lineage_graph(self) -> Dict[str, Any]:
        """Exports DAG graph of datasets and upstream dependencies."""
        graph = {}
        for name, record in self.registry.items():
            graph[name] = {
                "version_hash": record.version_hash,
                "source_type": record.source_type,
                "upstream": record.upstream_dependencies,
                "records": record.record_count,
                "generated_at": record.generated_at,
            }
        return graph


# Global unified monitor instance
_default_alert_manager = AlertManager()
default_ingestion_monitor = IngestionFailureMonitor(_default_alert_manager)
default_freshness_monitor = DataFreshnessMonitor(_default_alert_manager)
default_quality_monitor = QualityProfileMonitor(_default_alert_manager)
default_provider_monitor = ProviderAvailabilityMonitor(_default_alert_manager)
default_execution_monitor = PipelineExecutionMonitor(_default_alert_manager)
default_lineage_tracker = DataLineageTracker()
