"""
Unit Tests for Data Monitoring, Quality Alerting, and Provenance Tracking (Phase 17.1).
"""

from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd

from src.validation.data_monitoring import (
    AlertEvent,
    AlertManager,
    AlertSeverity,
    DataFreshnessMonitor,
    DataLineageTracker,
    FreshnessReport,
    IngestionFailureMonitor,
    PipelineExecutionMonitor,
    ProviderAvailabilityMonitor,
    ProviderStatus,
    QualityProfileMonitor,
)


def test_alert_manager_severity_and_deduplication():
    """Verify alert manager triggers, severity routing, and cooldown deduplication."""
    manager = AlertManager(cooldown_seconds=60)
    received = []
    manager.register_handler(lambda a: received.append(a))

    # First alert
    alert = manager.trigger(
        rule_name="test_rule",
        severity=AlertSeverity.WARNING,
        message="Warning occurred",
        metric_name="error_rate",
        metric_value=0.10,
        threshold_value=0.05,
        dataset_name="transactions",
    )
    assert alert is not None
    assert alert.severity == AlertSeverity.WARNING
    assert len(received) == 1

    # Immediate second alert on same rule + dataset is suppressed by cooldown
    duplicate = manager.trigger(
        rule_name="test_rule",
        severity=AlertSeverity.WARNING,
        message="Warning occurred again",
        metric_name="error_rate",
        metric_value=0.12,
        threshold_value=0.05,
        dataset_name="transactions",
    )
    assert duplicate is None
    assert len(received) == 1

    # Alert on a different dataset triggers normally
    new_alert = manager.trigger(
        rule_name="test_rule",
        severity=AlertSeverity.CRITICAL,
        message="Critical on weather",
        metric_name="error_rate",
        metric_value=0.50,
        threshold_value=0.05,
        dataset_name="weather",
    )
    assert new_alert is not None
    assert len(received) == 2

    # Verify severity filtering
    criticals = manager.get_active_alerts(min_severity=AlertSeverity.CRITICAL)
    assert len(criticals) == 1
    assert criticals[0].dataset_name == "weather"


def test_ingestion_failure_monitoring_and_alerts():
    """Verify ingestion error tracking and failure rate breach alerts."""
    alert_mgr = AlertManager(cooldown_seconds=0)
    monitor = IngestionFailureMonitor(alert_mgr, max_failure_rate=0.10)

    # Normal ingestion attempt (5% failure rate -> under 10% threshold)
    rec1 = monitor.record_attempt(
        dataset_name="transactions",
        total_records=100,
        failed_records=5,
        errors=["missing latitude"],
    )
    assert rec1.failure_rate == 0.05
    assert rec1.success_count == 95
    assert len(alert_mgr.alerts) == 0

    # High failure rate attempt (30% failure rate -> triggers CRITICAL alert)
    rec2 = monitor.record_attempt(
        dataset_name="transactions",
        total_records=100,
        failed_records=30,
        errors=["invalid H3 cell", "malformed JSON"],
    )
    assert rec2.failure_rate == 0.30
    assert len(alert_mgr.alerts) == 1
    assert alert_mgr.alerts[0].severity == AlertSeverity.CRITICAL
    assert alert_mgr.alerts[0].metric_value == 0.30


def test_data_freshness_sla_monitoring():
    """Verify data freshness evaluation against defined SLAs."""
    alert_mgr = AlertManager(cooldown_seconds=0)
    slas = {"transactions": timedelta(minutes=15)}
    monitor = DataFreshnessMonitor(alert_mgr, slas=slas)

    now = datetime.now(timezone.utc)
    fresh_ts = now - timedelta(minutes=5)
    stale_ts = now - timedelta(minutes=45)

    # Fresh dataset check
    report_fresh = monitor.check_freshness("transactions", latest_timestamp=fresh_ts, as_of=now)
    assert report_fresh.is_fresh is True
    assert report_fresh.status == "FRESH"
    assert len(alert_mgr.alerts) == 0

    # Stale dataset check
    report_stale = monitor.check_freshness("transactions", latest_timestamp=stale_ts, as_of=now)
    assert report_stale.is_fresh is False
    assert report_stale.status == "STALE"
    assert len(alert_mgr.alerts) == 1
    assert alert_mgr.alerts[0].rule_name == "data_staleness_sla_breached"


def test_null_and_duplicate_monitoring():
    """Verify null and duplicate rate profiling and threshold alerts."""
    alert_mgr = AlertManager(cooldown_seconds=0)
    monitor = QualityProfileMonitor(alert_mgr, max_null_rate=0.05, max_dup_rate=0.05)

    # Clean dataset
    clean_df = pd.DataFrame({
        "id": ["tx_1", "tx_2", "tx_3", "tx_4"],
        "amount": [10.0, 20.0, 30.0, 40.0],
    })
    res_clean = monitor.profile_dataframe(clean_df, "clean_data", subset_keys=["id"])
    assert res_clean.is_acceptable is True
    assert res_clean.null_rate == 0.0
    assert res_clean.duplicate_count == 0
    assert len(alert_mgr.alerts) == 0

    # Dirty dataset with nulls and duplicates
    dirty_df = pd.DataFrame({
        "id": ["tx_1", "tx_1", "tx_2", "tx_3"],  # 1 duplicate
        "amount": [10.0, None, None, 40.0],      # 2 nulls (50% nulls in column)
    })
    res_dirty = monitor.profile_dataframe(dirty_df, "dirty_data", subset_keys=["id"])
    assert res_dirty.is_acceptable is False
    assert res_dirty.duplicate_count == 1
    assert res_dirty.duplicate_rate == 0.25
    # Triggers alerts for both null rate and duplicate rate
    assert len(alert_mgr.alerts) == 2


def test_provider_availability_and_fallback():
    """Verify external provider availability tracking and fallback alerts."""
    alert_mgr = AlertManager(cooldown_seconds=0)
    monitor = ProviderAvailabilityMonitor(alert_mgr)

    # Provider healthy
    h1 = monitor.record_status("open_meteo", is_success=True, latency_ms=120.0)
    assert h1.status == ProviderStatus.HEALTHY
    assert h1.consecutive_failures == 0

    # Provider fallback used
    h2 = monitor.record_status("open_meteo", is_success=True, latency_ms=50.0, fallback_used=True)
    assert h2.fallback_active is True
    assert h2.status == ProviderStatus.DEGRADED
    assert len(alert_mgr.alerts) == 1
    assert alert_mgr.alerts[0].rule_name == "provider_fallback_activated"

    # Provider 3 consecutive failures -> UNAVAILABLE alert
    monitor.record_status("osm_overpass", is_success=False, latency_ms=5000.0)
    monitor.record_status("osm_overpass", is_success=False, latency_ms=5000.0)
    h_down = monitor.record_status("osm_overpass", is_success=False, latency_ms=5000.0)
    assert h_down.status == ProviderStatus.UNAVAILABLE
    assert h_down.consecutive_failures == 3

    outage_alerts = [a for a in alert_mgr.alerts if a.rule_name == "provider_outage_detected"]
    assert len(outage_alerts) == 1
    assert outage_alerts[0].severity == AlertSeverity.CRITICAL


def test_pipeline_execution_benchmarking():
    """Verify pipeline duration tracking and SLA latency breaches."""
    alert_mgr = AlertManager(cooldown_seconds=0)
    monitor = PipelineExecutionMonitor(alert_mgr)

    # Step within SLA
    step1 = monitor.record_step("ingest_transactions", duration_seconds=12.5, rows_processed=1000, sla_max_seconds=30.0)
    assert step1.is_within_sla is True
    assert len(alert_mgr.alerts) == 0

    # Step breaching SLA
    step2 = monitor.record_step("build_feature_store", duration_seconds=95.0, rows_processed=50000, sla_max_seconds=60.0)
    assert step2.is_within_sla is False
    assert len(alert_mgr.alerts) == 1
    assert alert_mgr.alerts[0].rule_name == "pipeline_step_duration_sla_breach"


def test_data_lineage_and_provenance_tracking():
    """Verify data lineage DAG registration and export."""
    tracker = DataLineageTracker()

    # Register raw ingestion
    tracker.register(
        dataset_name="raw_transactions",
        source_type="real_telemetry",
        upstream_dependencies=[],
        transformation_function="validate_and_parse_transactions",
        record_count=1000,
        content_bytes=b"raw_tx_bytes",
    )

    # Register aggregated actuals
    tracker.register(
        dataset_name="hourly_actuals",
        source_type="real_telemetry",
        upstream_dependencies=["raw_transactions"],
        transformation_function="aggregate_actuals_by_hour_h3",
        record_count=150,
        content_bytes=b"actuals_bytes",
    )

    # Register feature store
    tracker.register(
        dataset_name="feature_store",
        source_type="hybrid",
        upstream_dependencies=["hourly_actuals", "static_features", "weather_hourly"],
        transformation_function="build_40_feature_store",
        record_count=150,
        content_bytes=b"feature_bytes",
    )

    dag = tracker.export_lineage_graph()
    assert "raw_transactions" in dag
    assert "hourly_actuals" in dag
    assert "feature_store" in dag
    assert dag["feature_store"]["upstream"] == ["hourly_actuals", "static_features", "weather_hourly"]
    assert len(dag["feature_store"]["version_hash"]) == 16
