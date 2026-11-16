"""
API Latency Metrics Collector & Prometheus Telemetry for GeoDemand AI.

Maintains rolling percentile statistics (p50, p90, p95, p99), error rates,
and route-level execution summaries. Exposes both JSON diagnostics and
standard Prometheus exposition format.
"""

from __future__ import annotations

import collections
import statistics
import threading
from typing import Any, Dict, List, Optional


class LatencyMetricsCollector:
    """Thread-safe latency tracker computing quantiles and Prometheus summaries."""

    def __init__(self, max_samples_per_route: int = 500, window_minutes: int = 15):
        self.max_samples = max_samples_per_route
        self.window_minutes = window_minutes
        self._latencies: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.max_samples)
        )
        self._request_counts: Dict[str, int] = collections.defaultdict(int)
        self._status_counts: Dict[str, int] = collections.defaultdict(int)
        self._lock = threading.RLock()

    def record(self, route: str, method: str, status_code: int, latency_ms: float):
        """Records a completed HTTP request's latency and response status."""
        norm_route = route or "unknown"
        norm_method = (method or "GET").upper()
        key = f"{norm_method}:{norm_route}"
        status_bucket = f"{status_code // 100}xx"

        with self._lock:
            self._latencies[key].append(latency_ms)
            self._request_counts[key] += 1
            self._status_counts[status_bucket] += 1
            self._status_counts[f"{status_code}"] += 1

    record_request = record

    def get_route_percentiles(self, key: str) -> Dict[str, float]:
        with self._lock:
            samples = list(self._latencies.get(key, []))

        if not samples:
            return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_s = sorted(samples)
        n = len(sorted_s)

        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(round((p / 100.0) * n)) - 1))
            return round(sorted_s[idx], 2)

        return {
            "count": self._request_counts[key],
            "sample_size": n,
            "avg": round(statistics.mean(samples), 2),
            "min": round(min(samples), 2),
            "max": round(max(samples), 2),
            "p50": pct(50.0),
            "p90": pct(90.0),
            "p95": pct(95.0),
            "p99": pct(99.0),
        }

    def get_metrics_json(self) -> Dict[str, Any]:
        """Returns structured JSON summary of system latency and status distributions."""
        with self._lock:
            all_keys = list(self._latencies.keys())
            status_copy = dict(self._status_counts)
            total_reqs = sum(self._request_counts.values())

        routes_summary = {k: self.get_route_percentiles(k) for k in all_keys}

        # Calculate overall latency stats
        all_samples: List[float] = []
        with self._lock:
            for d in self._latencies.values():
                all_samples.extend(list(d))

        global_p50 = global_p90 = global_p95 = global_p99 = global_avg = global_min = global_max = 0.0
        if all_samples:
            s_all = sorted(all_samples)
            n = len(s_all)
            global_avg = round(statistics.mean(s_all), 2)
            global_min = round(min(s_all), 2)
            global_max = round(max(s_all), 2)
            global_p50 = round(s_all[min(n - 1, max(0, int(0.50 * n) - 1))], 2)
            global_p90 = round(s_all[min(n - 1, max(0, int(0.90 * n) - 1))], 2)
            global_p95 = round(s_all[min(n - 1, max(0, int(0.95 * n) - 1))], 2)
            global_p99 = round(s_all[min(n - 1, max(0, int(0.99 * n) - 1))], 2)

        total_errors = sum(v for k, v in status_copy.items() if k.startswith(("4", "5")))
        error_rate = round((total_errors / max(1, total_reqs)) * 100.0, 2)

        return {
            "total_requests": total_reqs,
            "total_errors": total_errors,
            "error_rate_pct": error_rate,
            "latency_ms": {
                "avg": global_avg,
                "min": global_min,
                "max": global_max,
                "p50": global_p50,
                "p90": global_p90,
                "p95": global_p95,
                "p99": global_p99,
            },
            "status_codes": status_copy,
            "routes": routes_summary,
        }

    get_metrics_summary = get_metrics_json

    def format_prometheus_metrics(self) -> str:
        """Emits Prometheus standard plain-text exposition format."""
        lines = [
            "# HELP geodemand_api_requests_total Total HTTP requests handled by GeoDemand API",
            "# TYPE geodemand_api_requests_total counter",
            "# HELP geodemand_http_requests_total Total HTTP requests handled by GeoDemand API",
            "# TYPE geodemand_http_requests_total counter",
        ]
        with self._lock:
            for key, count in self._request_counts.items():
                parts = key.split(":", 1)
                m, r = parts[0], parts[1] if len(parts) > 1 else "unknown"
                lines.append(f'geodemand_api_requests_total{{method="{m}",route="{r}"}} {count}')
                lines.append(f'geodemand_http_requests_total{{method="{m}",route="{r}"}} {count}')

            lines.append("# HELP geodemand_api_latency_ms API request latency percentiles in milliseconds")
            lines.append("# TYPE geodemand_api_latency_ms summary")
            lines.append("# HELP geodemand_http_latency_ms_p50 50th percentile request latency in milliseconds")
            lines.append("# TYPE geodemand_http_latency_ms_p50 gauge")
            for key in self._latencies.keys():
                parts = key.split(":", 1)
                m, r = parts[0], parts[1] if len(parts) > 1 else "unknown"
                stats = self.get_route_percentiles(key)
                lines.append(f'geodemand_http_latency_ms_p50{{method="{m}",route="{r}"}} {stats["p50"]}')
                lines.append(f'geodemand_api_latency_ms{{quantile="0.5",method="{m}",route="{r}"}} {stats["p50"]}')
                lines.append(f'geodemand_api_latency_ms{{quantile="0.9",method="{m}",route="{r}"}} {stats["p90"]}')
                lines.append(f'geodemand_api_latency_ms{{quantile="0.99",method="{m}",route="{r}"}} {stats["p99"]}')
                lines.append(f'geodemand_http_latency_ms_p95{{method="{m}",route="{r}"}} {stats["p95"]}')
                lines.append(f'geodemand_http_latency_ms_p99{{method="{m}",route="{r}"}} {stats["p99"]}')

        return "\n".join(lines) + "\n"

    def reset(self):
        with self._lock:
            self._latencies.clear()
            self._request_counts.clear()
            self._status_counts.clear()


# Global Singleton Collector
latency_collector = LatencyMetricsCollector()
global_metrics_collector = latency_collector


def format_prometheus_metrics() -> str:
    return latency_collector.format_prometheus_metrics()
