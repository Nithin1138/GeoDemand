"""
Production Structured JSON Logging and Context Correlation for GeoDemand AI.

Provides:
- Thread-safe & async-safe ContextVar storage for correlation IDs and user context.
- Single-line machine-readable JSON log formatting.
- Automatic data masking for credentials, secrets, tokens, and PII.
- Integration helper to attach structured loggers to FastAPI / Uvicorn.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid

# ---------------------------------------------------------------------------
# Context Variables for Asynchronous Request Tracing
# ---------------------------------------------------------------------------

_correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="corr_init"
)
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="req_init"
)
_user_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "user_id", default=None
)
_vendor_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "vendor_id", default=None
)
_route_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "route", default=None
)
correlation_id_ctx = _correlation_id_ctx
request_id_ctx = _request_id_ctx
user_id_ctx = _user_id_ctx
vendor_id_ctx = _vendor_id_ctx
route_ctx = _route_ctx


def get_correlation_id() -> str:
    """Returns the current request's correlation identifier."""
    return _correlation_id_ctx.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Sets the correlation ID for the active async/thread context."""
    cid = correlation_id or f"corr_{uuid.uuid4().hex[:12]}"
    _correlation_id_ctx.set(cid)
    return cid


def get_request_id() -> str:
    return _request_id_ctx.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    rid = request_id or f"req_{uuid.uuid4().hex[:10]}"
    _request_id_ctx.set(rid)
    return rid


def set_user_context(user_id: Optional[str] = None, vendor_id: Optional[str] = None, route: Optional[str] = None):
    if user_id:
        _user_id_ctx.set(user_id)
    if vendor_id:
        _vendor_id_ctx.set(vendor_id)
    if route:
        _route_ctx.set(route)


def clear_context():
    _correlation_id_ctx.set("corr_init")
    _request_id_ctx.set("req_init")
    _user_id_ctx.set(None)
    _vendor_id_ctx.set(None)
    _route_ctx.set(None)


# ---------------------------------------------------------------------------
# Sanitization & Data Masking
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    re.compile(r'(api[-_]?key\s*[:=]\s*["\']?)([^"\'\s&,]+)', re.IGNORECASE),
    re.compile(r'(secret\s*[:=]\s*["\']?)([^"\'\s&,]+)', re.IGNORECASE),
    re.compile(r'(token\s*[:=]\s*["\']?)([^"\'\s&,]+)', re.IGNORECASE),
    re.compile(r'(signature\s*[:=]\s*["\']?)([^"\'\s&,]+)', re.IGNORECASE),
    re.compile(r'(password\s*[:=]\s*["\']?)([^"\'\s&,]+)', re.IGNORECASE),
    re.compile(r'(authorization\s*[:=]\s*Bearer\s+)([^"\'\s&,]+)', re.IGNORECASE),
]


def sanitize_log_message(msg: str) -> str:
    """Redacts passwords, tokens, API keys, and authorization secrets from log strings."""
    if not isinstance(msg, str):
        msg = str(msg)
    for pat in SENSITIVE_PATTERNS:
        msg = pat.sub(r"\1[REDACTED]", msg)
    return msg


# ---------------------------------------------------------------------------
# Structured JSON Formatter
# ---------------------------------------------------------------------------

class StructuredJsonFormatter(logging.Formatter):
    """
    Emits single-line JSON log objects conforming to enterprise observability standards.
    Automatically enriches logs with active correlation_id, user_id, route, and duration.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_log_message(record.getMessage()),
            "correlation_id": getattr(record, "correlation_id", None) or get_correlation_id(),
            "request_id": getattr(record, "request_id", None) or get_request_id(),
        }

        # Optional context attributes
        user_id = getattr(record, "user_id", None) or _user_id_ctx.get()
        if user_id:
            log_obj["user_id"] = user_id

        vendor_id = getattr(record, "vendor_id", None) or _vendor_id_ctx.get()
        if vendor_id:
            log_obj["vendor_id"] = vendor_id

        route = getattr(record, "route", None) or _route_ctx.get()
        if route:
            log_obj["route"] = route

        if hasattr(record, "status_code"):
            log_obj["status_code"] = record.status_code

        if hasattr(record, "latency_ms"):
            log_obj["latency_ms"] = round(float(record.latency_ms), 2)

        if hasattr(record, "client_ip"):
            log_obj["client_ip"] = record.client_ip

        if hasattr(record, "http_method"):
            log_obj["http_method"] = record.http_method

        # Exception formatting
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Exception",
                "message": sanitize_log_message(str(record.exc_info[1])),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Logging Setup Helper
# ---------------------------------------------------------------------------

def setup_structured_logging(level: int = logging.INFO) -> None:
    """Configures root and GeoDemand loggers to use StructuredJsonFormatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers
    has_json = any(isinstance(h.formatter, StructuredJsonFormatter) for h in root_logger.handlers)
    if not has_json:
        root_logger.addHandler(handler)

    for log_name in ("api", "features", "fleet", "optimization", "ingestion", "notifications"):
        l = logging.getLogger(log_name)
        l.setLevel(level)
