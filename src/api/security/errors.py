"""
Standardized error codes, response schemas, and exception hierarchy for GeoDemand AI.

Provides:
- Enumerated error catalogue with documented HTTP status codes and resolution guides.
- Uniform JSON error response contract (`StandardErrorResponse`).
- Domain-specific exception classes that automatically map to structured error payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import HTTPException, status


# ---------------------------------------------------------------------------
# Error Code Catalogue
# ---------------------------------------------------------------------------

class ErrorCode(str, Enum):
    # Authentication & Authorization (401, 403)
    AUTH_UNAUTHORIZED = "AUTH_UNAUTHORIZED"
    AUTH_INVALID_API_KEY = "AUTH_INVALID_API_KEY"
    AUTH_FORBIDDEN = "AUTH_FORBIDDEN"
    AUTH_INSUFFICIENT_ROLE = "AUTH_INSUFFICIENT_ROLE"
    AUTH_RESERVATION_OWNERSHIP_VIOLATION = "AUTH_RESERVATION_OWNERSHIP_VIOLATION"

    # Rate Limiting (429)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Validation (422, 400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_COORDINATES = "INVALID_COORDINATES"
    INVALID_H3_CELL = "INVALID_H3_CELL"

    # Resource & Conflict (404, 409)
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESERVATION_CONFLICT = "RESERVATION_CONFLICT"
    CELL_CAPACITY_EXCEEDED = "CELL_CAPACITY_EXCEEDED"

    # Provider & Reliability (502, 503, 504)
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_CIRCUIT_OPEN = "PROVIDER_CIRCUIT_OPEN"
    DATABASE_ERROR = "DATABASE_ERROR"
    REDIS_ERROR = "REDIS_ERROR"
    ROUTING_PROVIDER_FAILURE = "ROUTING_PROVIDER_FAILURE"
    SERVICE_DEGRADED = "SERVICE_DEGRADED"

    # Webhooks & Payments (400, 401, 409)
    WEBHOOK_VERIFICATION_FAILED = "WEBHOOK_VERIFICATION_FAILED"
    WEBHOOK_REPLAY_DETECTED = "WEBHOOK_REPLAY_DETECTED"
    WEBHOOK_DUPLICATE_IDEMPOTENCY = "WEBHOOK_DUPLICATE_IDEMPOTENCY"

    # General HTTP Aliases
    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "RESOURCE_NOT_FOUND"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_SERVER_ERROR"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    CIRCUIT_BREAKER_OPEN = "PROVIDER_CIRCUIT_OPEN"

    # Server (500)
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


ERROR_CODE_METADATA: Dict[ErrorCode, Dict[str, Any]] = {
    ErrorCode.AUTH_UNAUTHORIZED: {
        "status_code": status.HTTP_401_UNAUTHORIZED,
        "description": "Authentication credentials are missing or invalid.",
        "resolution": "Supply a valid 'X-API-Key' or 'Authorization: Bearer <token>' header.",
    },
    ErrorCode.AUTH_INVALID_API_KEY: {
        "status_code": status.HTTP_401_UNAUTHORIZED,
        "description": "The provided API key does not match recognized access credentials.",
        "resolution": "Verify the API key or rotate expired keys in the vendor portal.",
    },
    ErrorCode.AUTH_FORBIDDEN: {
        "status_code": status.HTTP_403_FORBIDDEN,
        "description": "Caller lacks permissions to access or modify this resource.",
        "resolution": "Elevate permissions or verify vendor/fleet role assignments.",
    },
    ErrorCode.AUTH_INSUFFICIENT_ROLE: {
        "status_code": status.HTTP_403_FORBIDDEN,
        "description": "Operation requires a higher role privilege (e.g. FLEET_MANAGER or ADMIN).",
        "resolution": "Authenticate with credentials possessing required administrative scope.",
    },
    ErrorCode.AUTH_RESERVATION_OWNERSHIP_VIOLATION: {
        "status_code": status.HTTP_403_FORBIDDEN,
        "description": "A vendor attempted to modify or cancel a reservation belonging to another vendor.",
        "resolution": "Only the reservation owner or a fleet manager may alter active reservations.",
    },
    ErrorCode.RATE_LIMIT_EXCEEDED: {
        "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
        "description": "Request volume exceeded sliding-window rate limit budget.",
        "resolution": "Back off and retry after the duration specified in the 'Retry-After' header.",
    },
    ErrorCode.VALIDATION_ERROR: {
        "status_code": 422,
        "description": "Request body failed boundary, format, or type constraints.",
        "resolution": "Inspect 'details' list for specific invalid fields and schema requirements.",
    },
    ErrorCode.INVALID_COORDINATES: {
        "status_code": status.HTTP_400_BAD_REQUEST,
        "description": "Latitude must be [-90, 90] and Longitude must be [-180, 180].",
        "resolution": "Provide valid numeric WGS-84 decimal degree coordinates.",
    },
    ErrorCode.INVALID_H3_CELL: {
        "status_code": status.HTTP_400_BAD_REQUEST,
        "description": "Provided string is not a valid 15-character hexadecimal H3 index.",
        "resolution": "Use h3.latlng_to_cell(lat, lng, 8) to generate valid cell indices.",
    },
    ErrorCode.RESOURCE_NOT_FOUND: {
        "status_code": status.HTTP_404_NOT_FOUND,
        "description": "Requested entity (reservation, vendor, notification) does not exist.",
        "resolution": "Verify target resource identifier before repeating request.",
    },
    ErrorCode.RESERVATION_CONFLICT: {
        "status_code": status.HTTP_409_CONFLICT,
        "description": "Conflicting active reservation exists for this vendor or cell spot.",
        "resolution": "Cancel existing reservation or select an alternative candidate zone.",
    },
    ErrorCode.CELL_CAPACITY_EXCEEDED: {
        "status_code": status.HTTP_409_CONFLICT,
        "description": "The target H3 zone has reached its maximum physical or category vendor capacity.",
        "resolution": "Select adjacent candidate cells with available capacity.",
    },
    ErrorCode.PROVIDER_TIMEOUT: {
        "status_code": status.HTTP_504_GATEWAY_TIMEOUT,
        "description": "Upstream geospatial, weather, or routing provider exceeded timeout deadline.",
        "resolution": "Request automatically fell back to synthetic estimates; retry shortly.",
    },
    ErrorCode.PROVIDER_CIRCUIT_OPEN: {
        "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
        "description": "Upstream provider circuit breaker is OPEN due to consecutive failures.",
        "resolution": "Provider is in cooldown recovery; fallback proxy mode is active.",
    },
    ErrorCode.DATABASE_ERROR: {
        "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
        "description": "Relational or persistence storage connection failure.",
        "resolution": "Automatic fallback store engaged; transactions queued in memory.",
    },
    ErrorCode.REDIS_ERROR: {
        "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
        "description": "Distributed cache or lock connection interrupted.",
        "resolution": "Local in-memory locks and TTL cache engaged automatically.",
    },
    ErrorCode.ROUTING_PROVIDER_FAILURE: {
        "status_code": status.HTTP_502_BAD_GATEWAY,
        "description": "Road routing engine unreachable.",
        "resolution": "Haversine straight-line distance and grid road interpolation active.",
    },
    ErrorCode.WEBHOOK_VERIFICATION_FAILED: {
        "status_code": status.HTTP_401_UNAUTHORIZED,
        "description": "Cryptographic webhook HMAC signature mismatch.",
        "resolution": "Ensure webhook payload is signed with the registered merchant secret.",
    },
    ErrorCode.WEBHOOK_REPLAY_DETECTED: {
        "status_code": status.HTTP_400_BAD_REQUEST,
        "description": "Webhook timestamp is older than freshness window (>300s).",
        "resolution": "Disallow replayed webhook deliveries; synchronize server clock.",
    },
    ErrorCode.WEBHOOK_DUPLICATE_IDEMPOTENCY: {
        "status_code": status.HTTP_409_CONFLICT,
        "description": "Idempotency key already processed.",
        "resolution": "Duplicate delivery acknowledged safely; no redundant state created.",
    },
    ErrorCode.INTERNAL_SERVER_ERROR: {
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "description": "An unexpected server exception occurred.",
        "resolution": "Check logs with the attached correlation_id for tracing.",
    },
    ErrorCode.SERVICE_DEGRADED: {
        "status_code": status.HTTP_200_OK,
        "description": "Request succeeded in degraded mode using offline synthetic data.",
        "resolution": "Inspect 'data_freshness' and 'source_type' metadata fields.",
    },
}


# ---------------------------------------------------------------------------
# Standard Error Response Schema
# ---------------------------------------------------------------------------

class StandardErrorResponse(BaseModel):
    """Uniform JSON error contract returned across all API endpoints."""
    error_code: str = Field(default="", description="Machine-readable error enumeration code")
    error: Optional[str] = Field(None, description="Alias for error_code")
    message: str = Field(..., description="Human-readable descriptive error explanation")
    detail: Optional[str] = Field(None, description="Backward-compatible detail field")
    status_code: int = Field(..., description="HTTP status code")
    correlation_id: str = Field(..., description="Trace correlation identifier for diagnostics")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    path: Optional[str] = Field(None, description="Request path that generated the error")
    details: Optional[Dict[str, Any]] = Field(None, description="Structured diagnostics context")

    def model_post_init(self, __context: Any) -> None:
        if not self.error_code and self.error:
            self.error_code = self.error
        elif not self.error and self.error_code:
            self.error = self.error_code
        if not self.detail:
            self.detail = self.message

    model_config = {
        "json_schema_extra": {
            "example": {
                "error_code": "AUTH_RESERVATION_OWNERSHIP_VIOLATION",
                "message": "Vendor VND-01 cannot cancel reservation belonging to VND-02",
                "status_code": 403,
                "correlation_id": "req_corr_7b8a912e8c",
                "timestamp": "2026-11-16T14:55:00Z",
                "path": "/v1/fleet/cancel-reservation",
                "details": {"requested_vendor_id": "VND-01", "owner_vendor_id": "VND-02"},
            }
        }
    }


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class GeoDemandAPIError(HTTPException):
    """Base API exception that renders into a StandardErrorResponse."""
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        meta = ERROR_CODE_METADATA.get(error_code, {})
        effective_status = status_code or meta.get("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(
            status_code=effective_status,
            detail={
                "error_code": error_code.value,
                "message": message,
                "details": self.details,
            },
            headers=headers,
        )

    def to_response(self, correlation_id: Optional[str] = None) -> StandardErrorResponse:
        return StandardErrorResponse(
            error_code=self.error_code.value,
            error=self.error_code.value,
            message=self.message,
            status_code=self.status_code,
            correlation_id=correlation_id or "corr_err",
            details=self.details,
        )


class AuthenticationError(GeoDemandAPIError):
    def __init__(self, message: str = "Invalid or missing authentication credentials", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.AUTH_UNAUTHORIZED,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
            headers={"WWW-Authenticate": "Bearer, API-Key"},
        )


class ForbiddenError(GeoDemandAPIError):
    def __init__(
        self,
        message: str = "Access forbidden to requested resource",
        error_code: ErrorCode = ErrorCode.AUTH_FORBIDDEN,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class RateLimitExceededError(GeoDemandAPIError):
    def __init__(self, retry_after_sec: int = 60, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=f"Rate limit exceeded. Retry in {retry_after_sec} seconds.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
            headers={"Retry-After": str(retry_after_sec)},
        )


class ResourceNotFoundError(GeoDemandAPIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class ConflictError(GeoDemandAPIError):
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.RESERVATION_CONFLICT,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class ProviderTimeoutError(GeoDemandAPIError):
    def __init__(self, provider_name: str, timeout_sec: float, details: Optional[Dict[str, Any]] = None):
        det = details or {}
        det.update({"provider": provider_name, "timeout_seconds": timeout_sec})
        super().__init__(
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            message=f"Provider '{provider_name}' exceeded timeout deadline of {timeout_sec}s.",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details=det,
        )


class CircuitOpenError(GeoDemandAPIError):
    def __init__(self, service_name: str, recovery_cooldown_sec: float, details: Optional[Dict[str, Any]] = None):
        det = details or {}
        det.update({"service": service_name, "recovery_cooldown_sec": recovery_cooldown_sec})
        super().__init__(
            error_code=ErrorCode.PROVIDER_CIRCUIT_OPEN,
            message=f"Circuit breaker for '{service_name}' is OPEN. Cooling down for {recovery_cooldown_sec}s.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=det,
        )
