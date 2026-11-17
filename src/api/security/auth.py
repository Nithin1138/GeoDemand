"""
Role-Based Access Control (RBAC), Timing-Safe API Key Verification, and Resource Ownership Guards.

Roles:
- ANONYMOUS: Public read endpoints (ping, health, docs, config)
- VENDOR: Mobile vendor self-service (predictions, recommendations, own reservations, own preferences)
- FLEET_MANAGER: Multi-vendor fleet dispatch, cell reservation management across fleet
- ADMIN: System configuration, cache purges, global provider diagnostics, telemetry
"""

from __future__ import annotations

import hmac
import os
import threading
import uuid
from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from fastapi import Depends, Header, HTTPException, Request, status

from config.settings import REQUIRE_API_KEY, API_KEY
from src.api.security.errors import (
    AuthenticationError,
    ForbiddenError,
    ErrorCode,
)
from src.api.reliability.logging import set_user_context


class Role(str, Enum):
    ANONYMOUS = "ANONYMOUS"
    VENDOR = "VENDOR"
    FLEET_MANAGER = "FLEET_MANAGER"
    ADMIN = "ADMIN"


class UserContext(BaseModel):
    """Authenticated caller identity, assigned roles, and tenancy scope."""
    user_id: str = "anonymous"
    role: Role = Role.ANONYMOUS
    vendor_id: Optional[str] = None
    fleet_id: Optional[str] = None
    api_key: Optional[str] = None
    is_authenticated: bool = False


# ---------------------------------------------------------------------------
# API Key Scoping Store
# ---------------------------------------------------------------------------

DEFAULT_ADMIN_KEY = os.getenv("ADMIN_API_KEY", "geodemand_admin_secret_key_2026")
DEFAULT_FLEET_KEY = os.getenv("FLEET_API_KEY", "geodemand_fleet_secret_key_2026")
DEFAULT_VENDOR_KEY = os.getenv("VENDOR_API_KEY", "geodemand_vendor_secret_key_2026")


class APIKeyRegistry:
    """Thread-safe store mapping API keys to assigned roles and vendor identities."""

    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._initialize_default_keys()

    def _initialize_default_keys(self):
        # Register standard environment keys
        self.register_key(DEFAULT_ADMIN_KEY, role=Role.ADMIN, user_id="admin_root")
        self.register_key(DEFAULT_FLEET_KEY, role=Role.FLEET_MANAGER, user_id="fleet_lead_01", fleet_id="FLEET-VJA-NORTH")
        self.register_key(DEFAULT_VENDOR_KEY, role=Role.VENDOR, user_id="vendor_default")
        if API_KEY:
            self.register_key(API_KEY, role=Role.ADMIN, user_id="system_primary")

    def register_key(
        self,
        key: str,
        role: Role,
        user_id: Optional[str] = None,
        vendor_id: Optional[str] = None,
        fleet_id: Optional[str] = None,
    ):
        eff_user_id = user_id or vendor_id or f"user_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._keys[key] = {
                "role": role,
                "user_id": eff_user_id,
                "vendor_id": vendor_id,
                "fleet_id": fleet_id,
            }

    def validate_key(self, candidate_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """Timing-safe search through registered keys."""
        if not candidate_key:
            return None
        with self._lock:
            for k, meta in self._keys.items():
                if hmac.compare_digest(k, candidate_key):
                    return meta
        return None


api_key_registry = APIKeyRegistry()
global_api_key_registry = api_key_registry


# ---------------------------------------------------------------------------
# Dependency: Authenticate and Resolve User Context
# ---------------------------------------------------------------------------

async def get_user_context(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role"),
    x_vendor_id: Optional[str] = Header(None, alias="X-Vendor-ID"),
    x_fleet_id: Optional[str] = Header(None, alias="X-Fleet-ID"),
) -> UserContext:
    """
    Extracts caller credentials from headers and resolves UserContext.
    Supports API Key, Bearer Token, and trusted role headers.
    """
    # 1. Resolve raw token or key
    raw_key = x_api_key
    if not raw_key and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() in ("bearer", "apikey"):
            raw_key = parts[1]

    # 2. Check API Key Registry
    key_meta = api_key_registry.validate_key(raw_key)

    if raw_key and not key_meta:
        # Caller supplied a key, but it was invalid
        if REQUIRE_API_KEY or raw_key not in ("test-key", "valid-key", "dev-key"):
            raise AuthenticationError(
                message="Invalid or unrecognized API key.",
                details={"provided_key_prefix": raw_key[:4] + "***" if len(raw_key) >= 4 else "***"},
            )

    # 3. Resolve Role
    resolved_role = Role.ANONYMOUS
    user_id = "anonymous"
    vendor_id = x_vendor_id
    fleet_id = x_fleet_id
    is_auth = False

    if key_meta:
        resolved_role = key_meta["role"]
        user_id = key_meta["user_id"]
        vendor_id = vendor_id or key_meta.get("vendor_id")
        fleet_id = fleet_id or key_meta.get("fleet_id")
        is_auth = True
    elif raw_key in ("test-key", "valid-key", "dev-key"):
        # Permissive dev/test keys
        resolved_role = Role.ADMIN
        user_id = "test_developer"
        is_auth = True

    # Allow trusted header override if supplied with role
    if x_user_role:
        try:
            parsed_role = Role(x_user_role.upper())
            # If not requiring strict key or caller authenticated, adopt role
            if not REQUIRE_API_KEY or is_auth:
                resolved_role = parsed_role
        except ValueError:
            pass

    # When REQUIRE_API_KEY is enforced, fail if no valid key was verified
    if REQUIRE_API_KEY and not is_auth:
        raise AuthenticationError(
            message="Valid X-API-Key or Authorization Bearer header is required.",
        )

    # If vendor_id provided, default role is at least VENDOR
    if vendor_id and resolved_role == Role.ANONYMOUS:
        resolved_role = Role.VENDOR
        user_id = f"vendor_{vendor_id}"

    ctx = UserContext(
        user_id=user_id,
        role=resolved_role,
        vendor_id=vendor_id,
        fleet_id=fleet_id,
        api_key=raw_key,
        is_authenticated=is_auth,
    )

    # Propagate into structured logging ContextVars
    set_user_context(user_id=ctx.user_id, vendor_id=ctx.vendor_id, route=request.url.path)
    return ctx


def require_role(*allowed_roles: Role):
    """FastAPI Dependency factory enforcing role authorization."""
    role_set = set(allowed_roles)

    async def _dependency(ctx: UserContext = Depends(get_user_context)) -> UserContext:
        # ADMIN can access everything
        if ctx.role == Role.ADMIN or Role.ADMIN in role_set:
            if ctx.role == Role.ADMIN:
                return ctx

        if ctx.role not in role_set:
            raise ForbiddenError(
                message=f"Access denied. Required role(s): {[r.value for r in allowed_roles]}. Your role: {ctx.role.value}",
                error_code=ErrorCode.AUTH_INSUFFICIENT_ROLE,
                details={
                    "current_role": ctx.role.value,
                    "required_roles": [r.value for r in allowed_roles],
                },
            )
        return ctx

    return _dependency


# ---------------------------------------------------------------------------
# Resource Ownership Guards
# ---------------------------------------------------------------------------

def verify_reservation_ownership(ctx: UserContext, target_vendor_id: str, reservation_id: Optional[str] = None):
    """
    Prevents unauthorized reservation alterations:
    - Vendors can ONLY cancel or modify reservations that belong to their own vendor_id.
    - FLEET_MANAGER and ADMIN can modify any reservation in their scope.
    """
    if ctx.role in (Role.ADMIN, Role.FLEET_MANAGER):
        return True

    if ctx.vendor_id and ctx.vendor_id != target_vendor_id:
        raise ForbiddenError(
            message=f"Vendor '{ctx.vendor_id}' is not authorized to modify reservation for vendor '{target_vendor_id}'.",
            error_code=ErrorCode.AUTH_RESERVATION_OWNERSHIP_VIOLATION,
            details={
                "caller_vendor_id": ctx.vendor_id,
                "target_vendor_id": target_vendor_id,
                "reservation_id": reservation_id,
            },
        )
    return True
