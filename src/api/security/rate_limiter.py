"""
API Rate Limiting Engine for GeoDemand AI.

Implements a thread-safe sliding-window counter rate limiter with role-aware tiers,
preventing brute force attacks, scraping, and endpoint denial of service.
Returns standard RFC 6585 headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, Retry-After).
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Dict, Optional, Tuple


# Default Tier Rate Limits (requests per 60 seconds)
DEFAULT_TIER_LIMITS = {
    "anonymous": 30,
    "vendor": 60,
    "fleet_manager": 300,
    "admin": 1000,
    "webhook": 120,
}
ROLE_RATE_LIMITS = DEFAULT_TIER_LIMITS


class RateLimiter:
    """Sliding-window rate limiter per client IP, API key, or vendor identity."""

    def __init__(self, default_window_sec: int = 60):
        self.default_window_sec = default_window_sec
        # client_id -> deque of request timestamps
        self._windows: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._lock = threading.RLock()

    def get_limit_for_role(self, role: str) -> int:
        return ROLE_RATE_LIMITS.get(str(role).lower().strip(), 60)

    def check_rate_limit(
        self,
        client_key: Optional[str] = None,
        limit: Optional[int] = None,
        window_sec: Optional[int] = None,
        role: Optional[str] = None,
        client_id: Optional[str] = None,
        window_seconds: Optional[int] = None,
        default_limit: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """
        Evaluates whether client request is allowed within the sliding window.

        Returns:
            (is_allowed, remaining, reset_after_seconds)
        """
        eff_client = client_id or client_key or "127.0.0.1"
        win_sec = window_seconds or window_sec or self.default_window_sec

        if limit is not None:
            eff_limit = limit
        elif default_limit is not None:
            eff_limit = default_limit
        elif role is not None:
            eff_limit = self.get_limit_for_role(role)
        else:
            eff_limit = 60

        now = time.time()
        cutoff = now - win_sec

        with self._lock:
            q = self._windows[eff_client]

            # Evict timestamps outside sliding window
            while q and q[0] <= cutoff:
                q.popleft()

            current_count = len(q)
            if current_count < eff_limit:
                q.append(now)
                remaining = max(0, eff_limit - len(q))
                reset_after = int(win_sec - (now - q[0])) if q else win_sec
                return True, remaining, max(1, reset_after)
            else:
                remaining = 0
                reset_after = int(win_sec - (now - q[0])) if q else win_sec
                return False, remaining, max(1, reset_after)

    def get_headers(
        self,
        limit: int,
        remaining: int,
        reset_after_sec: int,
    ) -> Dict[str, str]:
        """Formats standard RFC rate limit response headers."""
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_after_sec),
        }
        if remaining == 0:
            headers["Retry-After"] = str(reset_after_sec)
        return headers

    def reset(self):
        with self._lock:
            self._windows.clear()


# Global Singleton Rate Limiter
rate_limiter = RateLimiter()
global_rate_limiter = rate_limiter
