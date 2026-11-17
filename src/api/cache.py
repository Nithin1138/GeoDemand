"""
GeoDemand AI — High-Performance API Response Cache

Provides in-memory LRU caching with Time-To-Live (TTL) expiration:
- Prevents redundant ML feature assembly and inference for repeated queries
- Supports sub-millisecond response latency on cache hits
- Thread-safe with access tracking and cache telemetry statistics
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class APICache:
    """Thread-safe LRU in-memory response cache with TTL expiration."""

    def __init__(self, max_capacity: int = 500, default_ttl_sec: int = 60):
        self.max_capacity = max_capacity
        self.default_ttl_sec = default_ttl_sec
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve item if present and not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expiry = self._cache[key]
            now = time.time()
            if now > expiry:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end for LRU policy
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store item with expiration timestamp and enforce LRU capacity."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_sec
        expiry = time.time() + ttl

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expiry)

            # Evict oldest if exceeding capacity
            while len(self._cache) > self.max_capacity:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health telemetry and hit rates."""
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round(self._hits / total, 4) if total > 0 else 0.0
            now = time.time()
            active_entries = sum(1 for _, (_, expiry) in self._cache.items() if expiry > now)

            return {
                "active_entries": active_entries,
                "total_capacity": self.max_capacity,
                "default_ttl_seconds": self.default_ttl_sec,
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total,
                "hit_ratio": hit_ratio,
            }


def generate_cache_key(prefix: str, data: Dict[str, Any]) -> str:
    """Generate a deterministic SHA256 cache key from request parameters."""
    try:
        # Serialize with sorted keys to ensure deterministic representation
        serialized = json.dumps(data, sort_keys=True, default=str)
    except Exception:
        serialized = str(sorted(data.items()))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


# Global API cache singleton
api_response_cache = APICache(max_capacity=500, default_ttl_sec=60)
