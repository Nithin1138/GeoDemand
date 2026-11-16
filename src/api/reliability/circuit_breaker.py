"""
Circuit Breaker Pattern implementation for GeoDemand AI.

Prevents cascade failures across external geospatial APIs, database connections,
and message brokers using a 3-state finite state machine (CLOSED -> OPEN -> HALF_OPEN).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation: requests pass through
    OPEN = "OPEN"            # Tripped: requests fail immediately to fallback
    HALF_OPEN = "HALF_OPEN"  # Recovery testing: limited probe requests allowed


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the circuit breaker is OPEN."""
    def __init__(self, service_name: str, cooldown_remaining: float):
        self.service_name = service_name
        self.cooldown_remaining = round(cooldown_remaining, 2)
        super().__init__(
            f"Circuit breaker for '{service_name}' is OPEN. "
            f"Retry in {self.cooldown_remaining}s."
        )


class CircuitBreaker:
    """
    Thread-safe circuit breaker with automatic half-open probing and metrics tracking.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 30.0,
        half_open_success_threshold: int = 2,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_success_threshold = half_open_success_threshold
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._tripped_at: float = 0.0
        self._total_calls = 0
        self._total_failures = 0
        self._total_trips = 0
        self._last_failure_reason: Optional[str] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._tripped_at >= self.recovery_timeout_sec:
                    self._state = CircuitState.HALF_OPEN
                    self._consecutive_successes = 0
                    logger.info(f"CircuitBreaker '{self.name}' transitioned from OPEN to HALF_OPEN probe state.")
            return self._state

    @property
    def failure_count(self) -> int:
        return self._consecutive_failures

    def is_available(self) -> bool:
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self):
        with self._lock:
            self._total_calls += 1
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
                    logger.info(f"CircuitBreaker '{self.name}' probe succeeded; transitioned to CLOSED.")
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self, exc: Optional[Exception] = None):
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            reason = str(exc) if exc is not None else "Operation failure"
            self._last_failure_reason = reason

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately trips back to open
                self._state = CircuitState.OPEN
                self._tripped_at = time.time()
                self._total_trips += 1
                logger.warning(f"CircuitBreaker '{self.name}' probe failed ({reason}); re-tripped to OPEN.")
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._tripped_at = time.time()
                    self._total_trips += 1
                    logger.warning(
                        f"CircuitBreaker '{self.name}' reached {self._consecutive_failures} consecutive failures. "
                        f"Tripping to OPEN for {self.recovery_timeout_sec}s. Last error: {reason}"
                    )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Executes synchronous callable under circuit-breaker protection."""
        curr_state = self.state
        if curr_state == CircuitState.OPEN:
            cooldown_left = max(0.0, self.recovery_timeout_sec - (time.time() - self._tripped_at))
            raise CircuitBreakerOpenException(self.name, cooldown_left)

        try:
            res = func(*args, **kwargs)
            self.record_success()
            return res
        except Exception as e:
            self.record_failure(e)
            raise e

    execute = call

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            curr_state = self.state
            cooldown_left = 0.0
            if curr_state == CircuitState.OPEN:
                cooldown_left = max(0.0, self.recovery_timeout_sec - (time.time() - self._tripped_at))

            return {
                "name": self.name,
                "state": curr_state.value,
                "is_available": self.is_available(),
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.failure_threshold,
                "cooldown_remaining_sec": round(cooldown_left, 1),
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_trips": self._total_trips,
                "last_failure_reason": self._last_failure_reason,
            }

    def reset(self):
        """Forces circuit breaker back to initial CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._tripped_at = 0.0


# ---------------------------------------------------------------------------
# Global Circuit Breaker Registry
# ---------------------------------------------------------------------------

class CircuitBreakerRegistry:
    """Registry maintaining shared circuit breakers across backend subsystems."""

    _breakers: Dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 30.0,
    ) -> CircuitBreaker:
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout_sec=recovery_timeout_sec,
                )
            return cls._breakers[name]

    get_breaker = get

    @classmethod
    def get_all_metrics(cls) -> Dict[str, Any]:
        with cls._lock:
            return {name: cb.get_metrics() for name, cb in cls._breakers.items()}

    @classmethod
    def reset_all(cls):
        with cls._lock:
            for cb in cls._breakers.values():
                cb.reset()


global_circuit_breaker_registry = CircuitBreakerRegistry
