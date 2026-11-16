"""
Controlled retry mechanisms with exponential backoff and randomized jitter.

Prevents thundering herd problems during upstream transient outages.
Distinguishes between retryable transient network/service failures (HTTP 502/503/504, 429)
and terminal client errors (HTTP 400, 401, 403, 404, 422).
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Type

logger = logging.getLogger(__name__)

DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def calculate_backoff(
    attempt: int,
    base_delay_sec: float = 0.1,
    max_delay_sec: float = 2.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
) -> float:
    """Calculates exponential delay with optional uniform random jitter."""
    b_del = base_delay if base_delay is not None else base_delay_sec
    m_del = max_delay if max_delay is not None else max_delay_sec
    delay = min(m_del, b_del * (backoff_factor ** attempt))
    if jitter:
        delay = delay * random.uniform(0.75, 1.25)
    return max(0.001, round(delay, 4))


def execute_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    max_attempts: Optional[int] = None,
    base_delay_sec: float = 0.1,
    base_delay: Optional[float] = None,
    max_delay_sec: float = 2.0,
    max_delay: Optional[float] = None,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Sequence[Type[Exception]] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs,
) -> Any:
    """
    Executes callable with controlled retries on retryable exceptions.
    Non-retryable exceptions propagate immediately without retry delay.
    """
    eff_retries = (max_attempts - 1) if max_attempts is not None else max_retries
    eff_base_delay = base_delay if base_delay is not None else base_delay_sec
    eff_max_delay = max_delay if max_delay is not None else max_delay_sec
    last_exc: Optional[Exception] = None
    exc_tuple = tuple(retryable_exceptions)

    for attempt in range(eff_retries + 1):
        try:
            return func(*args, **kwargs)
        except exc_tuple as e:
            last_exc = e
            if attempt >= eff_retries:
                logger.warning(
                    f"Max retries ({eff_retries}) exhausted for {getattr(func, '__name__', str(func))}. Last error: {e}"
                )
                raise

            delay = calculate_backoff(
                attempt=attempt,
                base_delay_sec=eff_base_delay,
                max_delay_sec=eff_max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
            )
            if on_retry:
                on_retry(attempt + 1, e, delay)
            time.sleep(delay)

    if last_exc:
        raise last_exc


def with_retry(
    max_retries: int = 3,
    base_delay_sec: float = 0.1,
    max_delay_sec: float = 2.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Sequence[Type[Exception]] = DEFAULT_RETRYABLE_EXCEPTIONS,
):
    """Decorator to wrap synchronous functions with automatic controlled retries."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return execute_with_retry(
                fn,
                *args,
                max_retries=max_retries,
                base_delay_sec=base_delay_sec,
                max_delay_sec=max_delay_sec,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                **kwargs,
            )
        return wrapper
    return decorator
