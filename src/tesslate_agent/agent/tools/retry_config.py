"""
Multi-layer retry strategy for tool execution.

Implements automatic retry logic using the ``tenacity`` library to handle
transient failures without wasting LLM tokens or user time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# Transient errors that should be retried automatically.
# Note: we explicitly exclude FileNotFoundError and PermissionError even
# though they are subclasses of IOError/OSError, as they indicate permanent
# problems.
_RETRYABLE_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,  # Network issues
    TimeoutError,  # Request timeouts
)

# Non-retryable errors even if they're subclasses of retryable ones.
_NON_RETRYABLE_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    PermissionError,
    NotADirectoryError,
    IsADirectoryError,
)

# Backwards-compatible alias used by consumers.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = _RETRYABLE_EXCEPTION_TYPES

# Non-retryable errors that indicate configuration or logic problems.
NON_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
) + _NON_RETRYABLE_EXCEPTION_TYPES


def _should_retry_exception(exception: BaseException) -> bool:
    """
    Return ``True`` when ``exception`` should trigger a retry.

    Explicit non-retryable types win even if the exception is also a
    subclass of a retryable type (e.g. ``FileNotFoundError`` is an
    ``OSError`` but should never be retried).
    """
    if isinstance(exception, _NON_RETRYABLE_EXCEPTION_TYPES):
        return False

    if isinstance(exception, OSError):
        return not isinstance(exception, _NON_RETRYABLE_EXCEPTION_TYPES)

    return isinstance(exception, _RETRYABLE_EXCEPTION_TYPES)


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
) -> Callable:
    """
    Build a tenacity retry decorator for tool execution.

    Uses exponential backoff bounded by ``min_wait`` and ``max_wait``.

    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time in seconds between retries.
        max_wait: Maximum wait time in seconds between retries.
        exponential_base: Exponential backoff base.

    Returns:
        A decorator that can be applied to async callables.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=1, min=min_wait, max=max_wait, exp_base=exponential_base
        ),
        retry=retry_if_exception(_should_retry_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Default retry decorator for tool executors.
tool_retry = create_retry_decorator(max_attempts=3, min_wait=1.0, max_wait=10.0)

# Aggressive retry for critical operations (e.g. database writes).
tool_retry_aggressive = create_retry_decorator(max_attempts=5, min_wait=0.5, max_wait=15.0)

# Gentle retry for less critical operations.
tool_retry_gentle = create_retry_decorator(max_attempts=2, min_wait=2.0, max_wait=5.0)


def is_retryable_error(exception: BaseException) -> bool:
    """Return ``True`` when ``exception`` would trigger a retry."""
    return _should_retry_exception(exception)


def create_custom_retry(
    retryable_exceptions: tuple[type[BaseException], ...] = RETRYABLE_EXCEPTIONS,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Callable:
    """
    Build a retry decorator that retries on specific exception types.

    Unlike :func:`create_retry_decorator`, this variant does NOT filter
    out ``FileNotFoundError`` / ``PermissionError`` — it retries strictly
    on the types supplied by the caller.

    Args:
        retryable_exceptions: Exception types to retry on.
        max_attempts: Maximum retry attempts.
        min_wait: Minimum wait time in seconds.
        max_wait: Maximum wait time in seconds.

    Returns:
        A decorator that can be applied to async callables.
    """
    from tenacity import retry_if_exception_type

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retryable_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
