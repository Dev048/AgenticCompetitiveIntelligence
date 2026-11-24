"""Structured logging configuration with context propagation and secret redaction."""

from contextvars import ContextVar
import logging
import re
import sys
from typing import Any

# Sensitive key patterns that should be masked
SENSITIVE_KEY_PATTERNS = re.compile(
    r"(api_?key|password|secret|auth|bearer|token|credentials|private_?key)",
    re.IGNORECASE,
)

# Secret value patterns (e.g. sk-..., bearer tokens)
SECRET_VALUE_PATTERNS = re.compile(
    r"(sk-[a-zA-Z0-9_\-]{16,}|bearer\s+[a-zA-Z0-9_\-\.]+)",
    re.IGNORECASE,
)

# ContextVar for environments without structlog or as universal fallback
_local_context: ContextVar[dict[str, Any]] = ContextVar("local_log_context", default={})

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def redact_sensitive_value(val: Any) -> Any:
    """Scrub secrets, truncate long payloads, and mask credentials."""
    if isinstance(val, str):
        # Mask literal secrets matching patterns
        redacted = SECRET_VALUE_PATTERNS.sub("[REDACTED]", val)
        # Truncate overly long documents / raw dumps
        if len(redacted) > 500 and not redacted.startswith("{"):
            return redacted[:497] + "... [TRUNCATED]"
        return redacted
    elif isinstance(val, dict):
        return {
            k: ("[REDACTED]" if SENSITIVE_KEY_PATTERNS.search(str(k)) else redact_sensitive_value(v))
            for k, v in val.items()
        }
    elif isinstance(val, (list, tuple)):
        return [redact_sensitive_value(item) for item in val]
    return val



def redact_sensitive_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that scrubs sensitive fields and masks API keys."""
    redacted_dict: dict[str, Any] = {}
    for key, val in event_dict.items():
        if SENSITIVE_KEY_PATTERNS.search(str(key)):
            redacted_dict[key] = "[REDACTED]"
        else:
            redacted_dict[key] = redact_sensitive_value(val)
    return redacted_dict


def bind_log_context(**kwargs: Any) -> None:
    """Bind contextual attributes (correlation_id, workspace_id, run_id) to current async context."""
    redacted_kwargs = {k: redact_sensitive_value(v) for k, v in kwargs.items()}
    if HAS_STRUCTLOG:
        structlog.contextvars.bind_contextvars(**redacted_kwargs)
    ctx = dict(_local_context.get())
    ctx.update(redacted_kwargs)
    _local_context.set(ctx)


def clear_log_context() -> None:
    """Clear all context variables for current async task."""
    if HAS_STRUCTLOG:
        structlog.contextvars.clear_contextvars()
    _local_context.set({})


def get_log_context() -> dict[str, Any]:
    """Retrieve active log context for current async task."""
    if HAS_STRUCTLOG:
        return dict(structlog.contextvars.get_contextvars())
    return dict(_local_context.get())



def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structured or standard library logging with sensitive data redaction."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    use_json = log_format.lower() == "json" or (log_level.upper() != "DEBUG" and log_format.lower() != "console")
    renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()

    if HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                redact_sensitive_processor,
                renderer,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )


class _StandardLoggerAdapter:
    """Wraps standard logging.Logger to handle structured kwargs gracefully."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _format(self, msg: str, kwargs: dict[str, Any]) -> str:
        ctx = _local_context.get()
        merged = {**ctx, **kwargs}
        redacted = {
            k: ("[REDACTED]" if SENSITIVE_KEY_PATTERNS.search(k) else redact_sensitive_value(v))
            for k, v in merged.items()
        }
        extra_str = " ".join(f"{k}={v}" for k, v in redacted.items())
        return f"{msg} {extra_str}".strip() if extra_str else msg

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(self._format(msg, kwargs), *args)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(self._format(msg, kwargs), *args)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(self._format(msg, kwargs), *args)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        exc_info = kwargs.pop("exc_info", None)
        self._logger.error(self._format(msg, kwargs), *args, exc_info=exc_info)


def get_logger(name: str | None = None) -> Any:
    """Get a structured logger instance or standard logger wrapper."""
    if HAS_STRUCTLOG:
        return structlog.get_logger(name or "competitive_intelligence")
    return _StandardLoggerAdapter(logging.getLogger(name or "competitive_intelligence"))
