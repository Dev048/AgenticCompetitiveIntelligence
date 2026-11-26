"""Public demo safeguards: read-only request guard and query rate limiting.

Both are inert unless ``DEMO_MODE`` is enabled. They are deterministic, in-process and
dependency-free on purpose: the public demo runs as a single container, so no external
rate-limit store is justified (see AGENTS.md "No Premature Infrastructure").
"""

import re
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from competitive_intelligence.application.api.v1.schemas.common import (
    ErrorDetail,
    StandardErrorResponse,
)
from competitive_intelligence.core.config import Settings, get_settings
from competitive_intelligence.core.errors import RateLimitExceededError
from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DEMO_READ_ONLY_ERROR_CODE = "DEMO_READ_ONLY"
DEMO_READ_ONLY_MESSAGE = (
    "This is a read-only public demo. Creating workspaces, competitors or sources, "
    "crawling, and running workflows are disabled here. Run the project locally to try them."
)


def _query_route(api_prefix: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(api_prefix)}/workspaces/[^/]+/query/?$")


def _review_route(api_prefix: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(api_prefix)}/workspaces/[^/]+/recommendations/[^/]+/status/?$")


def is_write_allowed_in_demo(method: str, path: str, settings: Settings) -> bool:
    """Return True when a request may pass the demo read-only guard.

    Safe methods always pass. Every mutating request is rejected, on any path, unless it
    is one of exactly two allow-listed operations:
    - ``POST {api}/workspaces/{id}/query``: evidence query (rate-limited separately).
    - ``PATCH {api}/workspaces/{id}/recommendations/{id}/status``: human operator review
      of a PROPOSED recommendation (disable with DEMO_ALLOW_RECOMMENDATION_REVIEW=false).
    """
    verb = method.upper()
    if verb in SAFE_METHODS:
        return True
    if verb == "POST" and _query_route(settings.api_v1_str).match(path):
        return True
    return (
        settings.demo_allow_recommendation_review
        and verb == "PATCH"
        and _review_route(settings.api_v1_str).match(path) is not None
    )


def demo_read_only_response(correlation_id: str | None) -> JSONResponse:
    body = StandardErrorResponse(
        error_code=DEMO_READ_ONLY_ERROR_CODE,
        message=DEMO_READ_ONLY_MESSAGE,
        details=[
            ErrorDetail(
                message="Allowed in the demo: browsing, evidence queries, and accepting or "
                "rejecting proposed recommendations.",
                code="demo_allow_list",
            )
        ],
        correlation_id=correlation_id,
    )
    return JSONResponse(status_code=403, content=body.model_dump())


class DemoReadOnlyMiddleware(BaseHTTPMiddleware):
    """Rejects every mutating request when DEMO_MODE is on, except the allow-list.

    This blocks workspace/competitor/source creation and updates, crawl triggers, workflow
    runs/resumes/cancels and decision-context creation (SSRF and quota-abuse vectors).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        settings = get_settings()
        if settings.demo_mode and not is_write_allowed_in_demo(
            request.method, request.url.path, settings
        ):
            logger.info(
                "Demo read-only guard blocked request",
                method=request.method,
                path=request.url.path,
            )
            return demo_read_only_response(getattr(request.state, "correlation_id", None))
        return await call_next(request)


class Clock:
    """Time source (injectable for deterministic tests)."""

    def monotonic(self) -> float:
        return time.monotonic()

    def utc_now(self) -> datetime:
        return datetime.now(UTC)


class QueryRateLimiter:
    """Per-client sliding 60-second window plus a global daily cap (resets at UTC midnight).

    Only accepted requests count towards either limit.
    """

    WINDOW_SECONDS = 60.0
    _MAX_TRACKED_CLIENTS = 10_000

    def __init__(
        self,
        per_client_per_minute: int,
        daily_cap: int,
        clock: Clock | None = None,
    ) -> None:
        if per_client_per_minute < 1 or daily_cap < 1:
            raise ValueError("Rate limits must be positive")
        self.per_client_per_minute = per_client_per_minute
        self.daily_cap = daily_cap
        self._clock = clock or Clock()
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = {}
        self._day = self._clock.utc_now().date()
        self._daily_count = 0

    def check(self, client_key: str) -> None:
        """Record one request for ``client_key`` or raise RateLimitExceededError."""
        with self._lock:
            now = self._clock.monotonic()
            utc_now = self._clock.utc_now()
            if utc_now.date() != self._day:
                self._day = utc_now.date()
                self._daily_count = 0

            if self._daily_count >= self.daily_cap:
                raise RateLimitExceededError(
                    "The demo's daily live-query budget is used up. Seeded pages still work; "
                    "please try a live query again tomorrow.",
                    retry_after_seconds=_seconds_until_utc_midnight(utc_now),
                    details={"scope": "global_daily"},
                )

            window = self._windows.setdefault(client_key, deque())
            while window and now - window[0] >= self.WINDOW_SECONDS:
                window.popleft()
            if len(window) >= self.per_client_per_minute:
                retry_after = max(1, int(self.WINDOW_SECONDS - (now - window[0])) + 1)
                raise RateLimitExceededError(
                    f"Too many live queries. Please wait {retry_after}s and try again.",
                    retry_after_seconds=retry_after,
                    details={"scope": "per_client"},
                )

            window.append(now)
            self._daily_count += 1
            if len(self._windows) > self._MAX_TRACKED_CLIENTS:
                self._windows = {
                    key: hits
                    for key, hits in self._windows.items()
                    if hits and now - hits[-1] < self.WINDOW_SECONDS
                }

    @property
    def remaining_today(self) -> int:
        with self._lock:
            return max(0, self.daily_cap - self._daily_count)


def _seconds_until_utc_midnight(now: datetime) -> int:
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def client_key_for(request: Request, trusted_proxy_hops: int) -> str:
    """Identify the caller for per-client limits.

    With ``trusted_proxy_hops`` reverse proxies in front of the app (Hugging Face Spaces
    has one), each proxy appends the address it received the request from to
    X-Forwarded-For. Entries to the left of those are client-supplied and spoofable, so the
    caller is the entry ``trusted_proxy_hops`` positions from the right. With no trusted
    proxies, the direct peer address is used.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [part.strip() for part in forwarded.split(",") if part.strip()]
    if len(hops) >= trusted_proxy_hops:
        return hops[-trusted_proxy_hops]
    return peer


_limiter: QueryRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_query_rate_limiter() -> QueryRateLimiter:
    """Process-wide limiter built from settings on first use."""
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            settings = get_settings()
            _limiter = QueryRateLimiter(
                per_client_per_minute=settings.demo_query_per_ip_per_minute,
                daily_cap=settings.demo_query_daily_cap,
            )
        return _limiter


def reset_query_rate_limiter() -> None:
    global _limiter
    with _limiter_lock:
        _limiter = None


async def enforce_query_rate_limit(request: Request) -> None:
    """FastAPI dependency: apply the query rate limit when DEMO_MODE is enabled."""
    settings = get_settings()
    if not settings.demo_mode:
        return
    get_query_rate_limiter().check(client_key_for(request, settings.demo_trusted_proxy_hops))
