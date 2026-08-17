"""Configuration resolution: explicit options first, environment second."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern

from .types import Agent, CaptureMode, Mode

ENV_API_KEY = "COGNIPEER_API_KEY"
ENV_BASE_URL = "COGNIPEER_BASE_URL"
ENV_AGENT_NAME = "COGNIPEER_AGENT_NAME"
ENV_AGENT_VERSION = "COGNIPEER_AGENT_VERSION"
ENV_ENABLED = "COGNIPEER_TRACING_ENABLED"
ENV_CAPTURE = "COGNIPEER_CAPTURE_CONTENT"
ENV_DEBUG = "COGNIPEER_DEBUG"
ENV_MODE = "COGNIPEER_TRACING_MODE"

DEFAULT_BASE_URL = "https://console.cognipeer.com"

logger = logging.getLogger("cognipeer")

_FALSEY = {"0", "false", "no", "off"}


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() not in _FALSEY


def normalize_base_url(value: str) -> str:
    """Normalize a base URL to the host root.

    Accepts both the host (``https://console.cognipeer.com``) and the legacy
    fully-qualified ingest path (``.../api/client/v1``) that other Cognipeer
    SDKs ask for.
    """
    return re.sub(r"/api/client/v1$", "", value.strip().rstrip("/"))


def _resolve_capture(value: Optional[str]) -> CaptureMode:
    raw = (value or "").lower()
    if raw in {"none", "off", "false"}:
        return "none"
    if raw in {"metadata", "meta", "structure"}:
        return "metadata"
    return "all"


def _resolve_mode(value: Optional[str]) -> Optional[Mode]:
    if value in {"auto", "stream", "batch"}:
        return value  # type: ignore[return-value]
    return None


@dataclass
class Config:
    """Fully resolved settings shared by the client, sessions and transport."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    agent: Agent = field(default_factory=dict)  # type: ignore[arg-type]
    #: Default attribution tags stamped on every session; see
    #: ``types.SessionPayload.metadata``. Per-session values are merged on
    #: top, same as ``agent``.
    metadata: Dict[str, str] = field(default_factory=dict)
    thread_id: Optional[str] = None
    enabled: bool = True
    capture: CaptureMode = "all"
    redact_patterns: List[Pattern[str]] = field(default_factory=list)
    max_content_chars: int = 50_000
    mode: Mode = "auto"
    stream_after_ms: int = 2_000
    stream_after_events: int = 25
    timeout: float = 30.0
    max_retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    debug: bool = False
    on_error: Optional[Callable[[BaseException], None]] = None

    def report_error(self, error: BaseException) -> None:
        """Never raise out of the exporter — observability must not break the app.

        The user's ``on_error`` is arbitrary code (often a Sentry call that can
        itself raise before it is initialised), and some call sites run on the
        traced application's own stack rather than the dispatcher thread, so a
        raising handler must not propagate.
        """
        try:
            if self.on_error is not None:
                self.on_error(error)
            else:
                logger.warning("cognipeer: trace export failed: %s", error)
        except Exception:  # pragma: no cover - the reporter itself is broken
            pass


def resolve_config(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    agent: Optional[Agent] = None,
    metadata: Optional[Dict[str, str]] = None,
    thread_id: Optional[str] = None,
    enabled: Optional[bool] = None,
    capture: Optional[CaptureMode] = None,
    redact_patterns: Optional[List[Any]] = None,
    max_content_chars: Optional[int] = None,
    mode: Optional[Mode] = None,
    stream_after_ms: Optional[int] = None,
    stream_after_events: Optional[int] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
    debug: Optional[bool] = None,
    on_error: Optional[Callable[[BaseException], None]] = None,
) -> Config:
    """Merge explicit options with environment defaults.

    A missing API key is deliberately not an error: it disables the exporter
    and warns once. An observability SDK that raises at import time in
    production is worse than one that stays quiet.
    """
    resolved_debug = debug if debug is not None else _env_bool(ENV_DEBUG, False)
    resolved_key = api_key or _env(ENV_API_KEY) or ""
    resolved_capture = capture or _resolve_capture(_env(ENV_CAPTURE))
    configured_enabled = enabled if enabled is not None else _env_bool(ENV_ENABLED, True)
    is_enabled = configured_enabled and resolved_capture != "none" and bool(resolved_key)

    if configured_enabled and resolved_capture != "none" and not resolved_key:
        logger.warning(
            "cognipeer: no API key — tracing disabled. Set %s or pass api_key= to init().",
            ENV_API_KEY,
        )

    merged_agent: Agent = dict(agent or {})  # type: ignore[assignment]
    if not merged_agent.get("name") and _env(ENV_AGENT_NAME):
        merged_agent["name"] = _env(ENV_AGENT_NAME)  # type: ignore[typeddict-item]
    if not merged_agent.get("version") and _env(ENV_AGENT_VERSION):
        merged_agent["version"] = _env(ENV_AGENT_VERSION)  # type: ignore[typeddict-item]

    compiled = [
        pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
        for pattern in (redact_patterns or [])
    ]

    return Config(
        api_key=resolved_key,
        base_url=normalize_base_url(base_url or _env(ENV_BASE_URL) or DEFAULT_BASE_URL),
        agent=merged_agent,
        metadata=dict(metadata or {}),
        thread_id=thread_id,
        enabled=is_enabled,
        capture=resolved_capture,
        redact_patterns=compiled,
        max_content_chars=max_content_chars if max_content_chars is not None else 50_000,
        mode=mode or _resolve_mode(_env(ENV_MODE)) or "auto",
        stream_after_ms=stream_after_ms if stream_after_ms is not None else 2_000,
        stream_after_events=stream_after_events if stream_after_events is not None else 25,
        timeout=timeout if timeout is not None else 30.0,
        max_retries=max_retries if max_retries is not None else 3,
        headers=dict(headers or {}),
        debug=resolved_debug,
        on_error=on_error,
    )
