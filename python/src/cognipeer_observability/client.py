"""The client every integration hangs off.

Most users never construct one: :func:`init` installs a process-wide default
and the framework integrations pick it up. An explicit client exists for
multi-tenant servers that ship traces to more than one Console project.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import weakref
from typing import Any, Dict, Iterator, List, Optional, Set

from ._config import Config, resolve_config
from ._transport import Transport
from .session import TraceSession
from .types import Agent, CaptureMode, Mode


class Cognipeer:
    """Owns configuration, the transport and the set of live sessions."""

    def __init__(self, **options: Any) -> None:
        self.config: Config = resolve_config(**options)
        self._transport = Transport(self.config)
        self._live: Set[TraceSession] = set()
        self._lock = threading.Lock()
        if self.config.enabled:
            _register_for_exit(self)

    @property
    def enabled(self) -> bool:
        """True when traces are actually being shipped."""
        return self.config.enabled

    def start_session(self, **options: Any) -> TraceSession:
        """Open a session. Remember to ``end()`` it — that is what delivers it."""
        session = TraceSession(self.config, self._transport, self._release, **options)
        with self._lock:
            self._live.add(session)
        return session

    @contextlib.contextmanager
    def trace(self, **options: Any) -> Iterator[TraceSession]:
        """Session context manager.

        On exception the session is marked ``error`` and the exception
        propagates untouched — tracing never swallows application errors.
        """
        session = self.start_session(**options)
        try:
            yield session
        except BaseException as error:
            session.end(status="error", error=error)
            raise
        else:
            session.end()

    def flush(self, timeout: Optional[float] = None) -> None:
        """Await delivery of everything queued so far."""
        self._transport.flush(timeout)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Close every still-open session, then flush."""
        _deregister_for_exit(self)
        with self._lock:
            sessions: List[TraceSession] = list(self._live)
        for session in sessions:
            session.end()
        self.flush(timeout)

    def get_transport(self) -> Transport:
        """Escape hatch for the OpenTelemetry exporter and custom senders."""
        return self._transport

    def _release(self, session: TraceSession) -> None:
        with self._lock:
            self._live.discard(session)


#: Clients whose sessions should be delivered at interpreter exit.
#:
#: Held weakly so that registering here never keeps a discarded client — or its
#: config closures and transport — alive; a client that is garbage before exit
#: had nothing worth delivering anyway.
_exit_clients: "weakref.WeakSet[Cognipeer]" = weakref.WeakSet()
_exit_hook_installed = False
_exit_lock = threading.Lock()


def _register_for_exit(client: "Cognipeer") -> None:
    global _exit_hook_installed
    with _exit_lock:
        _exit_clients.add(client)
        if _exit_hook_installed:
            return
        _exit_hook_installed = True
    atexit.register(_flush_at_exit)


def _deregister_for_exit(client: "Cognipeer") -> None:
    with _exit_lock:
        _exit_clients.discard(client)


def _flush_at_exit() -> None:
    """End open sessions at interpreter exit, then let the queue drain.

    Flushing the dispatcher queue alone was not enough: a script that exits
    with a session still open never enqueued that session's payload in the
    first place, so the whole run was lost. Bounded, because an exit hook that
    can hang a process is worse than a dropped trace.
    """
    with _exit_lock:
        clients = list(_exit_clients)
    for client in clients:
        try:
            client.shutdown(timeout=5.0)
        except Exception:  # pragma: no cover - interpreter teardown
            pass


_default: Optional[Cognipeer] = None
_default_lock = threading.Lock()


def init(
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
    on_error: Optional[Any] = None,
) -> Cognipeer:
    """Configure the process-wide client.

    Call once at startup, before creating agents. Calling it again replaces the
    client (the previous one is flushed) — convenient in tests.
    """
    global _default
    with _default_lock:
        previous = _default
        _default = Cognipeer(
            api_key=api_key,
            base_url=base_url,
            agent=agent,
            metadata=metadata,
            thread_id=thread_id,
            enabled=enabled,
            capture=capture,
            redact_patterns=redact_patterns,
            max_content_chars=max_content_chars,
            mode=mode,
            stream_after_ms=stream_after_ms,
            stream_after_events=stream_after_events,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            debug=debug,
            on_error=on_error,
        )
    if previous is not None:
        previous.shutdown(timeout=2.0)
    return _default


def get_client() -> Cognipeer:
    """The process-wide client, built from environment variables on first use.

    With no ``COGNIPEER_API_KEY`` present this returns a disabled client, so
    integrations can be wired up unconditionally.
    """
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Cognipeer()
    return _default


def reset_client() -> None:
    """Drop the process-wide client. Intended for tests."""
    global _default
    with _default_lock:
        _default = None
