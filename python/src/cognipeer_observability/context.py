"""Ambient session/span context.

Framework integrations receive their own handles, but manual instrumentation
(:func:`~cognipeer_observability.decorators.observe`) and cross-cutting helpers
need to know "which run am I inside right now". ``contextvars`` is the right
primitive: it follows ``await`` boundaries and each ``asyncio`` task gets its
own copy, so two concurrent agent runs in one process never mix up their spans.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar, Token
from typing import Iterator, Optional

from .session import TraceSession

_current_session: ContextVar[Optional[TraceSession]] = ContextVar(
    "cognipeer_current_session", default=None
)
_current_span: ContextVar[Optional[str]] = ContextVar(
    "cognipeer_current_span", default=None
)


def get_current_session() -> Optional[TraceSession]:
    """The session enclosing the current execution, if any."""
    return _current_session.get()


def get_current_span_key() -> Optional[str]:
    """The span key enclosing the current execution — used as a parent key."""
    return _current_span.get()


def set_current_session(session: Optional[TraceSession]) -> Token:
    """Bind a session to this context. Reset with :func:`reset_session`."""
    return _current_session.set(session)


def reset_session(token: Token) -> None:
    _current_session.reset(token)


@contextlib.contextmanager
def use_session(session: TraceSession) -> Iterator[TraceSession]:
    """Bind ``session`` for the duration of the block."""
    token = _current_session.set(session)
    try:
        yield session
    finally:
        _current_session.reset(token)


@contextlib.contextmanager
def use_span(key: Optional[str]) -> Iterator[None]:
    """Bind ``key`` as the parent span for nested work inside the block."""
    token = _current_span.set(key)
    try:
        yield
    finally:
        _current_span.reset(token)
