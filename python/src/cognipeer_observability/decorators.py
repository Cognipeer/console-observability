"""Manual instrumentation: ``@observe`` and ``trace()``.

This is the escape hatch for code no framework integration covers — a bespoke
orchestration loop, a retrieval helper, a business rule you want to see in the
timeline next to the model calls.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, cast

from ._ids import new_span_id
from .client import get_client
from .context import get_current_session, get_current_span_key, use_session, use_span
from .session import TraceSession
from .types import Agent, Section

F = TypeVar("F", bound=Callable[..., Any])


@contextlib.contextmanager
def trace(
    *,
    name: Optional[str] = None,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    agent: Optional[Agent] = None,
    session_config: Optional[Dict[str, Any]] = None,
    **options: Any,
) -> Iterator[TraceSession]:
    """Open a session and bind it as the ambient one for the block.

    ``name`` is a shorthand for ``agent={"name": name}`` — it is what the
    Console groups runs by on the Agents screen.
    """
    merged_agent: Agent = dict(agent or {})  # type: ignore[assignment]
    if name and not merged_agent.get("name"):
        merged_agent["name"] = name  # type: ignore[typeddict-item]

    client = get_client()
    session = client.start_session(
        session_id=session_id,
        thread_id=thread_id,
        agent=merged_agent or None,
        session_config=session_config,
        **options,
    )
    try:
        # `use_span(None)` matters as much as `use_session`: without it an
        # `@observe` inside this block would parent onto whatever span was
        # open OUTSIDE the new session, producing an event whose parent lives
        # in a different session.
        with use_session(session), use_span(None):
            yield session
    except BaseException as error:
        session.end(status="error", error=error)
        raise
    else:
        session.end()


def observe(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    type: str = "span",  # noqa: A002 - mirrors the wire field name
    capture_input: bool = True,
    capture_output: bool = True,
    tool_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Record one event per call of the decorated function.

    Works on sync, async, generator and async-generator functions. Nesting is
    automatic: a decorated function called from inside another one becomes its
    child span in the Console timeline.

    When no session is active a root session is created for the outermost
    call and closed when it returns, so a single decorated entry point is
    enough to get a complete trace.

    ``@observe(type="tool_call", tool_name="search")`` makes the event render
    as a tool invocation rather than a generic span.
    """

    def decorate(func: F) -> F:
        # `__name__`, not `__qualname__`: a qualified name drags in `<locals>`
        # and the enclosing class or function, which makes a noisy timeline
        # label for no gain.
        label = name or getattr(func, "__name__", "observed")

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _Observation(
                    label, type, tool_name, metadata, capture_input, capture_output, args, kwargs,
                    scoped_span=True,
                ) as obs:
                    chunks: List[Any] = []
                    inner = func(*args, **kwargs)
                    while True:
                        # The span is bound only while the generator BODY runs,
                        # never across a `yield`. A generator shares its
                        # caller's context, so leaving the span bound while
                        # suspended would make the caller's next `observe()`
                        # a child of this generator.
                        with obs.bound():
                            try:
                                item = await inner.__anext__()
                            except StopAsyncIteration:
                                break
                        if capture_output:
                            chunks.append(item)
                        yield item
                    obs.set_result(chunks)

            return cast(F, async_gen_wrapper)

        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _Observation(
                    label, type, tool_name, metadata, capture_input, capture_output, args, kwargs,
                    scoped_span=True,
                ) as obs:
                    chunks: List[Any] = []
                    inner = func(*args, **kwargs)
                    while True:
                        with obs.bound():  # see the async note above
                            try:
                                item = next(inner)
                            except StopIteration:
                                break
                        # Buffering every item to record it would hold a whole
                        # stream in memory when the caller does not want it.
                        if capture_output:
                            chunks.append(item)
                        yield item
                    obs.set_result(chunks)

            return cast(F, gen_wrapper)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _Observation(
                    label, type, tool_name, metadata, capture_input, capture_output, args, kwargs
                ) as obs:
                    result = await func(*args, **kwargs)
                    obs.set_result(result)
                    return result

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _Observation(
                label, type, tool_name, metadata, capture_input, capture_output, args, kwargs
            ) as obs:
                result = func(*args, **kwargs)
                obs.set_result(result)
                return result

        return cast(F, sync_wrapper)

    if _func is not None:
        return decorate(_func)
    return decorate


class _Observation:
    """One decorated call: opens a span, closes it, owns the implicit session."""

    def __init__(
        self,
        label: str,
        event_type: str,
        tool_name: Optional[str],
        metadata: Optional[Dict[str, Any]],
        capture_input: bool,
        capture_output: bool,
        args: Any,
        kwargs: Any,
        *,
        scoped_span: bool = False,
    ) -> None:
        self._label = label
        self._type = event_type
        self._tool_name = tool_name
        self._metadata = metadata
        self._capture_output = capture_output
        self._args = args if capture_input else None
        self._kwargs = kwargs if capture_input else None
        self._result: Any = None
        self._has_result = False

        self._session: Optional[TraceSession] = None
        self._owns_session = False
        self._key = new_span_id()
        self._session_ctx: Any = None
        self._span_ctx: Any = None
        #: Generators bind their span per-block instead of for the whole call.
        self._scoped_span = scoped_span
        self._span_bound = False

    def set_result(self, result: Any) -> None:
        self._result = result
        self._has_result = True

    def __enter__(self) -> "_Observation":
        session = get_current_session()
        if session is None:
            # A decorated entry point with no enclosing session gets one, so a
            # single @observe is enough to produce a complete trace.
            session = get_client().start_session(agent={"name": self._label})
            self._owns_session = True
            self._session_ctx = use_session(session)
            self._session_ctx.__enter__()
        self._session = session

        sections: List[Section] = []
        if self._args is not None or self._kwargs is not None:
            payload: Dict[str, Any] = {}
            if self._args:
                payload["args"] = list(self._args)
            if self._kwargs:
                payload["kwargs"] = self._kwargs
            if payload:
                sections.append({"kind": "message", "label": "Input", "content": payload})

        session.open_span(
            self._key,
            type=self._type,
            label=self._label,
            parent_key=get_current_span_key(),
            tool_name=self._tool_name,
            sections=sections,
            metadata=self._metadata,
        )
        if not self._scoped_span:
            self._span_ctx = use_span(self._key)
            self._span_ctx.__enter__()
            self._span_bound = True
        return self

    @contextlib.contextmanager
    def bound(self) -> Iterator[None]:
        """Bind this span for the duration of the block.

        Used by the generator wrappers, which unbind across `yield`: a
        generator shares its caller's context, so a span left bound while the
        generator is suspended would silently adopt the caller's next span as
        a child.
        """
        if self._span_bound:
            # Already bound for the whole call (the sync/async wrappers).
            yield
            return
        with use_span(self._key):
            yield

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._span_ctx is not None:
            self._span_ctx.__exit__(None, None, None)

        sections: List[Section] = []
        if self._capture_output and self._has_result and self._result is not None:
            sections.append({"kind": "message", "label": "Output", "content": self._result})

        # A caller that stops consuming a generator gets GeneratorExit thrown
        # at the yield. That is an ordinary early exit — recording it as a
        # failed step would mark every `break` over a stream as an error.
        abandoned = isinstance(exc, GeneratorExit)
        failed = exc is not None and not abandoned

        if self._session is not None:
            self._session.close_span(
                self._key,
                sections=sections,
                status="error" if failed else "success",
                error=exc if failed else None,
                metadata={"abandoned": True} if abandoned else None,
            )

        if self._owns_session and self._session is not None:
            if self._session_ctx is not None:
                self._session_ctx.__exit__(None, None, None)
            self._session.end(
                status="error" if failed else "success", error=exc if failed else None
            )
