"""LangGraph tracing.

LangGraph runs its graph through LangChain's callback manager, so the handler
in :mod:`cognipeer_observability.langchain` already sees every node, model,
tool and retriever call. This module adds the graph-specific ergonomics::

    from cognipeer_observability.langgraph import graph_config

    result = agent.invoke(
        {"messages": [("user", "book me a flight")]},
        config=graph_config("conv-42"),
    )

Three LangGraph behaviours shape how a run appears in the Console, and all
three are handled for you:

**One conversation is many sessions.** Every ``invoke`` — including every
resume after an interrupt — starts a fresh root run with a new trace id.
Nothing but ``thread_id`` ties them together, so the thread id is what
:func:`graph_config` plumbs, and the Console's Threads view is where a whole
conversation is read back.

**Interrupts are not failures.** ``interrupt()`` raises a ``GraphInterrupt``
that surfaces as a chain error. The handler recognises the ``GraphBubbleUp``
family and closes the session as ``in_progress`` rather than ``error``.

**Resuming re-runs the node from the top.** Everything before the
``interrupt()`` call executes again, so a model or tool call placed before it
is genuinely made twice and is genuinely recorded twice. That is the run, not
a tracing artefact — put side effects after the interrupt if you do not want
them counted twice.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .langchain import CognipeerCallbackHandler, cognipeer_config
from .types import Agent

__all__ = ["CognipeerCallbackHandler", "cognipeer_config", "graph_config", "trace_graph"]


def graph_config(
    thread_id: str,
    *,
    agent: Optional[Agent] = None,
    base: Optional[Dict[str, Any]] = None,
    **handler_options: Any,
) -> Dict[str, Any]:
    """``RunnableConfig`` with tracing attached and the thread id plumbed.

    ``thread_id`` is written to ``configurable`` (which the checkpointer needs)
    and to ``metadata`` (which is where a third-party callback handler can
    still read it — langchain-core 1.3 stopped copying ``configurable`` across
    for anything but LangSmith's own tracer).
    """
    return cognipeer_config(thread_id, base=base, agent=agent, **handler_options)


#: Entry points that start a run and therefore need their own handler.
_TRACED_METHODS = ("invoke", "ainvoke", "stream", "astream", "astream_events", "batch", "abatch")


def trace_graph(
    graph: Any,
    thread_id: Optional[str] = None,
    *,
    agent: Optional[Agent] = None,
    **handler_options: Any,
) -> Any:
    """Wrap a compiled graph so every invocation is traced.

    ```python
    traced = trace_graph(agent_graph, "conv-42")
    traced.invoke({"messages": [("user", "hi")]})
    ```

    A **fresh handler per call**, which is why this is a wrapper rather than a
    `with_config` binding: a handler carries per-run state (its open session,
    its span map), so one shared instance would interleave concurrent
    invocations into the same session. Everything else — ``get_state``,
    ``update_state``, ``get_graph`` — passes straight through.

    Omit ``thread_id`` to take it from each call's own
    ``configurable.thread_id``, which is what a server handling many
    conversations wants.
    """
    return _TracedGraph(graph, thread_id, agent, handler_options)


class _TracedGraph:
    """Attribute-forwarding wrapper that injects a handler per invocation."""

    def __init__(
        self,
        graph: Any,
        thread_id: Optional[str],
        agent: Optional[Agent],
        handler_options: Dict[str, Any],
    ) -> None:
        self.__graph = graph
        self.__thread_id = thread_id
        self.__agent = agent
        self.__handler_options = handler_options

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self.__graph, name)
        if name not in _TRACED_METHODS or not callable(attribute):
            return attribute

        def traced(*args: Any, **kwargs: Any) -> Any:
            config = kwargs.get("config")
            if config is None and len(args) > 1 and isinstance(args[1], dict):
                config, args = args[1], (args[0], *args[2:])
            kwargs["config"] = self.__with_handler(config)
            return attribute(*args, **kwargs)

        return traced

    def __with_handler(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(config or {})
        thread_id = self.__thread_id or (merged.get("configurable") or {}).get("thread_id")
        return cognipeer_config(
            thread_id,
            base=merged,
            agent=self.__agent,
            **self.__handler_options,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<traced {self.__graph!r}>"
