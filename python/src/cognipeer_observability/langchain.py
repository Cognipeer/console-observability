"""LangChain / LangGraph callback handler.

One handler covers both: LangGraph runs its graph through LangChain's callback
manager, so nodes, models, tools and retrievers all arrive here. Attach it
per-invocation::

    from cognipeer_observability.langchain import CognipeerCallbackHandler

    result = agent.invoke(
        {"messages": [("user", "book me a flight")]},
        config={"callbacks": [CognipeerCallbackHandler()],
                "configurable": {"thread_id": "conv-42"}},
    )

or globally, for every chain in the process::

    from cognipeer_observability.langchain import install_langchain_tracing
    install_langchain_tracing()

**Version support.** The handler subclasses ``BaseCallbackHandler`` and reads
every payload defensively, so it works from LangChain 0.1 through 1.x. Where
newer versions expose better data it is preferred and older ones fall back:

* token usage — ``AIMessage.usage_metadata`` (0.3+, includes a cache-read
  breakdown) → ``LLMResult.llm_output["token_usage"]`` (0.1/0.2) →
  ``generation_info["usage"]``;
* model name — ``metadata["ls_model_name"]`` (0.2+) →
  ``invocation_params["model"|"model_name"]`` → ``llm_output["model_name"]``;
* thread id — ``metadata["thread_id"]``, which LangGraph populates from
  ``configurable.thread_id``, so conversations group themselves in the Console
  with no extra wiring.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Sequence, Set
from uuid import UUID

from .client import Cognipeer, get_client
from .session import TraceSession
from .types import Agent, Section, ToolDefinition

logger = logging.getLogger("cognipeer")

try:  # pragma: no cover - import guard
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError as error:  # pragma: no cover - import guard
    raise ImportError(
        "cognipeer_observability.langchain requires langchain-core. "
        "Install it with `pip install cognipeer-observability[langchain]`."
    ) from error


#: Runnables LangChain/LangGraph create for plumbing. Recording them buries the
#: model and tool calls the user actually came to look at.
NOISE_CHAIN_NAMES: Set[str] = {
    "ChannelRead",
    "ChannelWrite",
    "RunnableAssign",
    "RunnableParallel",
    "RunnablePassthrough",
    "RunnableSequence",
    "_write",
    "__start__",
    "__end__",
}

_ROLE_BY_MESSAGE_TYPE = {
    "human": "user",
    "ai": "assistant",
    "AIMessageChunk": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "chat": "assistant",
}


class CognipeerCallbackHandler(BaseCallbackHandler):
    """Turns LangChain run events into one Cognipeer tracing session.

    Args:
        client: Client to send through. Defaults to the process-wide one.
        session: Send into an existing session instead of opening one. The
            handler never ends a session it did not create.
        session_id: Fixed session id — pass the same value again to append to
            an existing run.
        thread_id: Conversation id. Usually unnecessary: LangGraph's
            ``configurable.thread_id`` is picked up automatically.
        agent: Agent identity for the session (``{"name": …, "version": …}``).
        capture_chains: Record chain/graph-node runs as spans. Plumbing
            runnables (see :data:`NOISE_CHAIN_NAMES`) are always skipped.
        metadata: Extra metadata attached to every event.
    """

    #: Never take a user's chain down because an exporter misbehaved. LangChain
    #: logs and continues when this is False.
    raise_error = False
    #: Run callbacks in the calling thread. On async runs LangChain otherwise
    #: dispatches sync handlers onto a 10-worker thread pool with a *copy* of
    #: the context and no ordering guarantee, which reorders spans and breaks
    #: the start/end pairing this handler depends on.
    run_inline = True

    def __init__(
        self,
        *,
        client: Optional[Cognipeer] = None,
        session: Optional[TraceSession] = None,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent: Optional[Agent] = None,
        capture_chains: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._client = client or get_client()
        self._session = session
        self._owns_session = session is None
        self._session_id = session_id
        self._thread_id = thread_id
        self._agent = agent
        self._capture_chains = capture_chains
        self._metadata = metadata or {}
        #: run_id of the run that opened the session — its end closes it.
        self._root_run: Optional[str] = None
        #: Model name per run, remembered from `start` for use in `end`.
        self._models: Dict[str, str] = {}
        #: Plumbing runs we chose not to record, mapped to their own parent.
        #: Children resolve through this so the tree never points at a span
        #: that was never sent.
        self._skipped: Dict[str, Optional[str]] = {}

    # ── Session plumbing ─────────────────────────────────────────────────
    @property
    def session(self) -> Optional[TraceSession]:
        """The session this handler is writing into, once one exists."""
        return self._session

    def _ensure_session(self, run_id: str, metadata: Optional[Dict[str, Any]]) -> TraceSession:
        # Remember the thread id once seen: a follow-up tool or model run
        # started outside the graph carries no `configurable`, and its session
        # should still land in the same conversation.
        self._thread_id = self._thread_id or _thread_id_from(metadata)
        # The first run seen owns the per-run bookkeeping, whether or not we
        # opened the session ourselves. With a caller-supplied session this is
        # the ONLY thing that ever arms `_root_run`, and without it the skip
        # map grows by one entry per plumbing runnable for the handler's life.
        if self._root_run is None:
            self._root_run = run_id
        if self._session is None:
            self._session = self._client.start_session(
                session_id=self._session_id,
                thread_id=self._thread_id,
                agent=self._agent,
            )
        return self._session

    def _parent_key(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        """Nearest ancestor that was actually recorded.

        Plumbing runs are skipped, so a child's literal ``parent_run_id`` can
        name a span that was never sent. Walking the skip map keeps the tree
        connected instead of orphaning the node.
        """
        key = str(parent_run_id) if parent_run_id else None
        seen: Set[str] = set()
        while key is not None and key in self._skipped and key not in seen:
            seen.add(key)
            key = self._skipped[key]
        return key

    def _maybe_end_session(self, run_id: str, status: str, error: Any = None) -> None:
        """Close the session when the run that opened it finishes.

        The handler then forgets it, so reusing one handler across several
        ``invoke()`` calls produces one session per call — which is what the
        Console's Threads view expects.

        Forgetting it matters even when ``session_id`` is fixed: an ended
        session rejects further events, so keeping the object would silently
        drop every event of the next run. A fresh object under the same id is
        what makes "append to one long-lived session" work, because the
        server's reopen path merges it and keeps the running totals.
        """
        if self._root_run != run_id:
            return

        # Per-run bookkeeping is released whoever owns the session: a handler
        # given an external one still accumulates a skip-map entry per plumbing
        # runnable per run, and nothing else ever clears it.
        self._root_run = None
        self._skipped.clear()
        self._models.clear()

        # Never end a session we did not create; the caller owns its lifetime.
        if not self._owns_session or self._session is None:
            return
        session, self._session = self._session, None
        session.end(status=status, error=error)

    def flush(self, timeout: Optional[float] = None) -> None:
        """Deliver whatever is buffered. Rarely needed — ``end`` flushes."""
        if self._session is not None:
            self._session.flush(timeout)

    # ── Chat / LLM ───────────────────────────────────────────────────────
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        session = self._ensure_session(key, metadata)
        params = kwargs.get("invocation_params") or {}
        sections: List[Section] = []
        for batch in messages or []:
            sections.extend(_messages_to_sections(batch))

        session.open_span(
            key,
            type="ai_call",
            label=_model_label(serialized, params, metadata),
            parent_key=self._parent_key(parent_run_id),
            model=_model_name(params, metadata),
            actor={"scope": "model", "name": _model_name(params, metadata) or "model"},
            sections=sections,
            metadata=self._event_metadata(tags, metadata, params),
            tool_definitions=_tool_definitions(params),
            response_format=_response_format(params),
        )
        model = _model_name(params, metadata)
        if model:
            self._models[key] = model

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        session = self._ensure_session(key, metadata)
        params = kwargs.get("invocation_params") or {}
        sections: List[Section] = [
            {"kind": "message", "label": "Prompt", "role": "user", "content": prompt}
            for prompt in prompts or []
        ]
        session.open_span(
            key,
            type="ai_call",
            label=_model_label(serialized, params, metadata),
            parent_key=self._parent_key(parent_run_id),
            model=_model_name(params, metadata),
            actor={"scope": "model", "name": _model_name(params, metadata) or "model"},
            sections=sections,
            metadata=self._event_metadata(tags, metadata, params),
            tool_definitions=_tool_definitions(params),
            response_format=_response_format(params),
        )
        model = _model_name(params, metadata)
        if model:
            self._models[key] = model

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return
        usage = _extract_usage(response)
        sections = _generations_to_sections(response)
        model = _response_model(response) or self._models.pop(key, None)
        self._session.close_span(
            key,
            status="success",
            model=model,
            sections=sections,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        self._maybe_end_session(key, "success")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return
        self._session.close_span(key, status="error", error=error)
        self._models.pop(key, None)
        self._maybe_end_session(key, "error", error)

    # ── Tools ────────────────────────────────────────────────────────────
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        session = self._ensure_session(key, metadata)
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        # `tool_call_id` correlates the run with the AIMessage tool call that
        # requested it. Passed to handlers from langchain-core 1.2.0 onwards.
        tool_call_id = kwargs.get("tool_call_id")
        session.open_span(
            key,
            type="tool_call",
            label=name,
            parent_key=self._parent_key(parent_run_id),
            tool_name=name,
            actor={"scope": "tool", "name": name},
            tool_execution_id=str(tool_call_id) if tool_call_id else None,
            sections=[
                {
                    "kind": "tool_call",
                    "label": "Arguments",
                    "tool": name,
                    "content": inputs if inputs is not None else input_str,
                }
            ],
            metadata=self._event_metadata(tags, metadata, None),
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return
        self._session.close_span(
            key,
            status="success",
            sections=[
                {"kind": "tool_result", "label": "Result", "content": _tool_output(output)}
            ],
        )
        self._maybe_end_session(key, "success")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return
        self._session.close_span(key, status="error", error=error)
        self._maybe_end_session(key, "error", error)

    # ── Retrievers ───────────────────────────────────────────────────────
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        session = self._ensure_session(key, metadata)
        name = (serialized or {}).get("name") or "retriever"
        session.open_span(
            key,
            type="retrieval",
            label=name,
            parent_key=self._parent_key(parent_run_id),
            actor={"scope": "retriever", "name": name},
            sections=[{"kind": "message", "label": "Query", "role": "user", "content": query}],
            metadata=self._event_metadata(tags, metadata, None),
        )

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return
        self._session.close_span(
            key,
            status="success",
            sections=[
                {
                    "kind": "metadata",
                    "label": f"Documents ({len(documents or [])})",
                    "content": [_document_to_dict(doc) for doc in (documents or [])],
                }
            ],
        )
        self._maybe_end_session(key, "success")

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return
        self._session.close_span(key, status="error", error=error)
        self._maybe_end_session(key, "error", error)

    # ── Chains / graph nodes ─────────────────────────────────────────────
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        name = _chain_name(serialized, metadata, kwargs)

        # The outermost chain opens the session even when chain capture is off,
        # so the run still has a root to hang model/tool events under.
        session = self._ensure_session(key, metadata)
        if not self._capture_chains or name in NOISE_CHAIN_NAMES:
            self._skipped[key] = self._parent_key(parent_run_id)
            return

        node = (metadata or {}).get("langgraph_node")
        session.open_span(
            key,
            type="span",
            label=node or name,
            parent_key=self._parent_key(parent_run_id),
            actor={"scope": "agent", "name": node or name},
            sections=[{"kind": "metadata", "label": "Input", "content": _chain_io(inputs)}],
            metadata=self._event_metadata(tags, metadata, None),
        )

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return
        if self._session.has_span(key):
            self._session.close_span(
                key,
                status="success",
                sections=[
                    {"kind": "metadata", "label": "Output", "content": _chain_io(outputs)}
                ],
            )
        self._maybe_end_session(key, "success")

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        if self._session is None:
            return

        # LangGraph signals human-in-the-loop pauses and parent commands by
        # RAISING, so a plain error mapping would mark every interrupt as a
        # failed run. Those are control flow, not failures.
        if _is_graph_control_flow(error):
            if self._session.has_span(key):
                self._session.close_span(
                    key,
                    status="success",
                    metadata={"interrupted": True, "interruptType": type(error).__name__},
                )
            self._maybe_end_session(key, "in_progress")
            return

        if self._session.has_span(key):
            self._session.close_span(key, status="error", error=error)
        self._maybe_end_session(key, "error", error)

    # ── Helpers ──────────────────────────────────────────────────────────
    def _event_metadata(
        self,
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self._metadata)
        if tags:
            out["tags"] = list(tags)
        for key in ("langgraph_node", "langgraph_step", "checkpoint_ns", "ls_provider"):
            value = (metadata or {}).get(key)
            if value is not None:
                out[key] = value
        if params:
            for key in ("temperature", "max_tokens", "top_p", "stop"):
                if params.get(key) is not None:
                    out[key] = params[key]
        return out


def _is_graph_control_flow(error: BaseException) -> bool:
    """True for LangGraph's ``GraphBubbleUp`` family (interrupt, parent command).

    Matched by class identity rather than by importing ``langgraph.errors``,
    so the handler stays usable in a LangChain-only install.
    """
    return any(
        cls.__module__.startswith("langgraph") and cls.__name__ == "GraphBubbleUp"
        for cls in type(error).__mro__
    )


def _thread_id_from(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Best-effort conversation id from run metadata.

    Up to langchain-core 1.2.x, ``ensure_config`` copied every primitive
    ``configurable`` key — including LangGraph's ``thread_id`` — into
    ``metadata``, so this worked with no extra wiring. From 1.3.0 only
    ``model`` and ``checkpoint_ns`` are copied and the rest is routed to a
    LangSmith-only channel that third-party handlers never see.

    So: read whatever is there, and let callers who need it on newer versions
    pass ``thread_id=`` to the handler or use :func:`cognipeer_config`.
    """
    meta = metadata or {}
    for key in ("thread_id", "session_id", "conversation_id"):
        value = meta.get(key)
        if value:
            return str(value)
    # 1.3.0+ parks the inheritable metadata under its own key on some paths.
    inheritable = meta.get("langsmith_inheritable_metadata")
    if isinstance(inheritable, dict) and inheritable.get("thread_id"):
        return str(inheritable["thread_id"])
    return None


def cognipeer_config(
    thread_id: Optional[str] = None,
    *,
    handler: Optional["CognipeerCallbackHandler"] = None,
    base: Optional[Dict[str, Any]] = None,
    **handler_options: Any,
) -> Dict[str, Any]:
    """Build a ``RunnableConfig`` that traces correctly on every version.

    ``thread_id`` is written to BOTH ``configurable`` (where LangGraph's
    checkpointer needs it) and ``metadata`` (where a third-party callback
    handler can still see it on langchain-core 1.3+)::

        agent.invoke(state, config=cognipeer_config("conv-42"))
    """
    config: Dict[str, Any] = {**(base or {})}
    callbacks = list(config.get("callbacks") or [])
    callbacks.append(handler or CognipeerCallbackHandler(thread_id=thread_id, **handler_options))
    config["callbacks"] = callbacks

    if thread_id:
        config["configurable"] = {**(config.get("configurable") or {}), "thread_id": thread_id}
        config["metadata"] = {**(config.get("metadata") or {}), "thread_id": thread_id}
    return config


def _model_name(params: Optional[Dict[str, Any]], metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    ls_model = (metadata or {}).get("ls_model_name")
    if isinstance(ls_model, str) and ls_model:
        return ls_model
    for key in ("model", "model_name", "model_id", "deployment_name"):
        value = (params or {}).get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _model_label(
    serialized: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> str:
    model = _model_name(params, metadata)
    if model:
        return model
    name = (serialized or {}).get("name")
    if isinstance(name, str) and name:
        return name
    ids = (serialized or {}).get("id")
    if isinstance(ids, list) and ids:
        return str(ids[-1])
    return "model"


def _chain_name(
    serialized: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> str:
    node = (metadata or {}).get("langgraph_node")
    if isinstance(node, str) and node:
        return node
    for candidate in (kwargs.get("name"), (serialized or {}).get("name")):
        if isinstance(candidate, str) and candidate:
            return candidate
    ids = (serialized or {}).get("id")
    if isinstance(ids, list) and ids:
        return str(ids[-1])
    return "chain"


def _messages_to_sections(messages: Sequence[Any]) -> List[Section]:
    """Render a LangChain message list as message / tool_call sections."""
    sections: List[Section] = []
    for message in messages or []:
        role = _message_role(message)
        content = getattr(message, "content", message)
        sections.append(
            {
                "kind": "message",
                "label": f"{role.capitalize()} message",
                "role": role,
                "content": content,
            }
        )
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            sections.append(
                {
                    "kind": "tool_call",
                    "label": f"Tool call: {name}",
                    "tool": name or "tool",
                    "content": args,
                }
            )
    return sections


def _message_role(message: Any) -> str:
    explicit = getattr(message, "type", None)
    if isinstance(explicit, str) and explicit in _ROLE_BY_MESSAGE_TYPE:
        return _ROLE_BY_MESSAGE_TYPE[explicit]
    if isinstance(message, dict):
        role = message.get("role") or message.get("type")
        if isinstance(role, str):
            return _ROLE_BY_MESSAGE_TYPE.get(role, role)
    class_name = type(message).__name__
    if class_name.startswith("Human"):
        return "user"
    if class_name.startswith("AI"):
        return "assistant"
    if class_name.startswith("System"):
        return "system"
    if class_name.startswith("Tool") or class_name.startswith("Function"):
        return "tool"
    return "user"


def _generations_to_sections(response: Any) -> List[Section]:
    sections: List[Section] = []
    for batch in getattr(response, "generations", None) or []:
        for generation in batch or []:
            message = getattr(generation, "message", None)
            if message is not None:
                sections.extend(_messages_to_sections([message]))
                continue
            text = getattr(generation, "text", None)
            if text:
                sections.append(
                    {
                        "kind": "message",
                        "label": "Assistant message",
                        "role": "assistant",
                        "content": text,
                    }
                )
    return sections


def _extract_usage(response: Any) -> Dict[str, Optional[int]]:
    """Token usage, newest source first.

    ``usage_metadata`` (LangChain 0.3+) is normalized across providers and is
    the only source carrying the cache-read breakdown. Note its
    ``input_tokens`` already INCLUDES cached tokens — which is exactly what the
    Console's pricing expects, since it treats cached tokens as a subset.
    """
    for batch in getattr(response, "generations", None) or []:
        for generation in batch or []:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict) and usage:
                details = usage.get("input_token_details") or {}
                return {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cached_input_tokens": details.get("cache_read"),
                }

    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if token_usage:
        cached = None
        details = token_usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
        return {
            "input_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens"),
            "output_tokens": (
                token_usage.get("completion_tokens") or token_usage.get("output_tokens")
            ),
            "total_tokens": token_usage.get("total_tokens"),
            "cached_input_tokens": cached or token_usage.get("cache_read_input_tokens"),
        }

    for batch in getattr(response, "generations", None) or []:
        for generation in batch or []:
            info = getattr(generation, "generation_info", None) or {}
            usage = info.get("usage") or info.get("token_usage")
            if isinstance(usage, dict) and usage:
                return {
                    "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
                    "output_tokens": (
                        usage.get("completion_tokens") or usage.get("output_tokens")
                    ),
                    "total_tokens": usage.get("total_tokens"),
                    "cached_input_tokens": usage.get("cache_read_input_tokens"),
                }

    return {}


def _response_model(response: Any) -> Optional[str]:
    llm_output = getattr(response, "llm_output", None) or {}
    for key in ("model_name", "model", "model_id"):
        value = llm_output.get(key)
        if isinstance(value, str) and value:
            return value
    for batch in getattr(response, "generations", None) or []:
        for generation in batch or []:
            message = getattr(generation, "message", None)
            meta = getattr(message, "response_metadata", None) or {}
            for key in ("model_name", "model", "model_id"):
                value = meta.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _tool_definitions(params: Optional[Dict[str, Any]]) -> Optional[List[ToolDefinition]]:
    """The tool menu offered to the model on THIS call.

    ``bind_tools()`` puts the provider-formatted schemas into
    ``invocation_params``: OpenAI-style ``tools`` envelopes, legacy
    ``functions`` entries, or Anthropic-style ``{name, input_schema}``. All
    three are normalized to ``{name, description, parameters}``.
    """
    if not params:
        return None
    raw = params.get("tools") or params.get("functions") or params.get("tool_schemas")
    if not isinstance(raw, list):
        return None

    out: List[ToolDefinition] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        definition: ToolDefinition = {"name": name}
        description = fn.get("description")
        if isinstance(description, str) and description:
            definition["description"] = description
        parameters = fn.get("parameters") or fn.get("input_schema") or fn.get("inputSchema")
        if isinstance(parameters, dict):
            definition["parameters"] = parameters
        out.append(definition)
    return out or None


def _response_format(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The structured-output contract LangChain bound for THIS call.

    ``with_structured_output()`` and an explicit ``response_format`` both land
    in the same ``invocation_params`` bag the tool schemas do, so this is the
    only place a callback handler can see them. Without it a trace cannot tell
    a model that CHOSE prose from one that was never asked for JSON.
    """
    if not params:
        return None
    raw = params.get("response_format") or params.get("responseFormat")
    if not isinstance(raw, dict):
        return None

    raw_type = raw.get("type")
    type_ = raw_type if isinstance(raw_type, str) and raw_type else None
    nested = raw.get("json_schema") or raw.get("jsonSchema")
    source = nested if isinstance(nested, dict) else raw
    schema = source.get("schema")
    schema = schema if isinstance(schema, dict) else None
    if not type_ and schema is None:
        return None

    out: Dict[str, Any] = {"type": type_ or "json_schema"}
    name = source.get("name")
    if isinstance(name, str) and name:
        out["name"] = name
    strict = source.get("strict")
    if isinstance(strict, bool):
        out["strict"] = strict
    if schema is not None:
        out["schema"] = schema
    return out


def _chain_io(payload: Any) -> Any:
    """Render chain / graph-node inputs and outputs.

    LangGraph state is a dict whose ``messages`` key holds BaseMessage objects;
    rendering those as message sections would duplicate what the model events
    already show, so here they are flattened to a compact role/content list.
    """
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "messages" and isinstance(value, (list, tuple)):
                out[key] = [
                    {"role": _message_role(message), "content": getattr(message, "content", message)}
                    for message in value
                ]
            else:
                out[key] = value
        return out
    if isinstance(payload, (list, tuple)):
        return [_chain_io(item) for item in payload]
    return payload


def _tool_output(output: Any) -> Any:
    """ToolMessage wraps the real payload — unwrap it so the UI shows content."""
    content = getattr(output, "content", None)
    return content if content is not None else output


def _document_to_dict(document: Any) -> Any:
    content = getattr(document, "page_content", None)
    if content is None:
        return document
    return {"page_content": content, "metadata": getattr(document, "metadata", {})}


# ── Global installation ──────────────────────────────────────────────────

#: Keeps the globally installed handler alive and inheritable by child
#: contexts. LangChain's configure hook reads the handler out of this var
#: every time it builds a callback manager.
_GLOBAL_HANDLER: ContextVar[Optional["CognipeerCallbackHandler"]] = ContextVar(
    "cognipeer_langchain_handler", default=None
)
_HOOK_REGISTERED = False


def install_langchain_tracing(**handler_options: Any) -> "CognipeerCallbackHandler":
    """Attach a handler to every LangChain run in this process.

    Uses LangChain's global configure hook, so no per-invoke ``config`` is
    needed. Convenient for an application you do not want to edit call by
    call.

    Two caveats worth knowing:

    * every run in the process shares ONE session, which is what you want for
      a script and not what you want for a server handling concurrent, unrelated
      conversations — there, pass a per-invocation handler instead;
    * ``contextvars`` are inherited at task/thread creation, so call this before
      spawning the work that should be traced.
    """
    global _HOOK_REGISTERED
    handler = CognipeerCallbackHandler(**handler_options)
    _GLOBAL_HANDLER.set(handler)

    if _HOOK_REGISTERED:
        return handler

    register_hook = None
    try:  # LangChain 0.1+ keeps it here…
        from langchain_core.tracers.context import register_configure_hook as register_hook
    except ImportError:  # pragma: no cover - layout differs across versions
        try:
            from langchain_core.callbacks.manager import (  # type: ignore[no-redef]
                register_configure_hook as register_hook,
            )
        except ImportError:
            register_hook = None

    if register_hook is None:  # pragma: no cover - very old langchain
        logger.warning(
            "cognipeer: this langchain version has no global callback hook; "
            "pass the handler via config={'callbacks': [...]} instead"
        )
        return handler

    register_hook(_GLOBAL_HANDLER, inheritable=True)
    _HOOK_REGISTERED = True
    return handler


__all__ = [
    "CognipeerCallbackHandler",
    "NOISE_CHAIN_NAMES",
    "cognipeer_config",
    "install_langchain_tracing",
]
