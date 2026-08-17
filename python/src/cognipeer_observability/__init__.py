"""Cognipeer Observability — ship agent traces into Cognipeer Console.

Quick start::

    import cognipeer_observability as cognipeer

    cognipeer.init(api_key="cpeer_…", agent={"name": "support-bot"})

    from cognipeer_observability.langchain import CognipeerCallbackHandler
    chain.invoke({"input": "…"}, config={"callbacks": [CognipeerCallbackHandler()]})

Framework integrations live in submodules and import their framework lazily,
so ``pip install cognipeer-observability`` pulls in nothing you do not use:

===============================================  ===================================
``cognipeer_observability.langchain``            LangChain / LangGraph callback handler
``cognipeer_observability.langgraph``            LangGraph helpers on top of it
``cognipeer_observability.openai_agents``        OpenAI Agents SDK tracing processor
``cognipeer_observability.claude_agent_sdk``     Claude Agent SDK hooks + message tracer
``cognipeer_observability.otel``                 OpenTelemetry span exporter
===============================================  ===================================
"""

from ._config import (
    ENV_AGENT_NAME,
    ENV_AGENT_VERSION,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_CAPTURE,
    ENV_DEBUG,
    ENV_ENABLED,
    ENV_MODE,
    Config,
)
from .client import Cognipeer, get_client, init, reset_client
from .context import get_current_session, get_current_span_key, use_session, use_span
from .decorators import observe, trace
from .session import TraceSession
from .types import Agent, Event, ResponseFormat, Section, Summary, ToolDefinition

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Cognipeer",
    "Config",
    "ENV_AGENT_NAME",
    "ENV_AGENT_VERSION",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_CAPTURE",
    "ENV_DEBUG",
    "ENV_ENABLED",
    "ENV_MODE",
    "Event",
    "ResponseFormat",
    "Section",
    "Summary",
    "ToolDefinition",
    "TraceSession",
    "__version__",
    "flush",
    "get_client",
    "get_current_session",
    "get_current_span_key",
    "init",
    "observe",
    "reset_client",
    "shutdown",
    "trace",
    "use_session",
    "use_span",
]


def flush(timeout: float = 10.0) -> None:
    """Block until every queued trace has been delivered.

    Call this before a short-lived process exits (a script, a Lambda handler,
    a CI job). Long-running services do not need it.
    """
    get_client().flush(timeout)


def shutdown(timeout: float = 10.0) -> None:
    """End every open session and flush. Safe to call more than once."""
    get_client().shutdown(timeout)
