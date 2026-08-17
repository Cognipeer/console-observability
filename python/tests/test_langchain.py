"""LangChain and LangGraph integration, driven through the real frameworks.

No model API key is needed: a fake chat model exercises the callback path, and
the one test that needs `bind_tools` points a real `ChatOpenAI` at a closed
port — `on_chat_model_start` fires before the request, which is the only part
under test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

import cognipeer_observability as cognipeer
from cognipeer_observability import _transport

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from cognipeer_observability.langchain import (  # noqa: E402
    CognipeerCallbackHandler,
    cognipeer_config,
)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, Any]]:
    sent: List[Tuple[str, Any]] = []
    monkeypatch.setattr(
        _transport.Transport,
        "_submit",
        lambda self, path, payload: sent.append((path, payload)),
    )
    monkeypatch.setenv("COGNIPEER_API_KEY", "cpeer_test_key_000000000000")
    cognipeer.reset_client()
    cognipeer.init(mode="batch", agent={"name": "test-agent"})
    return sent


def events_of(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return payload["events"]


def test_records_a_chain_run_as_one_session(captured: List[Tuple[str, Any]]) -> None:
    chain = ChatPromptTemplate.from_messages([("system", "be nice"), ("user", "{q}")]) | (
        FakeListChatModel(responses=["hello"])
    )
    handler = CognipeerCallbackHandler()

    chain.invoke({"q": "hi"}, config={"callbacks": [handler]})

    assert len(captured) == 1
    events = events_of(captured[0][1])
    types = [event["type"] for event in events]
    assert "ai_call" in types

    model_call = next(event for event in events if event["type"] == "ai_call")
    contents = [section["content"] for section in model_call["sections"]]
    assert "be nice" in contents
    assert "hi" in contents
    assert "hello" in contents


def test_span_ids_are_unique_and_the_tree_is_connected(
    captured: List[Tuple[str, Any]],
) -> None:
    chain = ChatPromptTemplate.from_messages([("user", "{q}")]) | FakeListChatModel(
        responses=["hello"]
    )
    chain.invoke({"q": "hi"}, config={"callbacks": [CognipeerCallbackHandler()]})

    events = events_of(captured[0][1])
    span_ids = [event["spanId"] for event in events]
    assert len(set(span_ids)) == len(span_ids)

    # Plumbing runnables are filtered, so their children must be re-parented
    # onto something that was actually sent — never a dangling span id.
    root = captured[0][1]["rootSpanId"]
    known = set(span_ids) | {root}
    assert all(event["parentSpanId"] in known for event in events)


def test_tools_are_recorded_with_their_arguments_and_result(
    captured: List[Tuple[str, Any]],
) -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    add.invoke({"a": 2, "b": 3}, config={"callbacks": [CognipeerCallbackHandler()]})

    event = events_of(captured[0][1])[0]
    assert event["type"] == "tool_call"
    assert event["toolName"] == "add"
    kinds = {section["kind"] for section in event["sections"]}
    assert kinds == {"tool_call", "tool_result"}


def test_thread_id_comes_from_the_config_and_survives_later_runs(
    captured: List[Tuple[str, Any]],
) -> None:
    chain = ChatPromptTemplate.from_messages([("user", "{q}")]) | FakeListChatModel(
        responses=["a", "b"]
    )
    handler = CognipeerCallbackHandler()

    chain.invoke({"q": "one"}, config=cognipeer_config("conv-42", handler=handler))
    # A second run through the same handler opens a NEW session on the SAME
    # thread — that is what the Threads view is built on.
    chain.invoke({"q": "two"}, config={"callbacks": [handler]})

    assert len(captured) == 2
    assert [payload["threadId"] for _, payload in captured] == ["conv-42", "conv-42"]
    assert captured[0][1]["sessionId"] != captured[1][1]["sessionId"]


def test_errors_mark_the_session_failed(captured: List[Tuple[str, Any]]) -> None:
    @tool
    def explode(x: int) -> int:
        """Always fails."""
        raise ValueError("boom")

    with pytest.raises(ValueError):
        explode.invoke({"x": 1}, config={"callbacks": [CognipeerCallbackHandler()]})

    payload = captured[0][1]
    assert payload["status"] == "error"
    assert events_of(payload)[0]["status"] == "error"


def test_captures_the_tool_menu_offered_to_the_model(
    monkeypatch: pytest.MonkeyPatch, captured: List[Tuple[str, Any]]
) -> None:
    """`bind_tools` puts the provider-formatted schemas into invocation_params.

    LangChain strips `tools` from the metadata it gives tracers, but the
    unfiltered dict still reaches a callback handler — which is why this works
    and the LangSmith/OTel export path shows no tool definitions at all.
    """
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-0000000000000000")
    # Closed port: the request fails immediately, after the start callback.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    model = ChatOpenAI(model="gpt-4.1-mini", max_retries=0, timeout=2).bind_tools([get_weather])
    handler = CognipeerCallbackHandler()

    with pytest.raises(Exception):
        model.invoke("weather in Istanbul?", config={"callbacks": [handler]})

    # The failed model call is the run's root, so the session closed itself.
    assert captured[0][1]["status"] == "error"
    event = events_of(captured[0][1])[0]
    assert event["model"] == "gpt-4.1-mini"
    assert event["toolDefinitions"] == [
        {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "type": "object",
            },
        }
    ]


def test_langgraph_interrupts_are_not_failures(captured: List[Tuple[str, Any]]) -> None:
    """A human-in-the-loop pause is control flow, not an error.

    LangGraph raises `GraphInterrupt` to suspend a run, and it surfaces through
    the callback API as a chain error. Mapping that straight to `error` would
    mark every approval step as a failed run.
    """
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    from cognipeer_observability.langgraph import graph_config

    class State(TypedDict):
        value: str

    def ask_human(state: State) -> State:
        answer = interrupt({"question": "approve?"})
        return {"value": f"{state['value']}|{answer}"}

    builder = StateGraph(State)
    builder.add_node("ask_human", ask_human)
    builder.add_edge(START, "ask_human")
    builder.add_edge("ask_human", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = graph_config("conv-77")
    graph.invoke({"value": "start"}, config=config)
    resumed = graph.invoke(
        Command(resume="yes"),
        config=graph_config("conv-77", base={"configurable": config["configurable"]}),
    )
    assert resumed["value"] == "start|yes"

    # Two invocations, two sessions, one thread.
    assert len(captured) == 2
    assert {payload["threadId"] for _, payload in captured} == {"conv-77"}
    assert [payload["status"] for _, payload in captured] == ["success", "success"]

    interrupted = [
        event
        for _, payload in captured
        for event in events_of(payload)
        if (event.get("metadata") or {}).get("interrupted")
    ]
    assert len(interrupted) == 1
    assert interrupted[0]["status"] == "success"
    assert interrupted[0]["metadata"]["interruptType"] == "GraphInterrupt"
