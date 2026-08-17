"""LangGraph with a human-in-the-loop interrupt, traced end to end.

Demonstrates the two things about LangGraph that surprise people when they
first look at their traces:

  * **One conversation is several sessions.** Every `invoke()` — including the
    resume after an interrupt — is a fresh root run with a fresh trace id.
    `thread_id` is the only thing tying them together, which is why
    `graph_config()` plumbs it into both `configurable` (for the checkpointer)
    and `metadata` (for the tracing handler).
  * **An interrupt is not a failure.** `interrupt()` raises a `GraphInterrupt`
    that surfaces through the callback API as a chain error. The handler knows
    the difference and records the step as successful with
    `metadata.interrupted`, instead of marking every approval step failed.

Runs entirely offline — the model is LangChain's `FakeListChatModel` — so it
costs nothing.

    pip install "cognipeer-observability[langgraph]"
    python examples/python/langgraph_agent.py

In Console: Tracing → Threads → "conv-77". One thread, two sessions. Open the
first and the `ask_human` node carries an `interrupted` flag.
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

import cognipeer_observability as cognipeer
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# ─────────────────────────────────────────────────────────────────────
#  Wire tracing in — this is the whole integration
# ─────────────────────────────────────────────────────────────────────
from cognipeer_observability.langgraph import graph_config

if not os.environ.get("COGNIPEER_API_KEY"):
    sys.exit("Set COGNIPEER_API_KEY (Settings → API Tokens, with `tracing` enabled).")

cognipeer.init(agent={"name": "refund-approver", "version": "1.0.0"})
# ─────────────────────────────────────────────────────────────────────

THREAD_ID = "conv-77"

model = FakeListChatModel(responses=["Refund of 240 EUR looks legitimate."])


class State(TypedDict):
    request: str
    assessment: str
    decision: str


def assess(state: State) -> dict:
    # The model call inherits the run's callbacks from LangGraph's config, so
    # it shows up as its own `ai_call` event under this node — nothing to wire.
    reply = model.invoke(f"Assess this refund request: {state['request']}")
    return {"assessment": reply.content}


def ask_human(state: State) -> dict:
    # Suspends the graph. Everything BEFORE this line re-runs on resume, so
    # keep model and tool calls after it unless you want them charged twice.
    decision = interrupt({"question": "Approve?", "assessment": state["assessment"]})
    return {"decision": decision}


builder = StateGraph(State)
builder.add_node("assess", assess)
builder.add_node("ask_human", ask_human)
builder.add_edge(START, "assess")
builder.add_edge("assess", "ask_human")
builder.add_edge("ask_human", END)
graph = builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    # Session 1 — runs until the interrupt suspends the graph.
    config = graph_config(THREAD_ID)
    first = graph.invoke({"request": "refund 240 EUR for order A-1001"}, config=config)
    print("suspended, waiting for:", first["__interrupt__"][0].value["question"])

    # Session 2 — the resume. A new root run on the SAME thread. Reusing the
    # first call's `configurable` is what points the checkpointer at the
    # suspended state.
    resumed = graph.invoke(
        Command(resume="approved"),
        config=graph_config(THREAD_ID, base={"configurable": config["configurable"]}),
    )
    print("decision:", resumed["decision"])

    cognipeer.flush()
    print(f"\nLook in Console → Tracing → Threads → '{THREAD_ID}' (two sessions, one thread)")


if __name__ == "__main__":
    main()
