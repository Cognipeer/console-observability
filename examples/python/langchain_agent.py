"""A real tool-calling LangChain agent, traced with one callback handler.

Demonstrates what the handler recovers from an ordinary agent run:

  * the model call, with the provider's model id and token counts;
  * **the tool menu the model was offered** — LangChain strips `tools` from
    the metadata it hands to tracers, but the unfiltered `invocation_params`
    still reach a callback handler, which is why this works here and shows
    nothing on the LangSmith/OTel export path;
  * each tool invocation with its arguments and result;
  * the whole thing as one session, nested by parent run.

⚠️ Calls the OpenAI API — this one costs money (a few tenths of a cent).

    pip install "cognipeer-observability[langchain]" langchain-openai
    export OPENAI_API_KEY=sk-…
    python examples/python/langchain_agent.py

In Console: Tracing → Sessions → "support-bot". Open the `ai_call` event and
expand its Tool Definitions section — that is the tool menu, captured per call.

Written against LangChain 1.x (`langchain.agents.create_agent`). On 0.x, build
an `AgentExecutor` instead and pass the same `config={"callbacks": [...]}`;
the handler itself is unchanged from 0.1 onwards.
"""

from __future__ import annotations

import os
import sys

import cognipeer_observability as cognipeer
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────────────────────────────────
#  Wire tracing in — this is the whole integration
# ─────────────────────────────────────────────────────────────────────
from cognipeer_observability.langchain import CognipeerCallbackHandler

for required in ("COGNIPEER_API_KEY", "OPENAI_API_KEY"):
    if not os.environ.get(required):
        sys.exit(f"Set {required} before running this example.")

cognipeer.init(agent={"name": "support-bot", "version": "1.0.0"})
# ─────────────────────────────────────────────────────────────────────

ORDERS = {
    "A-1001": {"status": "shipped", "carrier": "DHL", "eta": "2026-08-18"},
    "A-1002": {"status": "processing", "carrier": None, "eta": None},
}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up the delivery status of an order by its id."""
    return ORDERS.get(order_id, {"status": "not_found"})


@tool
def refund_policy(days_since_delivery: int) -> str:
    """Return the refund policy for an order delivered this many days ago."""
    return "Refundable" if days_since_delivery <= 30 else "Outside the refund window"


def main() -> None:
    agent = create_agent(
        ChatOpenAI(model="gpt-4.1-mini", temperature=0),
        tools=[lookup_order, refund_policy],
        system_prompt="You are a concise customer support agent.",
    )

    result = agent.invoke(
        {"messages": [("user", "Where is order A-1001, and can I still refund it?")]},
        # `thread_id` groups this run with the rest of the conversation in
        # Tracing → Threads. Everything else is default.
        config={"callbacks": [CognipeerCallbackHandler(thread_id="conv-42")]},
    )
    print(result["messages"][-1].content)

    # Short-lived process: without this the script can exit before the last
    # export lands.
    cognipeer.flush()
    print("\nLook in Console → Tracing → Sessions → agent 'support-bot'")


if __name__ == "__main__":
    main()
