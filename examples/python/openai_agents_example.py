"""OpenAI Agents SDK, traced through its own tracing processor.

Demonstrates:

  * `install_openai_agents_tracing()` — one call, and every `Runner.run` in the
    process is traced. By default it REPLACES the SDK's processor list, so
    traces stop going to OpenAI's dashboard and no prompt data leaves for a
    second destination. Pass `keep_openai_exporter=True` to export to both.
  * `RunConfig(group_id=...)` — the SDK's only conversation identifier, and
    nothing sets it for you. Without it every run is an orphan session, even
    when the app uses an SDK `Session`.
  * function tools, handoffs and guardrails all arriving as nested events.

⚠️ Calls the OpenAI API — this one costs money (a few tenths of a cent).

    pip install "cognipeer-observability[openai-agents]"
    export OPENAI_API_KEY=sk-…
    python examples/python/openai_agents_example.py

In Console: Tracing → Threads → "conv-42". The `ai_call` event carries the
full tool schemas, because this agent runs on the Responses API — on the Chat
Completions path the SDK only reports tool *names*, and the trace shows those.

Needs `openai-agents>=0.14` for per-call token usage; the tracing interface
itself is stable back to 0.9.
"""

from __future__ import annotations

import asyncio
import os
import sys

import cognipeer_observability as cognipeer
from agents import Agent, RunConfig, Runner, function_tool

# ─────────────────────────────────────────────────────────────────────
#  Wire tracing in — this is the whole integration
# ─────────────────────────────────────────────────────────────────────
from cognipeer_observability.openai_agents import install_openai_agents_tracing

for required in ("COGNIPEER_API_KEY", "OPENAI_API_KEY"):
    if not os.environ.get(required):
        sys.exit(f"Set {required} before running this example.")

cognipeer.init(agent={"name": "support-bot", "version": "1.0.0"})
install_openai_agents_tracing()
# ─────────────────────────────────────────────────────────────────────

ORDERS = {
    "A-1001": {"status": "shipped", "carrier": "DHL", "eta": "2026-08-18"},
    "A-1002": {"status": "processing", "carrier": None, "eta": None},
}


@function_tool
def lookup_order(order_id: str) -> dict:
    """Look up the delivery status of an order by its id."""
    return ORDERS.get(order_id, {"status": "not_found"})


async def main() -> None:
    agent = Agent(
        name="support-bot",
        instructions="You are a concise customer support agent.",
        model="gpt-4.1-mini",
        tools=[lookup_order],
    )

    result = await Runner.run(
        agent,
        "Where is order A-1001?",
        # `group_id` is what groups this run with the rest of the conversation
        # in Tracing → Threads. `workflow_name` becomes the trace name.
        run_config=RunConfig(workflow_name="support-bot", group_id="conv-42"),
    )
    print(result.final_output)

    # Short-lived process: without this the script can exit before the last
    # export lands.
    cognipeer.flush()
    print("\nLook in Console → Tracing → Threads → 'conv-42'")


if __name__ == "__main__":
    asyncio.run(main())
