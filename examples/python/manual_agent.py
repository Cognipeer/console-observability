"""A hand-rolled agent loop, traced with no framework at all.

This is the template to copy when your agent is your own code. It shows the
three things that matter:

  * `trace()` opens the session — one agent run, one row in Tracing → Sessions.
  * `@observe` turns any function into an event, nested automatically under
    whatever called it.
  * `type="tool_call"` / `type="ai_call"` make an event render as a tool
    invocation or a model call rather than a generic step.

Runs entirely offline — the "model" is a scripted stub — so it costs nothing
and is safe to run in CI.

    python examples/python/manual_agent.py

In Console: Tracing → Sessions → "order-bot". The timeline should read
plan → search_orders → summarise, with `search_orders` nested under `plan`,
and the token counts you see are the ones the loop reported.
"""

from __future__ import annotations

import os
import sys

import cognipeer_observability as cognipeer

# ─────────────────────────────────────────────────────────────────────
#  Wire tracing in — this is the whole integration
# ─────────────────────────────────────────────────────────────────────
if not os.environ.get("COGNIPEER_API_KEY"):
    sys.exit("Set COGNIPEER_API_KEY (Settings → API Tokens, with `tracing` enabled).")

cognipeer.init(agent={"name": "order-bot", "version": "1.0.0"})
# ─────────────────────────────────────────────────────────────────────


ORDERS = {
    "A-1001": {"status": "shipped", "carrier": "DHL", "eta": "2026-08-18"},
    "A-1002": {"status": "processing", "carrier": None, "eta": None},
}


@cognipeer.observe(type="tool_call", tool_name="search_orders")
def search_orders(order_id: str) -> dict:
    """A tool. Arguments and return value become the event's two sections."""
    return ORDERS.get(order_id, {"status": "not_found"})


def call_model(session, prompt: str) -> str:
    """A model call, recorded with everything Console can use.

    `@observe(type="ai_call")` would also work and is one line — but it can
    only record what a function signature shows. A model call is worth
    recording by hand because four extra fields unlock the rest of Console:

      * `model` — the provider's model id, which is what cost resolution
        matches against Model Hub. A nickname prices at zero.
      * token counts — the provider's own numbers. Never estimate them, and
        omit them entirely rather than sending zeros; an absent value shows up
        as unknown, a zero silently under-reports spend.
      * `cachedInputTokens` — a SUBSET of `inputTokens`, priced at the cached
        rate.
      * `toolDefinitions` — the tool menu this call was offered. It changes
        between turns and is often the biggest line item in the prompt bill,
        which is why it belongs on the event and not on the session.
    """
    completion = f"(pretend completion for: {prompt[:40]})"

    session.record(
        {
            "type": "ai_call",
            "label": "gpt-4.1-mini",
            "model": "gpt-4.1-mini",
            "inputTokens": 412,
            "outputTokens": 58,
            "cachedInputTokens": 256,
            "sections": [
                {"kind": "message", "role": "user", "content": prompt},
                {"kind": "message", "role": "assistant", "content": completion},
            ],
            "toolDefinitions": [
                {
                    "name": "search_orders",
                    "description": "Look up an order by id",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                }
            ],
        }
    )
    return completion


@cognipeer.observe()
def plan(question: str) -> dict:
    """A step that calls a tool. The tool event nests under this one."""
    order_id = question.split()[-1]
    return search_orders(order_id)


@cognipeer.observe()
def summarise(order: dict) -> str:
    if order["status"] == "not_found":
        return "I could not find that order."
    return f"Your order is {order['status']} (carrier: {order['carrier']})."


def main() -> None:
    question = "where is order A-1001"

    # `thread_id` groups this run with the rest of the conversation in
    # Tracing → Threads. Use whatever key your app already has.
    with cognipeer.trace(name="order-bot", thread_id="conv-42") as session:
        order = plan(question)
        call_model(session, f"Summarise this order: {order}")
        answer = summarise(order)
        print(answer)

    # Short-lived process: without this the script can exit before the last
    # export lands. A long-running server does not need it.
    cognipeer.flush()
    print("\nLook in Console → Tracing → Sessions → agent 'order-bot'")


if __name__ == "__main__":
    main()
