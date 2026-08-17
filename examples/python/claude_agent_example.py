"""Claude Agent SDK, traced by consuming its message stream.

`trace_query` is a drop-in for `claude_agent_sdk.query` — it yields the SDK's
messages untouched and records them on the way past, so existing consumer code
keeps working.

Demonstrates:

  * the assistant's text, thinking and `tool_use` blocks becoming one `ai_call`
    event per turn, with the model and the real token usage;
  * a `tool_use` block and its later matching `tool_result` merged into ONE
    `tool_call` event, so the timeline reads as calls rather than fragments;
  * subagents nesting under the tool call that spawned them, via
    `parent_tool_use_id`.

⚠️ Calls the Anthropic API through the Claude Code CLI — this one costs money.

    pip install "cognipeer-observability[claude-agent-sdk]"
    npm install -g @anthropic-ai/claude-code     # the SDK drives this binary
    export ANTHROPIC_API_KEY=sk-ant-…
    python examples/python/claude_agent_example.py

In Console: Tracing → Sessions → "code-assistant". Token counts are worth a
look: Anthropic reports `input_tokens` EXCLUDING cache reads, while Console
treats cached tokens as a subset of the input total, so the integration adds
the cache buckets back in. What you see is the full prompt volume with the
cached portion broken out, which is what prices correctly.

Tool JSON schemas are NOT available on this seam — the SDK reports tool
*names* only. That is a limit of the SDK, not of the integration.
"""

from __future__ import annotations

import asyncio
import os
import sys

import cognipeer_observability as cognipeer
from claude_agent_sdk import ClaudeAgentOptions

# ─────────────────────────────────────────────────────────────────────
#  Wire tracing in — this is the whole integration
# ─────────────────────────────────────────────────────────────────────
from cognipeer_observability.claude_agent_sdk import trace_query

for required in ("COGNIPEER_API_KEY", "ANTHROPIC_API_KEY"):
    if not os.environ.get(required):
        sys.exit(f"Set {required} before running this example.")

cognipeer.init(agent={"name": "code-assistant", "version": "1.0.0"})
# ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    options = ClaudeAgentOptions(
        model="claude-sonnet-5",
        allowed_tools=["Read", "Glob"],
        max_turns=3,
    )

    # `trace_query` replaces `query`. Everything else is unchanged — the
    # messages you get here are the SDK's own objects.
    async for message in trace_query(
        prompt="List the Python files in this directory and summarise what they do.",
        options=options,
        thread_id="conv-42",
    ):
        # A `result` message closes the session on its own; nothing to do here
        # but consume the stream as you normally would.
        if getattr(message, "subtype", None) == "success":
            print(getattr(message, "result", ""))

    # Short-lived process: without this the script can exit before the last
    # export lands.
    cognipeer.flush()
    print("\nLook in Console → Tracing → Sessions → agent 'code-assistant'")


if __name__ == "__main__":
    asyncio.run(main())
