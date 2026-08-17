# Examples

Each file is standalone and runnable as written. Start with the one that
matches your framework; if none does, start with `manual_agent.py` /
`manual-agent.ts`, which is the template for a bespoke agent.

## Two env vars

```bash
export COGNIPEER_API_KEY="cpeer_…"                      # Settings → API Tokens, with `tracing` enabled
export COGNIPEER_BASE_URL="https://console.acme.internal"   # self-hosted only; host root, no /api path
```

Every example checks for what it needs and exits with a clear message rather
than failing halfway through.

## Python

```bash
python examples/python/manual_agent.py
```

| Example | Shows | Needs | Cost |
|---|---|---|---|
| [`manual_agent.py`](python/manual_agent.py) | `@observe`, `trace()`, and recording a model call by hand with tokens and a tool menu | nothing | **free, offline** |
| [`langgraph_agent.py`](python/langgraph_agent.py) | interrupt + resume as two sessions on one thread; interrupts recorded as pauses, not failures | `[langgraph]` | **free, offline** |
| [`langchain_agent.py`](python/langchain_agent.py) | a real tool-calling agent; tool schemas captured per model call | `[langchain]`, `langchain-openai`, `OPENAI_API_KEY` | ⚠️ calls OpenAI |
| [`openai_agents_example.py`](python/openai_agents_example.py) | `install_openai_agents_tracing()`, `RunConfig(group_id=…)` | `[openai-agents]`, `OPENAI_API_KEY` | ⚠️ calls OpenAI |
| [`claude_agent_example.py`](python/claude_agent_example.py) | `trace_query` as a drop-in for `query`; tool_use + tool_result merged into one event | `[claude-agent-sdk]`, Claude Code CLI, `ANTHROPIC_API_KEY` | ⚠️ calls Anthropic |
| [`otel_crewai.py`](python/otel_crewai.py) | the "any framework" path — `CognipeerSpanExporter` in a `TracerProvider` | `[otel]`, `crewai`, `openinference-instrumentation-crewai`, `OPENAI_API_KEY` | ⚠️ calls OpenAI |

The bracketed names are extras: `pip install "cognipeer-observability[langgraph]"`.

## TypeScript

```bash
npx tsx examples/js/manual-agent.ts
```

| Example | Shows | Needs | Cost |
|---|---|---|---|
| [`manual-agent.ts`](js/manual-agent.ts) | `observe()`, `trace()`, and recording a model call by hand | nothing | **free, offline** |
| [`langchain-agent.ts`](js/langchain-agent.ts) | a real tool-calling agent with `CognipeerCallbackHandler` | `@langchain/*`, `OPENAI_API_KEY` | ⚠️ calls OpenAI |
| [`openai-agents.ts`](js/openai-agents.ts) | `installOpenAIAgentsTracing()`, `groupId` | `@openai/agents`, `OPENAI_API_KEY` | ⚠️ calls OpenAI |
| [`vercel-ai.ts`](js/vercel-ai.ts) | the native telemetry integration and the portable middleware route | `ai`, `@ai-sdk/openai`, `OPENAI_API_KEY` | ⚠️ calls OpenAI |

## n8n

See [`n8n/README.md`](n8n/README.md) — three routes, fastest first. The bridge
needs no files and works on n8n Cloud:

```bash
npx @cognipeer/observability cognipeer-n8n \
  --n8n-url https://n8n.acme.com --n8n-api-key "$N8N_API_KEY" \
  --api-key "$COGNIPEER_API_KEY" --once
```

## Nothing showing up?

Run again with `COGNIPEER_DEBUG=1` — it logs every request the exporter makes
and every failure it swallowed. The usual cause in a script is a missing
`flush()`: exports are asynchronous, and a short-lived process can exit before
the last one lands. Every example here calls it.
