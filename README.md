<div align="center">

# Cognipeer Observability

**Ship agent traces from whatever framework you already use into [Cognipeer Console](https://github.com/Cognipeer/console).**

[![npm](https://img.shields.io/npm/v/@cognipeer/observability?label=%40cognipeer%2Fobservability)](https://www.npmjs.com/package/@cognipeer/observability)
[![PyPI](https://img.shields.io/pypi/v/cognipeer-observability?label=cognipeer-observability)](https://pypi.org/project/cognipeer-observability/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

</div>

You already built the agent. This makes it observable — prompts, completions,
tool calls, token usage, cost and latency — without rewriting it.

```python
# Python
import cognipeer_observability as cognipeer
from cognipeer_observability.langchain import CognipeerCallbackHandler

cognipeer.init(api_key="cpeer_…", agent={"name": "support-bot"})
agent.invoke(state, config={"callbacks": [CognipeerCallbackHandler()]})
```

```ts
// TypeScript
import { init } from '@cognipeer/observability';
import { CognipeerCallbackHandler } from '@cognipeer/observability/langchain';

init({ apiKey: 'cpeer_…', agent: { name: 'support-bot' } });
await agent.invoke(state, { callbacks: [new CognipeerCallbackHandler()] });
```

That is the whole integration. Two lines, no proxy, no base-URL swap, no
change to how your agent calls its models.

## What you get

| | |
|---|---|
| **Timeline** | Every model call, tool call, retrieval and graph node, nested by parent/child, with per-step latency |
| **Content** | Prompts and completions as readable message blocks, tool arguments and results as structured JSON |
| **Tool menu** | The tool definitions the model was actually offered on each call — the thing that quietly doubles your prompt bill |
| **Tokens & cost** | Input, output and cached tokens per call, priced against your Model Hub or external pricing catalogue |
| **Threads** | Many runs grouped into one conversation, so a multi-turn agent reads as a conversation |
| **Errors** | Failed steps with the exception attached, and the run marked failed rather than silently short |

## Supported frameworks

| Framework | Python | TypeScript | How |
|---|:---:|:---:|---|
| [LangChain](https://cognipeer.github.io/console/guide/observability/langchain) | ✅ | ✅ | Callback handler (works from 0.1 to 1.x) |
| [LangGraph](https://cognipeer.github.io/console/guide/observability/langgraph) | ✅ | ✅ | Same handler + thread/interrupt handling |
| [OpenAI Agents SDK](https://cognipeer.github.io/console/guide/observability/openai-agents) | ✅ | ✅ | Tracing processor |
| [Claude Agent SDK](https://cognipeer.github.io/console/guide/observability/claude-agent-sdk) | ✅ | ✅ | Message-stream tracer |
| [Vercel AI SDK](https://cognipeer.github.io/console/guide/observability/vercel-ai) | — | ✅ | Model middleware or `experimental_telemetry` |
| [n8n](https://cognipeer.github.io/console/guide/observability/n8n) | — | ✅ | Execution bridge or external hook |
| [Anything OpenTelemetry](https://cognipeer.github.io/console/guide/observability/opentelemetry) | ✅ | ✅ | Span exporter — CrewAI, LlamaIndex, Pydantic AI, Google ADK, Strands, Semantic Kernel, smolagents … |
| [Anything else](https://cognipeer.github.io/console/guide/observability/manual) | ✅ | ✅ | `@observe` / `observe()` and the session API |

Not on the list? If it emits OpenTelemetry spans — and most 2026-era agent
frameworks do, natively or through an OpenInference/OpenLLMetry instrumentor —
the [OTLP route](https://cognipeer.github.io/console/guide/observability/opentelemetry) already covers it.

## Install

```bash
# Python — extras pull in only what you use
pip install cognipeer-observability[langchain]
pip install cognipeer-observability[all]

# TypeScript
npm install @cognipeer/observability
```

The core has **no required dependencies** in either language. Framework
packages are optional peers, so this cannot drag your dependency tree around.

## Configure

Everything reads from the environment, so the code above works unchanged
across dev, staging and production:

| Variable | Default | Meaning |
|---|---|---|
| `COGNIPEER_API_KEY` | — | Console API token. Without it, tracing disables itself and warns once — it never throws. |
| `COGNIPEER_BASE_URL` | `https://console.cognipeer.com` | Your Console, for self-hosted installs |
| `COGNIPEER_AGENT_NAME` | — | Default agent name for every session |
| `COGNIPEER_CAPTURE_CONTENT` | `all` | `all`, `metadata` (structure and tokens, no message bodies), or `none` |
| `COGNIPEER_TRACING_ENABLED` | `true` | Master switch |
| `COGNIPEER_TRACING_MODE` | `auto` | `auto`, `stream` (live updates) or `batch` (one request per run) |
| `COGNIPEER_DEBUG` | `false` | Log what the exporter is doing |

## Design rules

These are commitments, not aspirations — they are what the test suite checks.

- **Tracing never breaks the traced application.** Every export path swallows
  its own failures and reports them through `on_error`. A missing API key
  disables the exporter; it does not raise.
- **Tracing never blocks the traced application.** Exports run on a background
  thread (Python) or a promise chain (JS). No integration awaits network I/O
  on a framework's hot path.
- **No hidden dependencies.** Core is standard-library only. Every framework
  import is lazy and optional.
- **Secrets and blobs do not leave your process.** API keys in prompts are
  redacted by pattern, base64 data URLs are stripped, and oversized content is
  capped before it is sent.
- **Honest data.** When a framework cannot tell us something — token usage on
  a streaming call, tool schemas on a chat-completions path — the field is
  absent, not zero, and the docs say so.

## Repository layout

```
js/       @cognipeer/observability      TypeScript package (+ cognipeer-n8n CLI)
python/   cognipeer-observability       Python package
examples/ runnable examples per framework
docs/     the ingest contract every integration maps onto
```

Per-framework guides live with the
[Console documentation](https://cognipeer.github.io/console/guide/observability/overview)
rather than being duplicated here, so the two cannot drift.

## Roadmap

Two frameworks currently arrive through the OpenTelemetry route but have a
first-class seam that would capture strictly more:

- **CrewAI** — its public event bus (`crewai.events`: `BaseEventListener`,
  `crewai_event_bus`) reports token usage and tool lifecycle natively, where
  the OTel instrumentors resort to monkeypatching.
- **LlamaIndex** — `get_dispatcher()` with `BaseEventHandler` /
  `BaseSpanHandler` exposes real tool definitions on
  `LLMChatStartEvent.additional_kwargs["tools"]`, which the OTel path does not
  reliably carry.

Also wanted: **Mastra** (its `ObservabilityExporter` gives real 32-hex trace
ids and a first-class `conversationId`), and **C# / .NET**, where Semantic
Kernel and the Microsoft Agent Framework are OTel-native — so an OTLP endpoint
works today and a native package is a convenience, not a necessity.

## Contributing

Adding a framework is deliberately mechanical: implement the mapping onto the
event model in [`docs/data-model.md`](docs/data-model.md) and add an example.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT — see [LICENSE](LICENSE). Cognipeer Console itself is licensed separately.
