# n8n → Cognipeer Console

Three routes, fastest first. All of them read n8n's **execution run data**,
which is the only place its AI Agent nodes record prompts, completions, token
usage and tool results.

## 1. The bridge — one command, works on n8n Cloud

Polls the public REST API and mirrors finished executions. No files, no
restart, no licence, and it works on Cloud as well as self-hosted Community.

```bash
npx @cognipeer/observability cognipeer-n8n \
  --n8n-url https://n8n.acme.com \
  --n8n-api-key "$N8N_API_KEY" \
  --api-key "$COGNIPEER_API_KEY" \
  --once
```

`--once` mirrors the current page and exits — run it after a test execution to
verify the wiring end to end. Drop it to poll continuously.

Get `N8N_API_KEY` from n8n → **Settings → n8n API → Create an API key**.

| Flag | Env | Default |
|---|---|---|
| `--n8n-url` | `N8N_URL` | — |
| `--n8n-api-key` | `N8N_API_KEY` | — |
| `--api-key` | `COGNIPEER_API_KEY` | — |
| `--base-url` | `COGNIPEER_BASE_URL` | `https://console.cognipeer.com` |
| `--workflow-id` (repeatable) | `N8N_WORKFLOW_ID` (comma-separated) | all workflows |
| `--interval` (seconds) | `COGNIPEER_N8N_INTERVAL` | 15 |
| `--page-size` | `COGNIPEER_N8N_PAGE_SIZE` | 50 (n8n caps at 250) |
| `--since` (ISO) | `COGNIPEER_N8N_SINCE` | now — history before this is ignored |
| `--once` | — | poll forever |
| `--debug` | `COGNIPEER_DEBUG` | off |

To run it alongside n8n, use [`docker-compose.yml`](./docker-compose.yml):

```bash
export COGNIPEER_API_KEY=cpeer_… N8N_API_KEY=…
docker compose up
```

> **Keep the run data.** Set `EXECUTIONS_DATA_SAVE_ON_SUCCESS=all` on n8n. With
> n8n's pruning defaults the execution payloads are stripped and the traces
> arrive empty.

## 2. The external hook — push-based, self-hosted

Immediate rather than polled, at the cost of a file on the n8n container. See
the header of [`cognipeer-hook.js`](./cognipeer-hook.js) for the three install
steps.

```bash
EXTERNAL_HOOK_FILES=/opt/cognipeer/cognipeer-hook.js
COGNIPEER_API_KEY=cpeer_…
```

In queue mode set these on **every** instance type — main, workers and webhook
processors — or you will only mirror the executions that happen to run on main.

## 3. n8n's own OpenTelemetry — not recommended for agents

`N8N_OTEL_*` exports workflow and node spans, but for agent observability it is
the weakest of the three:

- the spans carry **no prompts, completions or model names**;
- **LLM token attributes are Enterprise-gated** (`n8n.node.custom.*`), so
  Community self-hosted gets spans with no token data at all;
- n8n's exporter speaks OTLP **protobuf**, while Console's `/traces` endpoint
  reads OTLP **JSON** — so it needs an OpenTelemetry Collector in between.

If you want it anyway, the full setup including a working collector config is
in the [n8n integration guide](https://cognipeer.github.io/console/guide/observability/n8n).

Two traps worth knowing before you try:

- `N8N_OTEL_EXPORTER_OTLP_ENDPOINT` must be the **base** URL — n8n appends
  `/v1/traces` itself. Pasting the full path yields `/v1/traces/v1/traces`.
- `N8N_OTEL_TRACES_PRODUCTION_ONLY` defaults to `true`, so clicking "Test
  workflow" in the editor produces nothing and looks like a broken integration.

## What lands, and what does not

| n8n | Console |
|---|---|
| execution | session |
| node run | event |
| `ai_languageModel` sub-node | `ai_call` with prompts, completions and tokens |
| `ai_tool` sub-node | `tool_call` with input and output |
| `ai_embedding` / `ai_vectorStore` | `embedding` / `retrieval` |
| workflow name | agent name |
| `$execution.customData.threadId` | `threadId` |

Known limits of the run-data route, stated plainly:

- **No cached-token counts.** n8n records only prompt and completion tokens.
- **Estimated tokens are common.** When a provider reports no usage (streaming,
  cancellation) n8n substitutes a tiktoken estimate; those events are marked
  `metadata.tokensEstimated` so a cost report can exclude them.
- **Tool definitions are reconstructed** from the workflow JSON rather than
  observed on the call, and carry
  `metadata.toolDefinitionsSource: 'n8n-workflow-json'`.
- **No first-class conversation id.** Set one in a Code node with
  `$execution.customData.set('threadId', chatId)` and it is picked up
  automatically.
