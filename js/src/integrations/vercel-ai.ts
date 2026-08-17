/**
 * Vercel AI SDK integration.
 *
 * The AI SDK moved its telemetry seam twice in two majors, so this file offers
 * three routes. Pick by the `ai` version you are on:
 *
 * | `ai` version | Recommended | How |
 * |---|---|---|
 * | 7.x | native telemetry integration | `await installVercelAITracing()` |
 * | 6.x | native telemetry integration | `await installVercelAITracing()` |
 * | 3.x – 5.x | tracer injection | `experimental_telemetry: cognipeerTelemetry({...})` |
 * | any | middleware | `withCognipeerTracing(model)` |
 *
 * **Why three.** `ai@7` deleted OpenTelemetry from the core package outright —
 * there is no `tracer` option left to inject into — and replaced it with a
 * first-class `Telemetry` callback interface that hands over structured tool
 * definitions, flat usage with a cache breakdown and per-tool timings. That is
 * strictly the best data available, so it is the recommended route wherever it
 * exists (6.x and up). On 3.x–5.x the only seam is OpenTelemetry, so
 * {@link CognipeerAISDKTracer} implements just enough of the OTel `Tracer`
 * surface for the SDK to talk to it — no `@opentelemetry/*` install needed. It
 * also works with `@ai-sdk/otel` on v7 if you would rather stay on OTLP.
 *
 * The middleware route ({@link cognipeerMiddleware}) is the only one that is
 * portable across every major unchanged, and it sees the exact wire prompt and
 * raw tool schemas. Its limit is scope: it wraps ONE model call, so it sees no
 * agent loop, no client-side tool execution and no step hierarchy. Reach for it
 * when you want provider-level truth or when you are pinned to a version whose
 * telemetry seam you do not want to depend on — and pair it with `trace()` so
 * the calls of one run land in one session.
 *
 * ```ts
 * import { init, trace } from '@cognipeer/observability';
 * import { installVercelAITracing } from '@cognipeer/observability/vercel-ai';
 *
 * init({ apiKey: 'cpeer_…', agent: { name: 'support-bot' } });
 * await installVercelAITracing();
 *
 * await generateText({
 *   model, prompt, tools,
 *   runtimeContext: { threadId: 'conv-42' },
 *   telemetry: { functionId: 'chat-turn', includeRuntimeContext: { threadId: true } },
 * });
 * ```
 *
 * Nothing here imports `ai` — not even its types. The type names moved with
 * every major (`LanguageModelV2Middleware` → `LanguageModelV4Middleware`,
 * `TelemetryIntegration` → `Telemetry`), so a `import type` would fail to
 * compile for users on the other major. Everything is typed structurally
 * instead, and the one place a runtime value is genuinely needed
 * (`registerTelemetry`) is reached through an optional dynamic import.
 */

import { getClient, type CognipeerObservability } from '../core/client';
import { getCurrentSession, getCurrentSpanKey, withContext } from '../core/context';
import { newSpanId, spanIdFrom } from '../core/ids';
import type { TraceSession } from '../core/session';
import type {
  TraceAgent,
  TraceEvent,
  TraceResponseFormat,
  TraceSection,
  TraceToolDefinition,
} from '../core/types';

/* ------------------------------------------------------------------ */
/*  Shared options                                                     */
/* ------------------------------------------------------------------ */

export interface VercelAITracingOptions {
  /** Client to send through. Defaults to the process-wide one. */
  client?: CognipeerObservability;
  /** Agent identity for sessions this integration opens. */
  agent?: TraceAgent;
  /**
   * Conversation id. The AI SDK has no session concept of its own, so without
   * this — or a `threadId` in `runtimeContext` / telemetry metadata — every
   * call becomes its own unattached session in the Console.
   */
  threadId?: string;
  /** Extra metadata attached to every event this integration emits. */
  metadata?: Record<string, unknown>;
}

/** Metadata keys treated as a conversation id, most specific first. */
const THREAD_ID_KEYS = ['threadId', 'thread_id', 'sessionId', 'session_id', 'conversationId'];

/* ------------------------------------------------------------------ */
/*  Usage normalisation                                                */
/* ------------------------------------------------------------------ */

interface NormalizedUsage {
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  totalTokens?: number;
}

/**
 * Normalize every usage shape the AI SDK has shipped.
 *
 * There are three, and feeding one to a reader written for another yields
 * all-`undefined` tokens with no error:
 *
 * 1. telemetry events (v6/v7) — flat, with a details object:
 *    `{ inputTokens, outputTokens, totalTokens,
 *       inputTokenDetails: { noCacheTokens, cacheReadTokens, cacheWriteTokens } }`
 * 2. provider results on v7 (`LanguageModelV4Usage`) — NESTED:
 *    `{ inputTokens: { total, noCache, cacheRead, cacheWrite },
 *       outputTokens: { total, text, reasoning } }`
 * 3. provider results on v3–v5 — flat, and v3/v4 spell it
 *    `promptTokens`/`completionTokens`.
 *
 * In all three the input figure is the TOTAL including cache reads, which is
 * exactly what the Console wants: it prices `cachedInputTokens` as a subset of
 * `inputTokens`. No arithmetic needed — unlike raw Anthropic usage, where the
 * cache buckets sit outside the input count.
 *
 * Missing values stay `undefined`. Reporting a real zero for "the provider did
 * not tell us" would quietly under-report a whole model's spend.
 */
export function normalizeAISDKUsage(usage: unknown): NormalizedUsage {
  const raw = asRecord(usage);
  if (!raw) return {};

  // Shape 2: nested provider usage.
  const nestedInput = asRecord(raw.inputTokens);
  const nestedOutput = asRecord(raw.outputTokens);
  if (nestedInput || nestedOutput) {
    const inputTokens = num(nestedInput?.total);
    const outputTokens = num(nestedOutput?.total);
    return dropUndefined({
      inputTokens,
      outputTokens,
      cachedInputTokens: num(nestedInput?.cacheRead),
      totalTokens:
        num(raw.totalTokens) ??
        (inputTokens !== undefined && outputTokens !== undefined
          ? inputTokens + outputTokens
          : undefined),
    });
  }

  // Shapes 1 and 3: flat, with the v3/v4 spellings as a fallback.
  const details = asRecord(raw.inputTokenDetails);
  return dropUndefined({
    inputTokens: num(raw.inputTokens) ?? num(raw.promptTokens),
    outputTokens: num(raw.outputTokens) ?? num(raw.completionTokens),
    cachedInputTokens:
      num(details?.cacheReadTokens) ??
      num(raw.cachedInputTokens) ??
      num(raw.cacheReadInputTokens),
    totalTokens: num(raw.totalTokens),
  });
}

/* ------------------------------------------------------------------ */
/*  Tool definitions                                                   */
/* ------------------------------------------------------------------ */

/**
 * Normalize the tool menu offered to the model.
 *
 * The schema key moved: v3–v5 providers use `parameters`, v7 uses
 * `inputSchema`, and the OTel attribute `ai.prompt.tools` carries the whole
 * list pre-stringified (sometimes as an array of JSON strings, sometimes as one
 * JSON string). All of it is accepted here so the Console gets a schema
 * whichever route the data arrived by.
 */
export function normalizeAISDKTools(tools: unknown): TraceToolDefinition[] | undefined {
  const list = Array.isArray(tools) ? tools : parseMaybeJson(tools);
  if (!Array.isArray(list)) return undefined;

  const out: TraceToolDefinition[] = [];
  for (const entry of list) {
    const item = asRecord(typeof entry === 'string' ? parseMaybeJson(entry) : entry);
    if (!item) continue;
    // Some providers nest the definition under `function`, OpenAI-style.
    const fn = asRecord(item.function) ?? item;
    const name = typeof fn.name === 'string' ? fn.name : undefined;
    if (!name) {
      // Provider-defined tools (web_search, …) have a type but no name.
      const type = typeof item.type === 'string' ? item.type : undefined;
      if (type && type !== 'function') out.push({ name: type });
      continue;
    }
    const parameters =
      asRecord(fn.parameters) ?? asRecord(fn.inputSchema) ?? asRecord(fn.input_schema);
    out.push({
      name,
      ...(typeof fn.description === 'string' && fn.description ? { description: fn.description } : {}),
      ...(parameters ? { parameters } : {}),
    });
  }
  return out.length > 0 ? out : undefined;
}

/**
 * Normalize the structured-output contract the model was called with.
 *
 * The AI SDK expresses it two ways depending on the route:
 *   - the LanguageModelV2 call options carry
 *     `responseFormat: { type: 'json', schema?, name?, description? }`;
 *   - `generateObject`'s telemetry splits it into `ai.schema` (a JSON-encoded
 *     schema) plus `ai.schema.name` and `ai.settings.output`.
 * Both land on our flat `{type, name?, strict?, schema?}` shape, so the
 * Console renders one thing whichever route the run took.
 */
export function normalizeAISDKResponseFormat(
  responseFormat: unknown,
  attributes?: Record<string, unknown>,
): TraceResponseFormat | undefined {
  const raw = asRecord(responseFormat) ?? asRecord(parseMaybeJson(responseFormat));
  if (raw) {
    const rawType = typeof raw.type === 'string' ? raw.type : undefined;
    const jsonSchema = asRecord(raw.json_schema) ?? asRecord(raw.jsonSchema);
    const source = jsonSchema ?? raw;
    const schema = asRecord(source.schema) ?? asRecord(parseMaybeJson(source.schema));
    if (rawType || schema) {
      // The SDK spells JSON mode `json`; the wire (and the Console) spell it
      // `json_schema` when a schema rides along, `json_object` when not.
      const type = rawType === 'json' ? (schema ? 'json_schema' : 'json_object') : (rawType ?? 'json_schema');
      return {
        type,
        ...(typeof source.name === 'string' && source.name ? { name: source.name } : {}),
        ...(typeof source.strict === 'boolean' ? { strict: source.strict } : {}),
        ...(schema ? { schema } : {}),
      };
    }
  }

  if (!attributes) return undefined;
  const schema = asRecord(parseMaybeJson(attributes['ai.schema']));
  if (schema) {
    const name = attributes['ai.schema.name'];
    return {
      type: 'json_schema',
      ...(typeof name === 'string' && name ? { name } : {}),
      schema,
    };
  }
  // `output: 'no-schema'` is still JSON mode — the model was told to emit JSON.
  return attributes['ai.settings.output'] === 'no-schema' ? { type: 'json_object' } : undefined;
}

/* ------------------------------------------------------------------ */
/*  Route 1 — native telemetry integration (ai@6, ai@7)                */
/* ------------------------------------------------------------------ */

/** Per-invocation state, keyed by the SDK's stable `callId`. */
interface CallState {
  session: TraceSession;
  /** True when we opened the session and are therefore responsible for it. */
  owned: boolean;
  /** Model-call span keys awaiting their `…End` event, keyed by callId. */
  modelSpans: Map<string, string>;
  toolSpans: Map<string, string>;
  /**
   * Span keys for tool executions that arrived without a `toolCallId`, queued
   * per tool name. Start and end must agree on a key or the arguments and the
   * result land on two unrelated events, so the anonymous case pairs FIFO by
   * name instead of inventing a fresh id on each side.
   */
  anonymousToolSpans: Map<string, string[]>;
}

/**
 * Implements the AI SDK's native `Telemetry` interface (ai@7) and its `ai@6`
 * predecessor `TelemetryIntegration`, structurally.
 *
 * Both are supported by one object because v7 kept the v6 method names as
 * deprecated aliases — but only the v7 names fire the newer events, so the v6
 * names are wired to the same handlers rather than left out.
 */
export class CognipeerAISDKTelemetry {
  private readonly client: CognipeerObservability;
  private readonly options: VercelAITracingOptions;
  private readonly calls = new Map<string, CallState>();

  constructor(options: VercelAITracingOptions = {}) {
    this.options = options;
    this.client = options.client ?? getClient();
  }

  // ── Operation lifecycle ────────────────────────────────────────────────

  /**
   * `runtimeContext` is populated on this event and on `onStepStart` only — it
   * is `undefined` on the model-call and tool events. The thread id therefore
   * has to be captured here and remembered for the rest of the call.
   */
  onStart = (event: unknown): void => {
    const e = asRecord(event);
    const callId = str(e?.callId);
    if (!callId || this.calls.has(callId)) return;

    const runtimeContext = asRecord(e?.runtimeContext);
    const ambient = getCurrentSession();
    const session =
      ambient ??
      this.client.startSession({
        threadId:
          this.options.threadId ??
          pickThreadId(runtimeContext) ??
          pickThreadId(asRecord(e?.metadata)),
        agent: {
          name: str(e?.functionId) ?? str(e?.operationId) ?? 'vercel-ai',
          ...(str(e?.modelId) ? { model: str(e?.modelId) } : {}),
          ...this.options.agent,
        },
      });

    this.calls.set(callId, {
      session,
      owned: !ambient,
      modelSpans: new Map(),
      toolSpans: new Map(),
      anonymousToolSpans: new Map(),
    });
  };

  /**
   * `onEnd.usage` is the AGGREGATE across every step. The session summary is
   * already accumulated from the per-call events, so recording it again here
   * would double every token count on a multi-step run.
   */
  onEnd = (event: unknown): void => {
    // No explicit status: the session derives it from the events it collected,
    // so a run whose tool failed still reports as failed. Forcing 'success'
    // here would hide exactly the runs an operator is looking for.
    this.finish(asRecord(event), undefined);
  };

  /**
   * Streaming runs that die mid-flight fire this and nothing else. Without it
   * the session never closes and shows as in-progress forever.
   */
  onAbort = (event: unknown): void => {
    this.finish(asRecord(event), 'error', 'Run aborted');
  };

  onError = (event: unknown): void => {
    // Prefer the call the payload names. Older shapes pass the bare Error, so
    // also look at `.error` before treating the argument itself as the cause.
    const record = asRecord(event);
    const callId = str(record?.callId);
    const cause = record && 'error' in record ? record.error : event;

    if (callId) {
      this.finish({ callId }, 'error', cause);
      return;
    }

    // No call id. Failing EVERY open call would mark unrelated concurrent runs
    // as errored with a foreign error attached — and this integration is
    // installed process-wide, so on a server those runs are other users'. The
    // most recently started call is the only one plausibly responsible, and
    // `Map` preserves insertion order.
    const openCalls = [...this.calls.keys()];
    const mostRecent = openCalls[openCalls.length - 1];
    if (mostRecent !== undefined) this.finish({ callId: mostRecent }, 'error', cause);
  };

  // ── Model calls ────────────────────────────────────────────────────────

  onLanguageModelCallStart = (event: unknown): void => {
    const e = asRecord(event);
    const state = this.stateFor(e);
    if (!state || !e) return;

    const key = newSpanId();
    state.modelSpans.set(modelCallKey(e), key);

    const sections: TraceSection[] = [];
    const instructions = str(e.instructions);
    if (instructions) {
      sections.push({ kind: 'message', label: 'Instructions', role: 'system', content: instructions });
    }
    sections.push(...promptToSections(e.prompt ?? e.messages));

    state.session.openSpan(key, {
      type: 'ai_call',
      label: str(e.modelId) ?? 'model call',
      parentKey: getCurrentSpanKey(),
      model: str(e.modelId),
      actor: { scope: 'model', name: str(e.modelId) ?? 'model' },
      sections,
      metadata: { ...this.options.metadata, provider: str(e.provider) },
      // v7 hands these over as a live array with full JSON Schema, under
      // `inputSchema` rather than `parameters`.
      toolDefinitions: normalizeAISDKTools(e.tools),
      responseFormat: normalizeAISDKResponseFormat(e.responseFormat),
    });
  };

  onLanguageModelCallEnd = (event: unknown): void => {
    const e = asRecord(event);
    const state = this.stateFor(e);
    if (!state || !e) return;

    // Exact match first; otherwise pair with the oldest span still open. Model
    // calls inside one operation are sequential — a step's call finishes before
    // the next begins — so oldest-first is the correct fallback when the SDK
    // version does not carry a per-call discriminator.
    const lookup = modelCallKey(e);
    const [pairedKey, spanKey] = state.modelSpans.has(lookup)
      ? ([lookup, state.modelSpans.get(lookup)] as const)
      : ([...state.modelSpans.entries()][0] ?? ([undefined, undefined] as const));
    if (!pairedKey || !spanKey) return;
    state.modelSpans.delete(pairedKey);
    const key = spanKey;

    const performance = asRecord(e.performance);
    state.session.closeSpan(key, {
      status: e.error ? 'error' : 'success',
      model: str(e.modelId),
      ...normalizeAISDKUsage(e.usage),
      sections: contentToSections(e.content ?? e.text),
      metadata: dropUndefined({
        finishReason: str(e.finishReason),
        timeToFirstOutputMs: num(performance?.timeToFirstOutputMs),
      }),
      error: e.error,
    });
  };

  // ── Tools ──────────────────────────────────────────────────────────────

  onToolExecutionStart = (event: unknown): void => {
    const e = asRecord(event);
    const state = this.stateFor(e);
    if (!state || !e) return;

    const toolCall = asRecord(e.toolCall) ?? {};
    const name = str(toolCall.toolName) ?? str(e.toolName) ?? 'tool';
    const toolCallId = str(toolCall.toolCallId) ?? str(e.toolCallId);
    const key = toolCallId ? spanIdFrom(toolCallId) : newSpanId();

    if (toolCallId) {
      state.toolSpans.set(toolCallId, key);
    } else {
      // Queue under the tool name so the matching end can find this exact
      // span — see `anonymousToolSpans`.
      const queue = state.anonymousToolSpans.get(name) ?? [];
      queue.push(key);
      state.anonymousToolSpans.set(name, queue);
    }

    state.session.openSpan(key, {
      type: 'tool_call',
      label: name,
      parentKey: getCurrentSpanKey(),
      toolName: name,
      actor: { scope: 'tool', name },
      sections: [
        { kind: 'tool_call', label: 'Arguments', tool: name, content: toolCall.input ?? e.input },
      ],
      // The provider's tool-call id correlates this execution with the
      // `tool_call` block in the assistant message that requested it. It rides
      // in metadata because `SpanInit` has no `toolExecutionId` field yet; the
      // span id is derived from it too, so correlation survives either way.
      metadata: dropUndefined({ ...this.options.metadata, toolCallId }),
    });
  };

  onToolExecutionEnd = (event: unknown): void => {
    const e = asRecord(event);
    const state = this.stateFor(e);
    if (!state || !e) return;

    const toolCall = asRecord(e.toolCall) ?? {};
    const name = str(toolCall.toolName) ?? str(e.toolName) ?? 'tool';
    const toolCallId = str(toolCall.toolCallId) ?? str(e.toolCallId);
    const key = this.takeToolSpan(state, toolCallId, name);

    // v7 discriminates on `toolOutput.type`. Some docs still describe an
    // `event.success` boolean, which does not exist on the widened event.
    const output = asRecord(e.toolOutput);
    const failed = output?.type === 'tool-error' || Boolean(e.error);
    const value = failed ? undefined : (output?.output ?? e.output);

    state.session.closeSpan(key, {
      status: failed ? 'error' : 'success',
      toolName: name,
      sections: failed
        ? []
        : [{ kind: 'tool_result', label: 'Result', tool: name, content: value }],
      metadata: dropUndefined({ toolExecutionMs: num(e.toolExecutionMs) }),
      error: failed ? (output?.error ?? e.error ?? 'Tool execution failed') : undefined,
    });
  };

  // ── Async-context wrappers ─────────────────────────────────────────────

  /**
   * The SDK offers these specifically because plain callbacks cannot establish
   * async parent/child context. Running the work inside our context makes any
   * nested `observe()` — and the provider's own instrumentation — attach to the
   * right parent instead of the session root.
   */
  executeLanguageModelCall = <T>(options: {
    callId: string;
    execute: () => PromiseLike<T>;
  }): PromiseLike<T> => this.inContext(options.callId, options.execute);

  executeTool = <T>(options: {
    callId: string;
    toolCallId: string;
    execute: () => PromiseLike<T>;
  }): PromiseLike<T> => this.inContext(options.callId, options.execute, options.toolCallId);

  // ── v6 aliases ─────────────────────────────────────────────────────────

  /** `ai@6` name for {@link onToolExecutionStart}. */
  onToolCallStart = (event: unknown): void => this.onToolExecutionStart(event);
  /** `ai@6` name for {@link onToolExecutionEnd}. */
  onToolCallFinish = (event: unknown): void => this.onToolExecutionEnd(event);
  /** `ai@6` name for {@link onEnd}. */
  onFinish = (event: unknown): void => this.onEnd(event);

  // ── Internals ──────────────────────────────────────────────────────────

  /**
   * The span key this tool execution's start opened, removed from the pending
   * set.
   *
   * When nothing matches — an end with no start, which the SDK should not do
   * but the handlers are public API — a fresh span is opened first, so the
   * result still arrives as a `tool_call` event rather than as the untyped
   * `span` that closing an unknown key would fabricate.
   */
  private takeToolSpan(state: CallState, toolCallId: string | undefined, name: string): string {
    if (toolCallId) {
      const known = state.toolSpans.get(toolCallId);
      state.toolSpans.delete(toolCallId);
      if (known) return known;
    } else {
      const queue = state.anonymousToolSpans.get(name);
      const pending = queue?.shift();
      if (queue && queue.length === 0) state.anonymousToolSpans.delete(name);
      if (pending) return pending;
    }

    const key = toolCallId ? spanIdFrom(toolCallId) : newSpanId();
    state.session.openSpan(key, {
      type: 'tool_call',
      label: name,
      parentKey: getCurrentSpanKey(),
      toolName: name,
      actor: { scope: 'tool', name },
      metadata: dropUndefined({ ...this.options.metadata, toolCallId, unpaired: true }),
    });
    return key;
  }

  private stateFor(event: Record<string, unknown> | undefined): CallState | undefined {
    const callId = str(event?.callId);
    if (!callId) return undefined;
    // A per-call `telemetry.integrations` array can start a run without ever
    // firing `onStart` on this instance; open the session lazily rather than
    // dropping the whole run.
    if (!this.calls.has(callId)) this.onStart({ callId });
    return this.calls.get(callId);
  }

  private inContext<T>(
    callId: string,
    execute: () => PromiseLike<T>,
    spanKey?: string,
  ): PromiseLike<T> {
    const state = this.calls.get(callId);
    if (!state) return execute();
    return withContext(
      { session: state.session, ...(spanKey ? { spanKey: spanIdFrom(spanKey) } : {}) },
      execute,
    );
  }

  private finish(
    event: Record<string, unknown> | undefined,
    status: 'error' | undefined,
    error?: unknown,
  ): void {
    const callId = str(event?.callId);
    if (!callId) return;
    const state = this.calls.get(callId);
    if (!state) return;
    this.calls.delete(callId);
    // Only end a session we opened — an ambient one belongs to its owner.
    if (state.owned) void state.session.end({ status, error });
  }
}

let installed: CognipeerAISDKTelemetry | undefined;

export interface InstallVercelAITracingOptions extends VercelAITracingOptions {
  /**
   * The SDK's registration function. Pass `registerTelemetry` (ai@7) or
   * `registerTelemetryIntegration` (ai@6) explicitly to skip the dynamic
   * `import('ai')` — useful in bundlers that cannot resolve it.
   */
  register?: (integration: unknown) => void;
}

/**
 * Register a telemetry integration globally, so every `generateText` /
 * `streamText` in the process is traced. Requires `ai@6` or newer.
 *
 * Guarded against double registration: the SDK's registry is append-only, so a
 * second call — trivially easy under Next.js HMR or a serverless warm start —
 * would otherwise double every event.
 *
 * ⚠️ A per-call `telemetry: { integrations: [...] }` REPLACES the global
 * registry for that call rather than merging with it, so a caller passing their
 * own array (or an empty one) silently opts out of Console tracing. Pass
 * {@link CognipeerAISDKTelemetry} in that array to keep both.
 */
export async function installVercelAITracing(
  options: InstallVercelAITracingOptions = {},
): Promise<CognipeerAISDKTelemetry> {
  if (installed) return installed;
  const integration = new CognipeerAISDKTelemetry(options);

  let register = options.register;
  if (!register) {
    try {
      // Indirected through a variable so TypeScript does not try to resolve
      // `ai` at compile time — it is an optional peer, and a user who only
      // installed the LangChain integration must still be able to typecheck.
      const specifier = 'ai';
      const sdk = (await import(/* @vite-ignore */ specifier)) as Record<string, unknown>;
      // v7 takes a variadic list; v6 takes exactly one integration. Calling
      // either with a single argument is correct.
      const fn = sdk.registerTelemetry ?? sdk.registerTelemetryIntegration;
      if (typeof fn === 'function') register = fn as (integration: unknown) => void;
    } catch {
      // `ai` not installed, or too old to export a registry.
    }
  }

  if (!register) {
    getClient().config.logger.warn(
      'vercel-ai: no telemetry registry found (needs ai@6+). ' +
        'Use cognipeerTelemetry() for ai@3–5, or withCognipeerTracing() on any version.',
    );
    return integration;
  }

  register(integration);
  installed = integration;
  return integration;
}

/** Forget the globally installed integration. Intended for tests. */
export function resetVercelAITracing(): void {
  installed = undefined;
}

/* ------------------------------------------------------------------ */
/*  Route 2 — OpenTelemetry tracer injection (ai@3 – ai@6, @ai-sdk/otel) */
/* ------------------------------------------------------------------ */

/** The subset of the OTel `Span` interface the AI SDK actually calls. */
interface ShimSpan {
  spanContext(): { traceId: string; spanId: string; traceFlags: number };
  setAttribute(key: string, value: unknown): ShimSpan;
  setAttributes(attributes: Record<string, unknown>): ShimSpan;
  addEvent(name: string, attributes?: Record<string, unknown>): ShimSpan;
  setStatus(status: { code: number; message?: string }): ShimSpan;
  updateName(name: string): ShimSpan;
  recordException(error: unknown): void;
  isRecording(): boolean;
  end(endTime?: number): void;
}

/**
 * A minimal OpenTelemetry `Tracer` that writes straight to Cognipeer.
 *
 * Pass it as `experimental_telemetry.tracer` (ai@3–6) or as
 * `new OpenTelemetry({ tracer })` from `@ai-sdk/otel` (ai@7). Either way the
 * spans never touch an OTel SDK — no provider, no exporter, no
 * `@opentelemetry/*` dependency — so tracing works with one option and no
 * bootstrap file.
 *
 * If you already run an OTel SDK, do NOT use this: point the AI SDK at your own
 * tracer and export to the Console's OTLP endpoint instead, so AI spans stay in
 * the same trace as your HTTP and database spans.
 */
export class CognipeerAISDKTracer {
  private readonly client: CognipeerObservability;

  constructor(private readonly options: VercelAITracingOptions = {}) {
    this.client = options.client ?? getClient();
  }

  /**
   * Called by the AI SDK for every operation. The callback form is what
   * establishes nesting: the child spans created inside `fn` see this span as
   * their parent through the package's async context.
   */
  startActiveSpan<T>(name: string, ...rest: unknown[]): T {
    const fn = rest[rest.length - 1] as (span: ShimSpan) => T;
    const attributes = asRecord(asRecord(rest[0])?.attributes) ?? {};
    if (typeof fn !== 'function') {
      // Defensive: an overload we do not know about. Do not break the caller.
      return undefined as T;
    }

    const { span, session, spanKey } = this.createSpan(name, attributes);
    return withContext({ session, spanKey }, () => fn(span));
  }

  /** Non-scoped variant. Produces an event but cannot establish nesting. */
  startSpan(name: string, options?: unknown): ShimSpan {
    return this.createSpan(name, asRecord(asRecord(options)?.attributes) ?? {}).span;
  }

  private createSpan(
    name: string,
    initialAttributes: Record<string, unknown>,
  ): { span: ShimSpan; session: TraceSession; spanKey: string } {
    const attributes: Record<string, unknown> = { ...initialAttributes };
    const parentSession = getCurrentSession();
    const parentKey = getCurrentSpanKey();
    const session =
      parentSession ??
      this.client.startSession({
        threadId: this.options.threadId ?? threadIdFromAttributes(attributes),
        agent: {
          name: str(attributes['ai.telemetry.functionId']) ?? operationName(name),
          ...(str(attributes['ai.model.id']) ? { model: str(attributes['ai.model.id']) } : {}),
          ...this.options.agent,
        },
      });
    const ownsSession = !parentSession;
    const spanKey = newSpanId();
    const startedAt = Date.now();
    let ended = false;
    let failure: unknown;

    const span: ShimSpan = {
      spanContext: () => ({ traceId: session.traceId, spanId: spanKey, traceFlags: 1 }),
      setAttribute(key, value) {
        attributes[key] = value;
        return span;
      },
      setAttributes(next) {
        Object.assign(attributes, next);
        return span;
      },
      addEvent(eventName, eventAttributes) {
        // The SDK marks stream milestones (`ai.stream.firstChunk`) this way.
        attributes[`event.${eventName}`] = eventAttributes ?? true;
        return span;
      },
      setStatus(status) {
        // OTel StatusCode.ERROR === 2.
        if (status.code === 2) failure = status.message ?? 'Span error';
        return span;
      },
      updateName: () => span,
      recordException(error) {
        failure = error;
      },
      isRecording: () => !ended,
      end: () => {
        if (ended) return;
        ended = true;
        this.emit({ name, attributes, session, spanKey, parentKey, startedAt, failure });
        if (ownsSession) void session.end(failure ? { status: 'error', error: failure } : {});
      },
    };

    return { span, session, spanKey };
  }

  private emit(args: {
    name: string;
    attributes: Record<string, unknown>;
    session: TraceSession;
    spanKey: string;
    parentKey?: string;
    startedAt: number;
    failure: unknown;
  }): void {
    const { attributes, name } = args;
    const kind = classifySpan(name);

    const event: TraceEvent = {
      id: args.spanKey,
      type: kind.eventType,
      label: str(attributes['ai.telemetry.functionId']) ?? operationName(name),
      spanId: args.spanKey,
      parentSpanId: args.parentKey ? spanIdFrom(args.parentKey) : args.session.rootSpanId,
      timestamp: new Date(args.startedAt).toISOString(),
      durationMs: Math.max(0, Date.now() - args.startedAt),
      status: args.failure ? 'error' : 'success',
      sections: sectionsFromAttributes(attributes, kind),
      metadata: metadataFromAttributes(attributes, this.options.metadata),
      ...(args.failure ? { error: errorMessage(args.failure) } : {}),
    };

    const model =
      str(attributes['ai.model.id']) ??
      str(attributes['gen_ai.request.model']) ??
      str(attributes['gen_ai.response.model']);
    if (model) event.model = model;

    if (kind.isTool) {
      const toolName = str(attributes['ai.toolCall.name']) ?? str(attributes['gen_ai.tool.name']);
      if (toolName) {
        event.toolName = toolName;
        event.label = toolName;
        event.actor = { scope: 'tool', name: toolName };
      }
      const toolCallId =
        str(attributes['ai.toolCall.id']) ?? str(attributes['gen_ai.tool.call.id']);
      if (toolCallId) event.toolExecutionId = toolCallId;
    } else if (kind.isModelCall) {
      event.actor = { scope: 'model', name: model ?? 'model' };
    }

    // Tokens come only from the leaf model-call span. The outer operation span
    // repeats the same figures as a run-level aggregate, so counting both would
    // double every multi-step run.
    if (kind.isModelCall) {
      const usage = normalizeAISDKUsage({
        inputTokens:
          attributes['ai.usage.inputTokens'] ??
          attributes['ai.usage.promptTokens'] ??
          attributes['gen_ai.usage.input_tokens'],
        outputTokens:
          attributes['ai.usage.outputTokens'] ??
          attributes['ai.usage.completionTokens'] ??
          attributes['gen_ai.usage.output_tokens'],
        cachedInputTokens:
          attributes['ai.usage.cachedInputTokens'] ??
          attributes['gen_ai.usage.cache_read.input_tokens'],
        totalTokens: attributes['ai.usage.totalTokens'],
      });
      Object.assign(event, usage);
    }

    const tools = normalizeAISDKTools(
      attributes['ai.prompt.tools'] ?? attributes['gen_ai.tool.definitions'],
    );
    if (tools) event.toolDefinitions = tools;

    const responseFormat = normalizeAISDKResponseFormat(
      attributes['ai.prompt.responseFormat'] ?? attributes['ai.response.format'],
      attributes,
    );
    if (responseFormat) event.responseFormat = responseFormat;

    args.session.record(event);
  }
}

export interface CognipeerTelemetrySettings extends Record<string, unknown> {
  isEnabled: true;
  functionId?: string;
  metadata?: Record<string, unknown>;
  tracer?: CognipeerAISDKTracer;
  recordInputs?: boolean;
  recordOutputs?: boolean;
}

/**
 * Build the `experimental_telemetry` value for `ai@3` – `ai@6`.
 *
 * ```ts
 * await generateText({
 *   model, prompt,
 *   experimental_telemetry: cognipeerTelemetry({ functionId: 'chat-turn', threadId: 'conv-42' }),
 * });
 * ```
 *
 * `isEnabled` is set for you — it defaults to `false` in the SDK, which is the
 * usual reason a first attempt produces no traces at all. The `threadId` is
 * written into `metadata` as well as used directly, so it survives whichever
 * path the SDK takes.
 *
 * On `ai@7` this option no longer exists: use {@link installVercelAITracing}.
 */
export function cognipeerTelemetry(
  options: VercelAITracingOptions & {
    functionId?: string;
    recordInputs?: boolean;
    recordOutputs?: boolean;
  } = {},
): CognipeerTelemetrySettings {
  const { functionId, recordInputs, recordOutputs, metadata, threadId, ...tracerOptions } = options;
  return dropUndefined({
    isEnabled: true as const,
    functionId,
    recordInputs,
    recordOutputs,
    // Attribute values must be primitives, so only scalars survive the trip.
    metadata: dropUndefined({ ...metadata, threadId }) as Record<string, unknown>,
    tracer: new CognipeerAISDKTracer({ ...tracerOptions, metadata, threadId }),
  }) as CognipeerTelemetrySettings;
}

/* ------------------------------------------------------------------ */
/*  Route 3 — language-model middleware (every version)                 */
/* ------------------------------------------------------------------ */

/** Structural shape of an AI SDK language model, across majors. */
interface LanguageModelLike {
  modelId?: string;
  provider?: string;
  doGenerate(options: unknown): PromiseLike<unknown>;
  doStream(options: unknown): PromiseLike<unknown>;
  [key: string]: unknown;
}

/** Structural shape of the middleware object `wrapLanguageModel` accepts. */
export interface CognipeerLanguageModelMiddleware {
  /** Required discriminator on `ai@7` (`LanguageModelV4Middleware`). */
  specificationVersion: 'v4';
  /** Optional discriminator on `ai@5` (`LanguageModelV2Middleware`). */
  middlewareVersion: 'v2';
  wrapGenerate(options: {
    doGenerate: () => PromiseLike<unknown>;
    params: unknown;
    model: LanguageModelLike;
  }): Promise<unknown>;
  wrapStream(options: {
    doStream: () => PromiseLike<unknown>;
    params: unknown;
    model: LanguageModelLike;
  }): Promise<unknown>;
}

/**
 * Middleware that records every model call, with no OpenTelemetry involved.
 *
 * ```ts
 * import { wrapLanguageModel } from 'ai';
 * const model = wrapLanguageModel({ model: openai('gpt-4.1'), middleware: cognipeerMiddleware() });
 * ```
 *
 * Both version discriminators are set so one object satisfies `ai@5`'s
 * `middlewareVersion: 'v2'` and `ai@7`'s required `specificationVersion: 'v4'`;
 * each major reads only the key it knows and ignores the other.
 *
 * Wrap the surrounding request in `trace()` to group a run's calls into one
 * session — on its own this seam sees one model call at a time and has no way
 * to know which run it belongs to.
 */
export function cognipeerMiddleware(
  options: VercelAITracingOptions = {},
): CognipeerLanguageModelMiddleware {
  const client = options.client ?? getClient();

  const begin = (params: unknown, model: LanguageModelLike) => {
    const ambient = getCurrentSession();
    const session =
      ambient ??
      client.startSession({
        threadId: options.threadId,
        agent: { name: model.modelId ?? 'vercel-ai', model: model.modelId, ...options.agent },
      });
    const key = newSpanId();
    const p = asRecord(params) ?? {};

    session.openSpan(key, {
      type: 'ai_call',
      label: model.modelId ?? 'model call',
      parentKey: getCurrentSpanKey(),
      model: model.modelId,
      actor: { scope: 'model', name: model.modelId ?? 'model' },
      sections: promptToSections(p.prompt),
      metadata: dropUndefined({ ...options.metadata, provider: model.provider }),
      // At this seam the tool list is the raw provider payload — `parameters`
      // on v5 and below, `inputSchema` on v7.
      toolDefinitions: normalizeAISDKTools(p.tools),
      responseFormat: normalizeAISDKResponseFormat(p.responseFormat),
    });

    return { session, key, owned: !ambient };
  };

  const finish = (
    ctx: { session: TraceSession; key: string; owned: boolean },
    result: unknown,
    error?: unknown,
  ) => {
    const r = asRecord(result) ?? {};
    ctx.session.closeSpan(ctx.key, {
      status: error ? 'error' : 'success',
      ...(error ? {} : normalizeAISDKUsage(r.usage)),
      sections: error ? [] : contentToSections(r.content ?? r.text),
      metadata: dropUndefined({ finishReason: str(r.finishReason) }),
      error,
    });
    // Only force a status on failure; otherwise let the session derive it from
    // the events, so an errored call is never reported as a clean run.
    if (ctx.owned) void ctx.session.end(error ? { status: 'error', error } : {});
  };

  return {
    specificationVersion: 'v4',
    middlewareVersion: 'v2',

    async wrapGenerate({ doGenerate, params, model }) {
      const ctx = begin(params, model);
      try {
        const result = await withContext({ session: ctx.session, spanKey: ctx.key }, doGenerate);
        finish(ctx, result);
        return result;
      } catch (error) {
        finish(ctx, undefined, error);
        throw error;
      }
    },

    async wrapStream({ doStream, params, model }) {
      const ctx = begin(params, model);
      let result: Record<string, unknown>;
      try {
        result = (asRecord(
          await withContext({ session: ctx.session, spanKey: ctx.key }, doStream),
        ) ?? {}) as Record<string, unknown>;
      } catch (error) {
        finish(ctx, undefined, error);
        throw error;
      }

      // Usage and finish reason only exist on the stream's final part, so the
      // event must close when the stream drains — not when this returns.
      const collected: { usage?: unknown; finishReason?: unknown; text: string } = { text: '' };
      const stream = instrumentStream(
        result.stream,
        (part) => {
          const p = asRecord(part);
          if (!p) return;
          if (typeof p.delta === 'string') collected.text += p.delta;
          else if (p.type === 'text-delta' && typeof p.textDelta === 'string') {
            collected.text += p.textDelta;
          }
          if (p.usage) collected.usage = p.usage;
          if (p.finishReason) collected.finishReason = p.finishReason;
        },
        (error) => {
          finish(
            ctx,
            { usage: collected.usage, finishReason: collected.finishReason, text: collected.text },
            error,
          );
        },
      );

      return stream ? { ...result, stream } : result;
    },
  };
}

/**
 * Apply {@link cognipeerMiddleware} to a model without importing `ai`.
 *
 * ```ts
 * const model = withCognipeerTracing(openai('gpt-4.1'));
 * ```
 *
 * Equivalent to `wrapLanguageModel({ model, middleware: cognipeerMiddleware() })`
 * but implemented as a proxy, so it needs no runtime dependency on `ai` and
 * behaves the same on every major. Every property other than `doGenerate` and
 * `doStream` passes through untouched, including `specificationVersion`,
 * `modelId` and `provider`.
 */
export function withCognipeerTracing<T extends object>(
  model: T,
  options: VercelAITracingOptions = {},
): T {
  const middleware = cognipeerMiddleware(options);
  const target = model as unknown as LanguageModelLike;

  return new Proxy(model, {
    get(source, property, receiver) {
      if (property === 'doGenerate') {
        return (callOptions: unknown) =>
          middleware.wrapGenerate({
            doGenerate: () => target.doGenerate(callOptions),
            params: callOptions,
            model: target,
          });
      }
      if (property === 'doStream') {
        return (callOptions: unknown) =>
          middleware.wrapStream({
            doStream: () => target.doStream(callOptions),
            params: callOptions,
            model: target,
          });
      }
      // NOT `Reflect.get(source, property, receiver)`. Forwarding the proxy as
      // the receiver runs every accessor with `this === proxy`, so any getter
      // reading private state (`#field`, or its downleveled WeakMap form)
      // throws a TypeError into the traced call before a request is even made.
      const value = Reflect.get(source, property);

      // Binding is for the same reason — a method must keep its original
      // `this` — but it rewrites the value, which the Proxy `get` invariant
      // forbids for a non-configurable, non-writable own property. A frozen
      // model literal would otherwise throw on its first method read, so
      // return those untouched.
      if (typeof value !== 'function') return value;
      const descriptor = Reflect.getOwnPropertyDescriptor(source, property);
      if (descriptor && !descriptor.configurable && !descriptor.writable) return value;
      return value.bind(source);
    },
  }) as T;
}

/* ------------------------------------------------------------------ */
/*  Shared helpers                                                     */
/* ------------------------------------------------------------------ */

interface SpanKind {
  eventType: TraceEvent['type'];
  isModelCall: boolean;
  isTool: boolean;
}

/**
 * Classify a span by name, covering both naming schemes.
 *
 * `ai@3`–`ai@6` emit `ai.generateText` / `ai.generateText.doGenerate` /
 * `ai.toolCall`. `@ai-sdk/otel` on `ai@7` emits GenAI semantic-convention names
 * instead — `chat gpt-4.1`, `invoke_agent …`, `execute_tool search` — so a
 * matcher written for one scheme sees nothing from the other.
 */
function classifySpan(name: string): SpanKind {
  const lower = name.toLowerCase();

  if (lower === 'ai.toolcall' || lower.startsWith('execute_tool')) {
    return { eventType: 'tool_call', isModelCall: false, isTool: true };
  }
  if (lower.includes('embed') || lower.startsWith('embeddings') || lower.startsWith('rerank')) {
    // `.doEmbed` is the leaf that carries the token count.
    return {
      eventType: 'embedding',
      isModelCall: lower.endsWith('.doembed') || lower.startsWith('embeddings'),
      isTool: false,
    };
  }
  if (lower.endsWith('.dogenerate') || lower.endsWith('.dostream') || lower.startsWith('chat ')) {
    return { eventType: 'ai_call', isModelCall: true, isTool: false };
  }
  return { eventType: 'span', isModelCall: false, isTool: false };
}

/** `ai.generateText.doGenerate` → `generateText.doGenerate`. */
function operationName(name: string): string {
  return name.startsWith('ai.') ? name.slice(3) : name;
}

function sectionsFromAttributes(
  attributes: Record<string, unknown>,
  kind: SpanKind,
): TraceSection[] {
  const sections: TraceSection[] = [];

  if (kind.isTool) {
    const tool = str(attributes['ai.toolCall.name']) ?? str(attributes['gen_ai.tool.name']);
    const args = attributes['ai.toolCall.args'] ?? attributes['gen_ai.tool.call.arguments'];
    const result = attributes['ai.toolCall.result'] ?? attributes['gen_ai.tool.call.result'];
    if (args !== undefined) {
      sections.push({ kind: 'tool_call', label: 'Arguments', tool, content: parseMaybeJson(args) });
    }
    if (result !== undefined) {
      sections.push({ kind: 'tool_result', label: 'Result', tool, content: parseMaybeJson(result) });
    }
    return sections;
  }

  const system =
    attributes['ai.prompt.system'] ?? attributes['gen_ai.system_instructions'];
  if (system !== undefined) {
    sections.push({ kind: 'message', label: 'Instructions', role: 'system', content: system });
  }

  const messages =
    attributes['ai.prompt.messages'] ??
    attributes['gen_ai.input.messages'] ??
    attributes['ai.prompt'];
  sections.push(...promptToSections(parseMaybeJson(messages)));

  const output =
    attributes['ai.response.text'] ??
    attributes['ai.response.object'] ??
    attributes['gen_ai.output.messages'];
  if (output !== undefined) {
    sections.push(...contentToSections(parseMaybeJson(output)));
  }

  const toolCalls = attributes['ai.response.toolCalls'];
  const parsedCalls = parseMaybeJson(toolCalls);
  if (Array.isArray(parsedCalls)) {
    for (const call of parsedCalls) {
      const c = asRecord(call);
      if (!c) continue;
      const toolName = str(c.toolName) ?? str(c.name) ?? 'tool';
      sections.push({
        kind: 'tool_call',
        label: `Tool call: ${toolName}`,
        tool: toolName,
        content: parseMaybeJson(c.args ?? c.input),
      });
    }
  }

  return sections;
}

/** Everything the Console has no first-class field for, kept as metadata. */
function metadataFromAttributes(
  attributes: Record<string, unknown>,
  extra?: Record<string, unknown>,
): Record<string, unknown> {
  const metadata: Record<string, unknown> = { ...extra };
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith('ai.telemetry.metadata.')) {
      metadata[key.slice('ai.telemetry.metadata.'.length)] = value;
    }
  }
  const finishReason =
    str(attributes['ai.response.finishReason']) ?? attributes['gen_ai.response.finish_reasons'];
  if (finishReason !== undefined) metadata.finishReason = finishReason;
  const provider =
    str(attributes['ai.model.provider']) ??
    str(attributes['gen_ai.provider.name']) ??
    str(attributes['gen_ai.system']);
  if (provider) metadata.provider = provider;
  return metadata;
}

function threadIdFromAttributes(attributes: Record<string, unknown>): string | undefined {
  for (const key of THREAD_ID_KEYS) {
    const value = attributes[`ai.telemetry.metadata.${key}`] ?? attributes[key];
    if (typeof value === 'string' && value) return value;
  }
  const conversation = attributes['gen_ai.conversation.id'];
  return typeof conversation === 'string' && conversation ? conversation : undefined;
}

/**
 * Pairing key for a model call's start and end events.
 *
 * `callId` identifies the whole `generateText`/`streamText` invocation, not one
 * model call, so a multi-step agent loop reuses it for every step. `stepNumber`
 * (or a per-call id, where the version provides one) is what separates them.
 */
function modelCallKey(event: Record<string, unknown>): string {
  const callId = str(event.callId) ?? '';
  const discriminator =
    str(event.modelCallId) ??
    str(event.id) ??
    (num(event.stepNumber) !== undefined ? String(event.stepNumber) : undefined);
  return `${callId}#${discriminator ?? ''}`;
}

function pickThreadId(source: Record<string, unknown> | undefined): string | undefined {
  if (!source) return undefined;
  for (const key of THREAD_ID_KEYS) {
    const value = source[key];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

/** Render an AI SDK prompt (message array, or a bare string) as sections. */
function promptToSections(prompt: unknown): TraceSection[] {
  if (prompt === undefined || prompt === null) return [];
  const parsed = typeof prompt === 'string' ? parseMaybeJson(prompt) : prompt;

  if (typeof parsed === 'string') {
    return [{ kind: 'message', label: 'User message', role: 'user', content: parsed }];
  }
  if (!Array.isArray(parsed)) {
    const record = asRecord(parsed);
    // `ai.prompt` is sometimes `{ system, prompt|messages }`.
    if (record && (record.messages || record.prompt)) {
      return [
        ...(record.system !== undefined
          ? ([{ kind: 'message', label: 'Instructions', role: 'system', content: record.system }] as TraceSection[])
          : []),
        ...promptToSections(record.messages ?? record.prompt),
      ];
    }
    return record ? [{ kind: 'metadata', label: 'Prompt', content: record }] : [];
  }

  const sections: TraceSection[] = [];
  for (const message of parsed) {
    const m = asRecord(message);
    if (!m) continue;
    const role = str(m.role) ?? 'user';
    sections.push({
      kind: 'message',
      label: `${role.charAt(0).toUpperCase()}${role.slice(1)} message`,
      role,
      content: m.content ?? m,
    });
  }
  return sections;
}

/** Render a model's output — a content-part array, or plain text. */
function contentToSections(content: unknown): TraceSection[] {
  if (content === undefined || content === null) return [];
  if (typeof content === 'string') {
    return content
      ? [{ kind: 'message', label: 'Assistant message', role: 'assistant', content }]
      : [];
  }
  if (!Array.isArray(content)) {
    return [{ kind: 'message', label: 'Assistant message', role: 'assistant', content }];
  }

  const sections: TraceSection[] = [];
  for (const part of content) {
    const p = asRecord(part);
    if (!p) continue;
    if (p.type === 'tool-call' || p.type === 'tool_call') {
      const toolName = str(p.toolName) ?? str(p.name) ?? 'tool';
      sections.push({
        kind: 'tool_call',
        label: `Tool call: ${toolName}`,
        tool: toolName,
        content: parseMaybeJson(p.input ?? p.args),
      });
      continue;
    }
    sections.push({
      kind: 'message',
      label: p.type === 'reasoning' ? 'Reasoning' : 'Assistant message',
      role: 'assistant',
      content: p.text ?? p.content ?? p,
    });
  }
  return sections;
}

/**
 * Observe a stream without altering what the consumer sees.
 *
 * A `TransformStream` would be shorter but only reports normal completion; this
 * also fires `onDone` when the consumer cancels or the producer errors, which is
 * what stops an abandoned stream from leaving its event open forever.
 */
function instrumentStream(
  source: unknown,
  onPart: (part: unknown) => void,
  onDone: (error?: unknown) => void,
): unknown {
  const stream = source as ReadableStream<unknown> | undefined;
  if (!stream || typeof stream.getReader !== 'function') return undefined;

  const reader = stream.getReader();
  let settled = false;
  const settle = (error?: unknown) => {
    if (settled) return;
    settled = true;
    try {
      onDone(error);
    } catch {
      // Never let bookkeeping break the stream.
    }
  };

  return new ReadableStream({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          settle();
          controller.close();
          return;
        }
        try {
          onPart(value);
        } catch {
          // Ignore — a malformed part must not stall the stream.
        }
        controller.enqueue(value);
      } catch (error) {
        settle(error);
        controller.error(error);
      }
    },
    cancel(reason) {
      settle(reason instanceof Error ? reason : undefined);
      return reader.cancel(reason);
    },
  });
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** OTel attributes are primitives, so structured payloads arrive as JSON. */
function parseMaybeJson(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const trimmed = value.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return value;
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  const record = asRecord(error);
  return str(record?.message) ?? 'Span error';
}

function dropUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as T;
}
