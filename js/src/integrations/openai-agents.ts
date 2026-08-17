/**
 * OpenAI Agents SDK tracing processor (JS/TS).
 *
 * ```ts
 * import { init } from '@cognipeer/observability';
 * import { installOpenAIAgentsTracing } from '@cognipeer/observability/openai-agents';
 *
 * init({ apiKey: 'cpeer_…' });
 * await installOpenAIAgentsTracing();          // replaces OpenAI's own exporter
 *
 * await run(agent, 'book me a flight', { groupId: 'conv-42' });
 * ```
 *
 * `groupId` is the SDK's only conversation identifier and nothing sets it for
 * you — not even an SDK `Session`. Pass it if you want runs to group into one
 * thread in the Console.
 *
 * **What the SDK can and cannot tell us.** Prompts, completions, tool
 * arguments, tool results, per-call token usage (with the cached-token
 * breakdown) and the agent/handoff/guardrail structure all come through. Full
 * tool JSON schemas only exist on the Responses API path
 * (`ResponseSpanData._response.tools`); on the Chat Completions path the SDK's
 * traceable-settings allowlist excludes `tools`, so only tool *names* are
 * recoverable — those are still emitted, without schemas.
 *
 * **Redaction is our job, not the SDK's.** `ModelTracing` is
 * `boolean | 'enabled_without_data'`, and the non-streaming Responses path
 * guards payload capture with a plain truthiness check — so the truthy string
 * `'enabled_without_data'` still captures prompts even when the integrator
 * asked for no sensitive data. Set `capture: 'metadata'` on the Cognipeer
 * client if content must not leave the process; that gate is enforced here.
 */

import { getClient, type CognipeerObservability } from '../core/client';
import { spanIdFrom, traceIdFrom } from '../core/ids';
import type { TraceSession } from '../core/session';
import type { TraceEvent, TraceResponseFormat, TraceSection, TraceToolDefinition } from '../core/types';

/* ------------------------------------------------------------------ */
/*  Minimal SDK type surface                                          */
/*                                                                    */
/*  Structural copies of `@openai/agents` shapes rather than imports:  */
/*  the SDK is an optional peer dependency, so this entry point must   */
/*  type-check and build without it installed.                         */
/* ------------------------------------------------------------------ */

/** Subset of `@openai/agents#SpanError`. */
export interface AgentsSpanError {
  message: string;
  data?: Record<string, unknown> | null;
}

/**
 * Subset of `@openai/agents#Span`. Note these are camelCase getters in JS —
 * Python's equivalents are snake_case attributes.
 */
export interface AgentsSpan {
  traceId: string;
  spanId: string;
  parentId: string | null;
  spanData: Record<string, unknown> & { type?: string };
  error: AgentsSpanError | null;
  startedAt: string | null;
  endedAt: string | null;
  traceMetadata?: Record<string, unknown>;
}

/** Subset of `@openai/agents#Trace`. Public fields in JS, unlike Python. */
export interface AgentsTrace {
  traceId: string;
  name: string;
  groupId?: string | null;
  metadata?: Record<string, unknown>;
}

/** Subset of `@openai/agents#TracingProcessor` — camelCase, all async. */
export interface AgentsTracingProcessor {
  start?(): void;
  onTraceStart(trace: AgentsTrace): Promise<void>;
  onTraceEnd(trace: AgentsTrace): Promise<void>;
  onSpanStart(span: AgentsSpan): Promise<void>;
  onSpanEnd(span: AgentsSpan): Promise<void>;
  shutdown(timeout?: number): Promise<void>;
  forceFlush(): Promise<void>;
}

/* ------------------------------------------------------------------ */
/*  Mapping tables                                                    */
/* ------------------------------------------------------------------ */

/** `spanData.type` → Console event type. Types absent here fall back to `span`. */
export const EVENT_TYPE_BY_SPAN: Record<string, TraceEvent['type']> = {
  generation: 'ai_call',
  response: 'ai_call',
  function: 'tool_call',
  guardrail: 'guardrail',
  transcription: 'span',
  speech: 'span',
  speech_group: 'span',
  mcp_tools: 'span',
  agent: 'span',
  handoff: 'span',
  custom: 'span',
  task: 'span',
  turn: 'span',
};

/**
 * Only leaf model calls carry tokens we may count. `task` and `turn` spans
 * (added in 0.14.0) repeat the same usage at run and turn granularity —
 * summing all three inflates a session's token totals threefold.
 */
export const TOKEN_BEARING_SPANS = new Set(['generation', 'response']);

/* ------------------------------------------------------------------ */
/*  Processor                                                         */
/* ------------------------------------------------------------------ */

export interface CognipeerTracingProcessorOptions {
  /** Client to send through. Defaults to the process-wide one. */
  client?: CognipeerObservability;
  /** Session agent name. Defaults to the trace's workflow name. */
  agentName?: string;
  /** Conversation id. Defaults to `RunConfig.groupId`. */
  threadId?: string;
  /**
   * Emit the agent's tool NAMES as a tool-definitions section on agent spans,
   * even when schemas are unavailable (default true).
   */
  captureAgentTools?: boolean;
}

/**
 * Bridges the Agents SDK tracing bus onto Cognipeer sessions.
 *
 * One SDK trace becomes one Console session; every span becomes one event,
 * linked by `parentId` so the Console renders the agent/handoff tree.
 */
export class CognipeerTracingProcessor implements AgentsTracingProcessor {
  private readonly client: CognipeerObservability;
  private readonly agentName?: string;
  private readonly threadId?: string;
  private readonly captureAgentTools: boolean;

  private readonly sessions = new Map<string, TraceSession>();
  /**
   * First model seen per trace — the SDK never puts a model on the agent
   * span, so the session's model has to be hoisted from a child.
   */
  private readonly models = new Map<string, string>();
  /** Deliveries in flight, so `shutdown`/`forceFlush` can await them. */
  private readonly closing = new Set<Promise<void>>();

  constructor(options: CognipeerTracingProcessorOptions = {}) {
    this.client = options.client ?? getClient();
    this.agentName = options.agentName;
    this.threadId = options.threadId;
    this.captureAgentTools = options.captureAgentTools ?? true;
  }

  // ── TracingProcessor ────────────────────────────────────────────────

  async onTraceStart(trace: AgentsTrace): Promise<void> {
    // Deliberately lazy elsewhere too: a resumed run (`ReattachedTrace`)
    // never fires this, so creating the session only here would drop the
    // whole run.
    this.sessionFor(trace);
  }

  async onTraceEnd(trace: AgentsTrace): Promise<void> {
    const session = this.sessions.get(trace.traceId);
    if (!session) return;
    this.sessions.delete(trace.traceId);
    this.models.delete(trace.traceId);
    this.track(session.end());
  }

  /**
   * No-op.
   *
   * The SDK's span data is only fully populated once the span finishes, so
   * everything is emitted from {@link onSpanEnd} — which also keeps this
   * callback, awaited by `MultiTracingProcessor` on the agent's hot path,
   * free of work.
   */
  async onSpanStart(_span: AgentsSpan): Promise<void> {}

  async onSpanEnd(span: AgentsSpan): Promise<void> {
    try {
      this.emit(span);
    } catch (error) {
      // The SDK swallows processor exceptions, so a silent failure here
      // would look like "tracing just doesn't work". Log it loudly.
      this.client.config.logger.error(
        'failed to export an Agents SDK span:',
        error instanceof Error ? error.message : error,
      );
    }
  }

  async shutdown(_timeout?: number): Promise<void> {
    for (const [traceId, session] of this.sessions) {
      this.sessions.delete(traceId);
      this.track(session.end());
    }
    this.models.clear();
    await this.settle();
  }

  async forceFlush(): Promise<void> {
    await this.settle();
    await this.client.flush();
  }

  // ── Mapping ─────────────────────────────────────────────────────────

  private sessionFor(trace: AgentsTrace): TraceSession | undefined {
    const traceId = trace.traceId;
    if (!traceId) return undefined;

    const existing = this.sessions.get(traceId);
    if (existing) return existing;

    const session = this.client.startSession({
      threadId: this.threadId ?? trace.groupId ?? undefined,
      agent: { name: this.agentName ?? trace.name ?? 'agent' },
      // `trace_<32hex>` → the bare 32 hex, which is already W3C-shaped.
      traceId: traceIdFrom(stripPrefix(traceId)),
      config: trace.metadata && Object.keys(trace.metadata).length > 0 ? trace.metadata : undefined,
    });
    this.sessions.set(traceId, session);
    return session;
  }

  private emit(span: AgentsSpan): void {
    const data = span.spanData;
    if (!data) return;

    // `span.toJSON().type` masquerades as 'custom' for task/turn spans; the
    // spanData attribute is the honest discriminator.
    const spanType = typeof data.type === 'string' ? data.type : 'custom';
    const session =
      this.sessions.get(span.traceId) ??
      this.sessionFor({ traceId: span.traceId, name: this.agentName ?? 'agent' });
    if (!session) return;

    const model = modelOf(data);
    if (model && !this.models.has(span.traceId)) {
      this.models.set(span.traceId, model);
    }

    const usage = TOKEN_BEARING_SPANS.has(spanType) ? usageOf(data) : {};
    const event: TraceEvent = {
      id: span.spanId,
      type: EVENT_TYPE_BY_SPAN[spanType] ?? 'span',
      label: labelOf(data, spanType),
      spanId: spanIdFrom(stripPrefix(span.spanId)),
      parentSpanId: span.parentId
        ? spanIdFrom(stripPrefix(span.parentId))
        : session.rootSpanId,
      traceId: session.traceId,
      status: span.error ? 'error' : 'success',
      sections: this.sectionsOf(data, spanType),
      metadata: { spanType },
    };

    if (model) event.model = model;
    if (span.startedAt) event.timestamp = span.startedAt;
    const durationMs = durationOf(span.startedAt, span.endedAt);
    if (durationMs !== undefined) event.durationMs = durationMs;
    if (span.error) {
      event.error = {
        message: span.error.message || 'Span error',
        ...(span.error.data ? { data: span.error.data } : {}),
      };
    }

    if (spanType === 'function') {
      const name = asString(data.name) ?? 'tool';
      event.toolName = name;
      event.actor = { scope: 'tool', name };
    } else if (TOKEN_BEARING_SPANS.has(spanType)) {
      event.actor = { scope: 'model', name: model ?? 'model' };
    } else if (spanType === 'agent') {
      event.actor = { scope: 'agent', name: asString(data.name) ?? 'agent' };
    }

    if (usage.inputTokens !== undefined) event.inputTokens = usage.inputTokens;
    if (usage.outputTokens !== undefined) event.outputTokens = usage.outputTokens;
    if (usage.totalTokens !== undefined) event.totalTokens = usage.totalTokens;
    if (usage.cachedInputTokens !== undefined) event.cachedInputTokens = usage.cachedInputTokens;

    const tools = this.toolDefinitionsOf(data, spanType);
    if (tools) event.toolDefinitions = tools;

    const responseFormat = this.responseFormatOf(data, spanType);
    if (responseFormat) event.responseFormat = responseFormat;

    session.record(event);
  }

  private sectionsOf(
    data: Record<string, unknown>,
    spanType: string,
  ): TraceSection[] {
    const sections: TraceSection[] = [];

    if (spanType === 'function') {
      const tool = asString(data.name) ?? 'tool';
      sections.push({
        kind: 'tool_call',
        label: 'Arguments',
        tool,
        // FunctionSpanData.input is the raw JSON argument STRING.
        content: maybeJson(data.input),
      });
      if (data.output !== undefined && data.output !== null) {
        sections.push({ kind: 'tool_result', label: 'Result', tool, content: data.output });
      }
      const mcpData = data.mcp_data ?? data.mcpData;
      if (mcpData) sections.push({ kind: 'metadata', label: 'MCP', content: maybeJson(mcpData) });
      return sections;
    }

    if (spanType === 'generation') {
      sections.push(...messagesToSections(data.input, 'user'));
      sections.push(...messagesToSections(data.output, 'assistant'));
      const config = data.model_config ?? data.modelConfig;
      if (config) {
        sections.push({ kind: 'metadata', label: 'Model settings', content: config });
      }
      return sections;
    }

    if (spanType === 'response') {
      // JS names these `_input` / `_response`; `toJSON()` deliberately drops
      // the payload, so the underscore attributes are the only source.
      sections.push(...messagesToSections(data._input ?? data.input, 'user'));
      const response = asRecord(data._response ?? data.response);
      if (response) {
        const instructions = response.instructions;
        if (instructions) {
          sections.unshift({
            kind: 'message',
            label: 'Instructions',
            role: 'system',
            content: instructions,
          });
        }
        if (response.output !== undefined && response.output !== null) {
          sections.push({
            kind: 'message',
            label: 'Assistant message',
            role: 'assistant',
            content: response.output,
          });
        }
      }
      return sections;
    }

    if (spanType === 'agent') {
      sections.push({
        kind: 'metadata',
        label: 'Agent',
        content: {
          handoffs: data.handoffs,
          tools: data.tools,
          output_type: data.output_type ?? data.outputType,
        },
      });
      return sections;
    }

    if (spanType === 'handoff') {
      sections.push({
        kind: 'metadata',
        label: 'Handoff',
        content: {
          from: data.from_agent ?? data.fromAgent,
          to: data.to_agent ?? data.toAgent,
        },
      });
      return sections;
    }

    if (spanType === 'guardrail') {
      sections.push({
        kind: 'metadata',
        label: 'Guardrail',
        content: { name: data.name, triggered: data.triggered },
      });
      return sections;
    }

    if (spanType === 'mcp_tools') {
      sections.push({
        kind: 'metadata',
        label: `MCP server: ${asString(data.server) ?? '?'}`,
        content: data.result,
      });
      return sections;
    }

    if (data.data !== undefined && data.data !== null) {
      sections.push({ kind: 'metadata', label: 'Data', content: data.data });
    }
    return sections;
  }

  /**
   * The tool menu offered to the model on this call.
   *
   * Full schemas exist only on the Responses path, where the API echoes the
   * tool array back. Elsewhere the agent span's `tools` list of names is the
   * best available and is emitted schema-less rather than not at all.
   */
  private toolDefinitionsOf(
    data: Record<string, unknown>,
    spanType: string,
  ): TraceToolDefinition[] | undefined {
    if (spanType === 'response') {
      const response = asRecord(data._response ?? data.response);
      return normalizeTools(response?.tools);
    }
    if (spanType === 'agent' && this.captureAgentTools) {
      const names = data.tools;
      if (Array.isArray(names) && names.length > 0) {
        return names.map((name) => ({ name: String(name) }));
      }
    }
    return undefined;
  }

  /**
   * The structured-output contract enforced on this call.
   *
   * The Responses path echoes it back as `text.format` (`{type:'json_schema',
   * name, strict, schema}`); the Chat Completions path uses `response_format`.
   * An agent span carries only the SDK-side `output_type` name, which is
   * emitted schema-less rather than not at all — knowing a contract existed is
   * what separates "the model chose prose" from "nothing asked for JSON".
   */
  private responseFormatOf(
    data: Record<string, unknown>,
    spanType: string,
  ): TraceResponseFormat | undefined {
    if (spanType === 'response') {
      const response = asRecord(data._response ?? data.response);
      const format =
        asRecord(asRecord(response?.text)?.format)
        ?? asRecord(response?.response_format)
        ?? asRecord(response?.responseFormat);
      return normalizeResponseFormat(format);
    }
    if (spanType === 'generation') {
      const config = asRecord(data.model_config) ?? asRecord(data.modelConfig);
      return normalizeResponseFormat(
        asRecord(config?.response_format) ?? asRecord(config?.responseFormat),
      );
    }
    if (spanType === 'agent') {
      const outputType = asString(data.output_type ?? data.outputType);
      // 'str' is the SDK's default — an unconstrained agent, not a contract.
      if (outputType && outputType !== 'str') return { type: 'json_schema', name: outputType };
    }
    return undefined;
  }

  // ── Delivery bookkeeping ────────────────────────────────────────────

  /**
   * Session delivery is started but never awaited inside a tracing callback:
   * `MultiTracingProcessor` awaits those, so awaiting an HTTP round trip in
   * `onTraceEnd` would add it to the latency of every `run()`.
   */
  private track(promise: Promise<void>): void {
    this.closing.add(promise);
    void promise.finally(() => this.closing.delete(promise));
  }

  private async settle(): Promise<void> {
    while (this.closing.size > 0) {
      await Promise.all([...this.closing]);
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Registration                                                      */
/* ------------------------------------------------------------------ */

/** Kept out of the `import()` call site so it stays an unresolved specifier. */
const OPENAI_AGENTS_MODULE = '@openai/agents';

export interface InstallOpenAIAgentsTracingOptions extends CognipeerTracingProcessorOptions {
  /**
   * Keep the SDK's built-in OpenAI exporter alongside this one (default
   * false). Leaving it off means traces stop going to OpenAI's dashboard,
   * the SDK no longer needs `OPENAI_API_KEY` for tracing, and no prompt data
   * leaves for a second destination.
   */
  keepOpenAIExporter?: boolean;
}

/**
 * Route Agents SDK traces to Cognipeer.
 *
 * By default this REPLACES the SDK's processor list. Pass
 * `keepOpenAIExporter: true` to export to both.
 */
export async function installOpenAIAgentsTracing(
  options: InstallOpenAIAgentsTracingOptions = {},
): Promise<CognipeerTracingProcessor> {
  const { keepOpenAIExporter = false, ...processorOptions } = options;

  let sdk: {
    addTraceProcessor?: (processor: AgentsTracingProcessor) => void;
    setTraceProcessors?: (processors: AgentsTracingProcessor[]) => void;
  };
  try {
    // The specifier goes through a variable on purpose: TypeScript only
    // resolves `import()` when its argument is a string literal, and this
    // package must type-check and build for users who have not installed
    // this optional peer.
    sdk = (await import(/* @vite-ignore */ OPENAI_AGENTS_MODULE)) as typeof sdk;
  } catch (error) {
    throw new Error(
      '@cognipeer/observability/openai-agents requires the OpenAI Agents SDK. ' +
        'Install it with `npm install @openai/agents`.',
      { cause: error },
    );
  }

  const processor = new CognipeerTracingProcessor(processorOptions);
  if (keepOpenAIExporter) {
    sdk.addTraceProcessor?.(processor);
  } else {
    sdk.setTraceProcessors?.([processor]);
  }
  return processor;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

/** Strip the SDK's `trace_` / `span_` prefix so the rest is plain hex. */
function stripPrefix(id: string): string {
  return id.replace(/^(?:trace|span)_/, '');
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function labelOf(data: Record<string, unknown>, spanType: string): string {
  const name = asString(data.name);
  if (name) return name;
  if (spanType === 'handoff') {
    const from = asString(data.from_agent ?? data.fromAgent) ?? '?';
    const to = asString(data.to_agent ?? data.toAgent) ?? '?';
    return `${from} → ${to}`;
  }
  if (spanType === 'mcp_tools') return `mcp: ${asString(data.server) ?? '?'}`;
  return modelOf(data) ?? spanType;
}

function modelOf(data: Record<string, unknown>): string | undefined {
  const direct = asString(data.model);
  if (direct) return direct;
  const response = asRecord(data._response ?? data.response);
  return asString(response?.model);
}

interface NormalizedUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cachedInputTokens?: number;
}

/**
 * Normalize the SDK's several usage shapes into one.
 *
 * `GenerationSpanData.usage` is deliberately loose and may still use the older
 * `prompt_tokens`/`completion_tokens` names. JS `ResponseSpanData` has NO
 * `usage` field at all — the numbers live on the raw `_response`.
 */
function usageOf(data: Record<string, unknown>): NormalizedUsage {
  let usage = asRecord(data.usage);
  if (!usage) {
    usage = asRecord(asRecord(data._response ?? data.response)?.usage);
  }
  if (!usage) return {};

  const details =
    asRecord(usage.input_tokens_details) ??
    asRecord(usage.prompt_tokens_details) ??
    asRecord(usage.details);

  const out: NormalizedUsage = {};
  const input = asNumber(usage.input_tokens ?? usage.prompt_tokens);
  const output = asNumber(usage.output_tokens ?? usage.completion_tokens);
  const total = asNumber(usage.total_tokens);
  const cached = asNumber(details?.cached_tokens ?? usage.cached_input_tokens);

  if (input !== undefined) out.inputTokens = input;
  if (output !== undefined) out.outputTokens = output;
  if (total !== undefined) out.totalTokens = total;
  if (cached !== undefined) out.cachedInputTokens = cached;
  return out;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** Render an OpenAI-shaped message list as message sections. */
function messagesToSections(payload: unknown, defaultRole: string): TraceSection[] {
  if (payload === undefined || payload === null) return [];
  if (typeof payload === 'string') {
    return [
      {
        kind: 'message',
        label: `${capitalize(defaultRole)} message`,
        role: defaultRole,
        content: payload,
      },
    ];
  }

  const items = Array.isArray(payload) ? payload : [payload];
  const sections: TraceSection[] = [];
  for (const item of items) {
    const record = asRecord(item);
    if (!record) {
      sections.push({ kind: 'metadata', label: 'Item', content: item });
      continue;
    }
    const role = asString(record.role) ?? defaultRole;
    sections.push({
      kind: 'message',
      label: `${capitalize(role)} message`,
      role,
      content: record.content ?? record,
    });
    const toolCalls = record.tool_calls ?? record.toolCalls;
    if (Array.isArray(toolCalls)) {
      for (const call of toolCalls) {
        const fn = asRecord(asRecord(call)?.function);
        const name = asString(fn?.name) ?? 'tool';
        sections.push({
          kind: 'tool_call',
          label: `Tool call: ${name}`,
          tool: name,
          content: maybeJson(fn?.arguments),
        });
      }
    }
  }
  return sections;
}

/**
 * Coerce an OpenAI structured-output block into our flat shape. Accepts both
 * the Responses `text.format` object (`{type, name, strict, schema}`) and the
 * Chat Completions `response_format` envelope (`{type, json_schema:{…}}`).
 */
function normalizeResponseFormat(value: unknown): TraceResponseFormat | undefined {
  const raw = asRecord(value);
  if (!raw) return undefined;
  const type = asString(raw.type);
  const jsonSchema = asRecord(raw.json_schema) ?? asRecord(raw.jsonSchema);
  const source = jsonSchema ?? raw;
  const schema = asRecord(source.schema);
  if (!type && !schema) return undefined;
  // The Responses API reports plain text mode as `{type:'text'}` — a contract
  // in name only, and recording it would put a section on every single call.
  if (type === 'text') return undefined;

  return {
    type: type ?? 'json_schema',
    ...(asString(source.name) ? { name: asString(source.name) as string } : {}),
    ...(typeof source.strict === 'boolean' ? { strict: source.strict } : {}),
    ...(schema ? { schema } : {}),
  };
}

function normalizeTools(tools: unknown): TraceToolDefinition[] | undefined {
  if (!Array.isArray(tools)) return undefined;
  const out: TraceToolDefinition[] = [];
  for (const entry of tools) {
    const record = asRecord(entry);
    if (!record) continue;
    const fn = asRecord(record.function) ?? record;
    const name = asString(fn.name);
    if (!name) {
      // Hosted tools (web_search, file_search, …) have no name, only a type.
      const type = asString(record.type);
      if (type) out.push({ name: type });
      continue;
    }
    const definition: TraceToolDefinition = { name };
    const description = asString(fn.description);
    if (description) definition.description = description;
    const parameters = asRecord(fn.parameters ?? fn.input_schema ?? fn.inputSchema);
    if (parameters) definition.parameters = parameters;
    out.push(definition);
  }
  return out.length > 0 ? out : undefined;
}

/** Tool arguments arrive as a JSON string — parse so the UI can tree them. */
function maybeJson(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function durationOf(startedAt: string | null, endedAt: string | null): number | undefined {
  if (!startedAt || !endedAt) return undefined;
  const start = Date.parse(startedAt);
  const end = Date.parse(endedAt);
  if (Number.isNaN(start) || Number.isNaN(end)) return undefined;
  return Math.max(0, end - start);
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
