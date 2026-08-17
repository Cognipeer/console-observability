/**
 * Claude Agent SDK tracing (JS/TS).
 *
 * The SDK's message stream is the richest seam it has: it carries the model,
 * the full assistant content (text, thinking and tool_use blocks), per-call
 * token usage with the cache breakdown, tool results, subagent nesting via
 * `parent_tool_use_id` and the final cost/duration summary. So the integration
 * consumes that stream rather than installing hooks:
 *
 * ```ts
 * import { init } from '@cognipeer/observability';
 * import { traceQuery } from '@cognipeer/observability/claude-agent-sdk';
 *
 * init({ apiKey: 'cpeer_…' });
 *
 * for await (const message of traceQuery({
 *   prompt: 'refactor the billing module',
 *   options: { model: 'claude-opus-5' },
 *   threadId: 'conv-42',
 * })) {
 *   console.log(message);
 * }
 * ```
 *
 * `traceQuery` yields the SDK's messages untouched, so it is a drop-in
 * replacement for `query`. If you drive the stream yourself, use
 * {@link ClaudeMessageTracer} and feed it each message.
 *
 * **Why not a fetch wrapper.** The SDK does not call the Anthropic API in your
 * process — it spawns the Claude Code CLI and talks JSON over stdio. HTTP
 * monkeypatching, OpenTelemetry auto-instrumentation of `http`, and
 * `@anthropic-ai/sdk` middleware all capture nothing here.
 *
 * **Not available through this seam:** tool JSON schemas. The SDK reports tool
 * *names* on the `system/init` message but never their input schemas — only
 * Claude Code's OTel `api_request_body` event carries those, and it is off by
 * default. Tool names are recorded; schemas are simply absent.
 */

import { getClient, type CognipeerObservability } from '../core/client';
import { spanIdFrom } from '../core/ids';
import { stringifyContent } from '../core/redact';
import type { TraceSession } from '../core/session';
import type { TraceAgent, TraceEvent, TraceSection } from '../core/types';

/* ------------------------------------------------------------------ */
/*  Minimal SDK type surface                                          */
/*                                                                    */
/*  Structural copies of `@anthropic-ai/claude-agent-sdk` shapes       */
/*  rather than imports: the SDK is an optional peer dependency, so    */
/*  this entry point must type-check and build without it installed.   */
/* ------------------------------------------------------------------ */

/** Any message the SDK's `query()` async iterator yields. */
export interface ClaudeSDKMessage {
  type?: string;
  subtype?: string;
  uuid?: string;
  session_id?: string;
  parent_tool_use_id?: string | null;
  /** `assistant` / `user`: the raw Anthropic Messages API object. */
  message?: Record<string, unknown>;
  /** `assistant`: retracts previously delivered messages by uuid. */
  supersedes?: string[];
  /** `assistant`: an interrupt truncated this message. */
  aborted?: true;
  /** `user`: structured (non-stringified) tool output. */
  tool_use_result?: unknown;
  [key: string]: unknown;
}

/* ------------------------------------------------------------------ */
/*  Tracer                                                            */
/* ------------------------------------------------------------------ */

export interface ClaudeMessageTracerOptions {
  /** Client to send through. Defaults to the process-wide one. */
  client?: CognipeerObservability;
  /** Agent identity. Defaults to `{ name: 'claude-agent' }`. */
  agent?: TraceAgent;
  /** Conversation id, so several turns group into one thread. */
  threadId?: string;
  /** Fixed session id; otherwise one is generated per stream. */
  sessionId?: string;
}

interface PendingTool {
  name: string;
  input: unknown;
  parent?: string;
}

/**
 * Turns a Claude Agent SDK message stream into a Cognipeer session.
 *
 * Feed it every message the SDK yields — in order — and call {@link close}
 * when the stream ends. A `result` message closes the session on its own, so
 * `close` is only needed when the stream is abandoned early.
 */
export class ClaudeMessageTracer {
  private readonly client: CognipeerObservability;
  private readonly threadId?: string;
  private readonly sessionId?: string;

  private agent: TraceAgent;
  private session?: TraceSession;
  /**
   * Conversation id shared by every turn of this stream, latched from the
   * first message that names one. See `ensureSession`.
   */
  private conversationId?: string;
  /** tool_use blocks awaiting their tool_result, keyed by tool_use id. */
  private readonly pendingTools = new Map<string, PendingTool>();
  /**
   * Assistant `message.id`s whose usage has already been counted. One API
   * turn can produce several assistant messages sharing an id; adding up each
   * one's `usage` would multiply that call's tokens.
   */
  private readonly countedMessageIds = new Set<string>();
  private toolNames: string[] = [];
  private closing?: Promise<void>;

  constructor(options: ClaudeMessageTracerOptions = {}) {
    this.client = options.client ?? getClient();
    this.agent = options.agent ?? { name: 'claude-agent' };
    this.threadId = options.threadId;
    this.sessionId = options.sessionId;
  }

  /** The session being written into, once one exists. */
  getSession(): TraceSession | undefined {
    return this.session;
  }

  /** Record one SDK message. Never throws. */
  handle(message: ClaudeSDKMessage): void {
    try {
      this.dispatch(message);
    } catch (error) {
      this.client.config.logger.warn(
        'failed to record a Claude SDK message:',
        error instanceof Error ? error.message : error,
      );
    }
  }

  /**
   * End the session if the stream stopped without a `result`.
   *
   * A crash, SIGKILL or transport gap leaves a run with no terminal message,
   * so this is what keeps such a session from hanging as `in_progress`.
   */
  async close(options: { status?: TraceEvent['status']; error?: unknown } = {}): Promise<void> {
    this.flushPendingTools();
    const session = this.session;
    if (!session) {
      await this.closing;
      return;
    }
    this.session = undefined;
    this.closing = session.end({
      status: options.status ?? 'success',
      error: options.error,
    });
    await this.closing;
  }

  // ── Dispatch ────────────────────────────────────────────────────────

  private dispatch(message: ClaudeSDKMessage): void {
    switch (message.type) {
      case 'system':
        this.handleSystem(message);
        break;
      case 'assistant':
        this.handleAssistant(message);
        break;
      case 'user':
        this.handleUser(message);
        break;
      case 'result':
        this.handleResult(message);
        break;
      default:
        // `stream_event` partials are deliberately ignored: `message_start`
        // carries usage with only cache reads and `message_delta` the final
        // output count, so accounting off partials produces output_tokens≈1.
        // The terminal assistant message already has complete usage.
        break;
    }
  }

  private ensureSession(message: ClaudeSDKMessage): TraceSession {
    if (!this.session) {
      // Latch the conversation id from the first message that carries one. In
      // streaming-input mode the SDK emits one `result` per TURN, so each turn
      // opens its own session — they must all land in the same thread, or a
      // conversation reads as a pile of unrelated runs.
      this.conversationId ??= this.threadId ?? asString(message.session_id);

      this.session = this.client.startSession({
        sessionId: this.sessionId,
        threadId: this.conversationId,
        agent: this.agent,
        // A caller-fixed session id means every turn re-uses one wire id, and
        // only the streaming shape appends: the batch endpoint DELETES a
        // session's existing events before writing the payload's, so a
        // multi-turn stream in batch (or auto, which may pick either) would
        // keep nothing but its last turn. Streaming legs are merged
        // server-side, running totals and all.
        ...(this.sessionId ? { mode: 'stream' as const } : {}),
      });
    }
    return this.session;
  }

  private handleSystem(message: ClaudeSDKMessage): void {
    const model = asString(message.model);
    const tools = message.tools;
    if (Array.isArray(tools)) {
      this.toolNames = tools.map((name) => String(name));
    }
    if (model) this.agent = { ...this.agent, model };

    const session = this.ensureSession(message);
    session.record({
      type: 'span',
      label: 'Session init',
      actor: { scope: 'agent', name: asString(this.agent.name) ?? 'claude-agent' },
      sections: [
        {
          kind: 'metadata',
          label: 'Init',
          content: {
            model,
            tools: this.toolNames,
            mcp_servers: message.mcp_servers,
            permission_mode: message.permissionMode,
            cwd: message.cwd,
          },
        },
      ],
    });
  }

  private handleAssistant(message: ClaudeSDKMessage): void {
    const session = this.ensureSession(message);
    const inner = message.message ?? {};
    const model = asString(message.model) ?? asString(inner.model);
    const messageId = asString(inner.id);
    const parent = asString(message.parent_tool_use_id ?? undefined);

    const sections: TraceSection[] = [];
    for (const block of contentBlocks(inner)) {
      const record = asRecord(block);
      if (!record) continue;

      switch (record.type) {
        case 'text':
          sections.push({
            kind: 'message',
            label: 'Assistant message',
            role: 'assistant',
            content: record.text,
          });
          break;
        case 'thinking':
          sections.push({
            kind: 'message',
            label: 'Thinking',
            role: 'assistant',
            content: record.thinking ?? record.text,
          });
          break;
        case 'tool_use': {
          const toolId = asString(record.id) ?? '';
          const name = asString(record.name) ?? 'tool';
          sections.push({
            kind: 'tool_call',
            label: `Tool call: ${name}`,
            tool: name,
            content: record.input,
          });
          // Held until the matching tool_result arrives so one call and its
          // result become a single event in the Console timeline.
          if (toolId) {
            this.pendingTools.set(toolId, { name, input: record.input, parent });
          }
          break;
        }
        default:
          break;
      }
    }

    const event: TraceEvent = {
      id: asString(message.uuid) ?? messageId,
      type: 'ai_call',
      label: model ?? 'assistant',
      actor: { scope: 'model', name: model ?? 'model' },
      sections,
      metadata: assistantMetadata(message, inner),
      // `timestamp` is deliberately left unset: the SDK's own field uses the
      // originating host's clock and is display-only, so the session's arrival
      // ordering is the honest sequence.
      status: message.aborted ? 'error' : 'success',
    };
    if (model) event.model = model;
    if (parent) event.parentSpanId = spanIdFrom(parent);
    if (message.aborted) {
      event.error = { message: 'Interrupted before the model finished responding' };
    }

    // Usage is attached only the first time a `message.id` is seen. Several
    // assistant messages can share one id (the .d.ts warns about it), and each
    // repeats the same `usage` — summing them multiplies that call's tokens.
    if (!messageId || !this.countedMessageIds.has(messageId)) {
      if (messageId) this.countedMessageIds.add(messageId);
      Object.assign(event, usageOf(asRecord(message.usage) ?? asRecord(inner.usage)));
    }

    session.record(event);
  }

  /** A user message carries tool results — pair them with their calls. */
  private handleUser(message: ClaudeSDKMessage): void {
    const session = this.ensureSession(message);
    const inner = message.message ?? {};
    const structured = message.tool_use_result;

    for (const block of contentBlocks(inner)) {
      const record = asRecord(block);
      if (!record || record.type !== 'tool_result') continue;

      const toolId = asString(record.tool_use_id) ?? '';
      const pending = toolId ? this.pendingTools.get(toolId) : undefined;
      if (toolId) this.pendingTools.delete(toolId);
      const isError = record.is_error === true;
      const name = pending?.name ?? 'tool';

      const sections: TraceSection[] = [];
      if (pending) {
        sections.push({
          kind: 'tool_call',
          label: 'Arguments',
          tool: name,
          content: pending.input,
        });
      }
      sections.push({
        kind: 'tool_result',
        label: 'Result',
        tool: name,
        // `tool_use_result` is the structured value; the block's own content
        // is the stringified version the model saw.
        content: structured !== undefined ? structured : record.content,
      });

      const event: TraceEvent = {
        type: 'tool_call',
        label: name,
        toolName: name,
        actor: { scope: 'tool', name },
        sections,
        status: isError ? 'error' : 'success',
      };
      if (toolId) {
        event.id = toolId;
        event.spanId = spanIdFrom(toolId);
        event.toolExecutionId = toolId;
      }
      const parent = pending?.parent ?? asString(message.parent_tool_use_id ?? undefined);
      if (parent) event.parentSpanId = spanIdFrom(parent);
      if (isError) {
        event.error = { message: stringifyContent(record.content).slice(0, 2_000) };
      }

      session.record(event);
    }
  }

  private handleResult(message: ClaudeSDKMessage): void {
    const session = this.ensureSession(message);
    this.flushPendingTools();
    const isError = message.is_error === true || message.subtype !== 'success';
    const result = message.result;

    const sections: TraceSection[] = [
      {
        kind: 'metadata',
        label: 'Run summary',
        content: {
          num_turns: message.num_turns,
          total_cost_usd: message.total_cost_usd,
          duration_api_ms: message.duration_api_ms,
          stop_reason: message.stop_reason,
          permission_denials: message.permission_denials,
        },
      },
    ];
    if (result) {
      sections.push({
        kind: 'message',
        label: 'Final answer',
        role: 'assistant',
        content: result,
      });
    }

    session.record({
      type: 'span',
      label: 'Result',
      status: isError ? 'error' : 'success',
      durationMs: asNumber(message.duration_ms),
      sections,
      // The run-level `usage` is CUMULATIVE across turns and is deliberately
      // not added here: the totals already came from the per-message usage,
      // and re-adding it would inflate a multi-turn session quadratically.
      metadata: { costUsd: message.total_cost_usd },
    });

    this.session = undefined;
    this.closing = session.end({ status: isError ? 'error' : 'success' });
  }

  /** Emit calls whose result never arrived (interrupt, denial, crash). */
  private flushPendingTools(): void {
    const session = this.session;
    if (!session) {
      this.pendingTools.clear();
      return;
    }
    for (const [toolId, pending] of this.pendingTools) {
      session.record({
        id: toolId,
        type: 'tool_call',
        label: pending.name,
        toolName: pending.name,
        toolExecutionId: toolId,
        spanId: spanIdFrom(toolId),
        status: 'error',
        error: { message: 'Tool call did not complete' },
        sections: [
          {
            kind: 'tool_call',
            label: 'Arguments',
            tool: pending.name,
            content: pending.input,
          },
        ],
      });
    }
    this.pendingTools.clear();
  }
}

/* ------------------------------------------------------------------ */
/*  Drop-in query wrapper                                             */
/* ------------------------------------------------------------------ */

/** Kept out of the `import()` call site so it stays an unresolved specifier. */
const CLAUDE_AGENT_SDK_MODULE = '@anthropic-ai/claude-agent-sdk';

export interface TraceQueryOptions extends ClaudeMessageTracerOptions {
  /** Passed straight through to the SDK's `query()`. */
  prompt: unknown;
  /** Passed straight through to the SDK's `query()`. */
  options?: Record<string, unknown>;
}

/**
 * Drop-in replacement for `query()` from `@anthropic-ai/claude-agent-sdk`
 * that also traces.
 *
 * Yields the SDK's messages unchanged, so existing consumer code keeps
 * working. The session is closed when the stream ends — including on an
 * exception and on an early `break`, which drives the generator's `finally`.
 */
export async function* traceQuery(
  params: TraceQueryOptions,
): AsyncGenerator<ClaudeSDKMessage, void, undefined> {
  const { prompt, options, ...tracerOptions } = params;

  let sdk: { query?: (args: { prompt: unknown; options?: unknown }) => AsyncIterable<ClaudeSDKMessage> };
  try {
    // The specifier goes through a variable on purpose: TypeScript only
    // resolves `import()` when its argument is a string literal, and this
    // package must type-check and build for users who have not installed
    // this optional peer.
    sdk = (await import(/* @vite-ignore */ CLAUDE_AGENT_SDK_MODULE)) as typeof sdk;
  } catch (error) {
    throw new Error(
      '@cognipeer/observability/claude-agent-sdk requires the Claude Agent SDK. ' +
        'Install it with `npm install @anthropic-ai/claude-agent-sdk`.',
      { cause: error },
    );
  }
  if (!sdk.query) {
    throw new Error('@anthropic-ai/claude-agent-sdk does not export `query`.');
  }

  const tracer = new ClaudeMessageTracer(tracerOptions);
  let failure: unknown;
  try {
    for await (const message of sdk.query({ prompt, options })) {
      tracer.handle(message);
      yield message;
    }
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    // Runs on normal completion, on throw, and on the consumer's `break`
    // (which calls the generator's `return()`).
    await tracer.close(
      failure === undefined ? {} : { status: 'error', error: failure },
    );
  }
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function contentBlocks(message: Record<string, unknown>): unknown[] {
  const content = message.content;
  if (Array.isArray(content)) return content;
  if (content === undefined || content === null) return [];
  return [content];
}

/**
 * Translate Anthropic usage into the Console's token model.
 *
 * Anthropic reports `input_tokens` EXCLUSIVE of cached tokens, while the
 * Console treats `cachedInputTokens` as a SUBSET of `inputTokens` when pricing
 * a call. Adding the cache buckets back into the input total is what makes the
 * cost come out right; leaving them out would under-report a cache-heavy
 * agent's prompt volume by an order of magnitude.
 */
function usageOf(usage: Record<string, unknown> | undefined): Partial<TraceEvent> {
  if (!usage) return {};
  const fresh = asNumber(usage.input_tokens) ?? 0;
  const cacheRead = asNumber(usage.cache_read_input_tokens) ?? 0;
  const cacheWrite = asNumber(usage.cache_creation_input_tokens) ?? 0;
  const output = asNumber(usage.output_tokens) ?? 0;

  const totalInput = fresh + cacheRead + cacheWrite;
  const out: Partial<TraceEvent> = {};
  if (totalInput) out.inputTokens = totalInput;
  if (output) out.outputTokens = output;
  if (cacheRead) out.cachedInputTokens = cacheRead;
  if (totalInput || output) out.totalTokens = totalInput + output;
  return out;
}

function assistantMetadata(
  message: ClaudeSDKMessage,
  inner: Record<string, unknown>,
): Record<string, unknown> {
  const metadata: Record<string, unknown> = {};
  for (const key of ['request_id', 'subagent_type', 'task_description', 'session_id'] as const) {
    if (message[key] !== undefined && message[key] !== null) metadata[key] = message[key];
  }
  if (inner.stop_reason !== undefined && inner.stop_reason !== null) {
    metadata.stop_reason = inner.stop_reason;
  }
  const cacheWrite = asNumber(asRecord(message.usage)?.cache_creation_input_tokens ??
    asRecord(inner.usage)?.cache_creation_input_tokens);
  if (cacheWrite) {
    // Not a Console field, but worth surfacing: cache writes are billed above
    // the normal input rate.
    metadata.cacheCreationInputTokens = cacheWrite;
  }
  // Events already shipped cannot be retracted, so a refusal-fallback that
  // supersedes earlier messages is recorded rather than applied.
  if (Array.isArray(message.supersedes) && message.supersedes.length > 0) {
    metadata.supersedes = message.supersedes;
  }
  return metadata;
}
