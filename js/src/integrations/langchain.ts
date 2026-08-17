/**
 * LangChain / LangGraph callback handler (JavaScript).
 *
 * One handler covers both: LangGraph runs its graph through LangChain's
 * callback manager, so nodes, models, tools and retrievers all arrive here.
 * Attach it per-invocation:
 *
 * ```ts
 * import { CognipeerCallbackHandler } from '@cognipeer/observability/langchain';
 *
 * await chain.invoke(input, { callbacks: [new CognipeerCallbackHandler()] });
 * ```
 *
 * or globally, for every chain in the process:
 *
 * ```ts
 * import { installLangChainTracing } from '@cognipeer/observability/langchain';
 * await installLangChainTracing();
 * ```
 *
 * **Serverless.** LangChain's JS callbacks are fire-and-forget unless a handler
 * opts in, so this one sets `_awaitHandler` (see the constructor). On platforms
 * that freeze the process the moment the response is returned — Lambda, Vercel,
 * Cloudflare — also `await awaitAllCallbacks()` from
 * `@langchain/core/callbacks/promises`, then `await flush()` from
 * `@cognipeer/observability`, before returning.
 *
 * **Version support.** Every payload is read defensively, so the handler works
 * from `@langchain/core` 0.1 through 1.x. Where newer versions expose better
 * data it is preferred and older ones fall back:
 *
 * - token usage — `usage_metadata` on the generation's message (0.2.9+, the
 *   only source with a cache-read breakdown) → `llmOutput.tokenUsage`;
 * - model name — `metadata.ls_model_name` (0.2+) →
 *   `invocation_params.model | model_name` → `llmOutput.model_name`;
 * - tool-call id — an extra `handleToolStart` argument on ~1.1.2x+;
 * - thread id — `metadata.thread_id`, which older versions copied out of
 *   `configurable`. Newer ones do not: use {@link cognipeerConfig}.
 */

import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { DocumentInterface } from '@langchain/core/documents';
import type { Serialized } from '@langchain/core/load/serializable';
import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ChainValues } from '@langchain/core/utils/types';

import { getClient, type CognipeerObservability } from '../core/client';
import type { TraceSession } from '../core/session';
import type { TraceAgent, TraceEvent, TraceResponseFormat, TraceSection, TraceToolDefinition } from '../core/types';

/**
 * Runnables LangChain and LangGraph create for plumbing. Recording them buries
 * the model and tool calls the user actually came to look at — a single
 * `createReactAgent` turn emits dozens of them.
 */
export const NOISE_CHAIN_NAMES: ReadonlySet<string> = new Set([
  'ChannelRead',
  'ChannelWrite',
  'RunnableAssign',
  'RunnableBinding',
  'RunnableMap',
  'RunnableParallel',
  'RunnablePassthrough',
  'RunnableSeq',
  'RunnableSequence',
  '_write',
  '__start__',
  '__end__',
]);

/**
 * LangGraph signals human-in-the-loop pauses and cross-graph commands by
 * throwing, and those throws reach `handleChainError`. Mapping them to a failed
 * session would mark every approval prompt as an error.
 */
const CONTROL_FLOW_ERROR_NAMES: ReadonlySet<string> = new Set([
  'GraphBubbleUp',
  'GraphInterrupt',
  'NodeInterrupt',
  'ParentCommand',
]);

const ROLE_BY_MESSAGE_TYPE: Record<string, string> = {
  human: 'user',
  ai: 'assistant',
  system: 'system',
  developer: 'system',
  tool: 'tool',
  function: 'tool',
  generic: 'user',
  remove: 'system',
};

export interface CognipeerCallbackHandlerOptions {
  /** Client to send through. Defaults to the process-wide one. */
  client?: CognipeerObservability;
  /**
   * Send into an existing session instead of opening one. The handler never
   * ends a session it did not create.
   */
  session?: TraceSession;
  /** Fixed session id — pass the same value again to append to an existing run. */
  sessionId?: string;
  /** Conversation id, so several runs group into one thread in the Console. */
  threadId?: string;
  /** Agent identity for the session. */
  agent?: TraceAgent;
  /**
   * Record chain / graph-node runs as spans (default true). Plumbing
   * runnables (see {@link NOISE_CHAIN_NAMES}) are always skipped.
   */
  captureChains?: boolean;
  /** Extra metadata attached to every event. */
  metadata?: Record<string, unknown>;
}

/** Turns LangChain run events into one Cognipeer tracing session. */
export class CognipeerCallbackHandler extends BaseCallbackHandler {
  name = 'cognipeer-tracing';

  private readonly client: CognipeerObservability;
  private readonly ownsSession: boolean;
  private readonly fixedSessionId?: string;
  private readonly agent?: TraceAgent;
  private readonly captureChains: boolean;
  private readonly extraMetadata: Record<string, unknown>;

  private activeSession?: TraceSession;
  private threadId?: string;
  /** run id of the run that opened the session — its end closes it. */
  private rootRun?: string;
  /** Model name per run, remembered from `start` for use in `end`. */
  private readonly models = new Map<string, string>();
  /**
   * Plumbing runs we chose not to record, mapped to their own parent. Children
   * resolve through this so the tree never points at a span that was never
   * sent.
   */
  private readonly skipped = new Map<string, string | undefined>();
  /**
   * Delivery of the session this handler most recently closed, still in
   * flight. `end()` deliberately is not awaited on the callback path, so this
   * is what `flush()` waits on — and it is the only remaining handle, since a
   * closed session is no longer tracked by the client.
   */
  private closing?: Promise<void>;

  constructor(options: CognipeerCallbackHandlerOptions = {}) {
    // `awaitHandlers` defaults to false, which makes LangChain queue handler
    // bodies and NOT await them. On a serverless runtime the process can be
    // frozen before the export leaves, so opt in to being awaited.
    super({ _awaitHandler: true });

    this.client = options.client ?? getClient();
    this.activeSession = options.session;
    this.ownsSession = options.session === undefined;
    this.fixedSessionId = options.sessionId;
    this.threadId = options.threadId;
    this.agent = options.agent;
    this.captureChains = options.captureChains ?? true;
    this.extraMetadata = options.metadata ?? {};
  }

  /** The session this handler is writing into, once one exists. */
  get session(): TraceSession | undefined {
    return this.activeSession;
  }

  /**
   * Await delivery of everything this handler has produced.
   *
   * Covers both the open session and the one most recently closed, whose
   * delivery is intentionally left running in the background so it never
   * blocks a chain. A short-lived process — a script, a Lambda handler — should
   * await this before returning.
   */
  async flush(): Promise<void> {
    await Promise.all([this.activeSession?.flush(), this.closing]);
  }

  // ── Chat / LLM ─────────────────────────────────────────────────────────

  handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string,
  ): void {
    this.guard(() => {
      const session = this.ensureSession(runId, metadata);
      const params = invocationParams(extraParams);
      const model = modelName(params, metadata);
      const sections: TraceSection[] = [];
      for (const batch of messages ?? []) sections.push(...messagesToSections(batch));

      session.openSpan(runId, {
        type: 'ai_call',
        label: model ?? runName ?? serializedName(llm) ?? 'model',
        parentKey: this.parentKey(parentRunId),
        model,
        actor: { scope: 'model', name: model ?? 'model' },
        sections,
        metadata: this.eventMetadata(tags, metadata, params),
        toolDefinitions: toolDefinitions(extraParams),
        responseFormat: responseFormat(extraParams),
      });
      if (model) this.models.set(runId, model);
    });
  }

  handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string,
  ): void {
    this.guard(() => {
      const session = this.ensureSession(runId, metadata);
      const params = invocationParams(extraParams);
      const model = modelName(params, metadata);

      session.openSpan(runId, {
        type: 'ai_call',
        label: model ?? runName ?? serializedName(llm) ?? 'model',
        parentKey: this.parentKey(parentRunId),
        model,
        actor: { scope: 'model', name: model ?? 'model' },
        sections: (prompts ?? []).map((prompt) => ({
          kind: 'message',
          label: 'Prompt',
          role: 'user',
          content: prompt,
        })),
        metadata: this.eventMetadata(tags, metadata, params),
        toolDefinitions: toolDefinitions(extraParams),
        responseFormat: responseFormat(extraParams),
      });
      if (model) this.models.set(runId, model);
    });
  }

  async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      const usage = extractUsage(output);
      this.activeSession.closeSpan(runId, {
        status: 'success',
        model: responseModel(output) ?? this.models.get(runId),
        sections: generationsToSections(output),
        ...usage,
      });
      this.models.delete(runId);
      await this.maybeEndSession(runId, 'success');
    });
  }

  async handleLLMError(error: unknown, runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      this.activeSession.closeSpan(runId, { status: 'error', error });
      this.models.delete(runId);
      await this.maybeEndSession(runId, 'error', error);
    });
  }

  // ── Tools ──────────────────────────────────────────────────────────────

  /**
   * Declared with a rest signature on purpose. The tool-call id was appended as
   * an eighth argument in a later 1.1.x release, so a fixed arity would either
   * drop it on new versions or fail to type-check against old ones.
   *
   * Runtime order: `(tool, input, runId, parentRunId, tags, metadata, runName,
   * toolCallId)`.
   */
  handleToolStart(...args: unknown[]): void {
    this.guard(() => {
      const [tool, input, runId, parentRunId, tags, metadata, runName, toolCallId] = args as [
        Serialized | undefined,
        string | undefined,
        string,
        string | undefined,
        string[] | undefined,
        Record<string, unknown> | undefined,
        string | undefined,
        string | undefined,
      ];

      const session = this.ensureSession(runId, metadata);
      const name = serializedName(tool) ?? runName ?? 'tool';

      session.openSpan(runId, {
        type: 'tool_call',
        label: name,
        parentKey: this.parentKey(parentRunId),
        toolName: name,
        // Correlates this run with the assistant tool_call that requested it.
        toolExecutionId: toolCallId,
        actor: { scope: 'tool', name },
        sections: [
          {
            kind: 'tool_call',
            label: 'Arguments',
            tool: name,
            // JS hands tool arguments over already stringified, unlike Python.
            content: maybeJson(input),
          },
        ],
        metadata: this.eventMetadata(tags, metadata, undefined),
      });
    });
  }

  async handleToolEnd(output: unknown, runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      this.activeSession.closeSpan(runId, {
        status: 'success',
        sections: [{ kind: 'tool_result', label: 'Result', content: toolOutput(output) }],
      });
      await this.maybeEndSession(runId, 'success');
    });
  }

  async handleToolError(error: unknown, runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      this.activeSession.closeSpan(runId, { status: 'error', error });
      await this.maybeEndSession(runId, 'error', error);
    });
  }

  // ── Retrievers ─────────────────────────────────────────────────────────

  handleRetrieverStart(
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string,
  ): void {
    this.guard(() => {
      const session = this.ensureSession(runId, metadata);
      const name = serializedName(retriever) ?? runName ?? 'retriever';
      session.openSpan(runId, {
        type: 'retrieval',
        label: name,
        parentKey: this.parentKey(parentRunId),
        actor: { scope: 'retriever', name },
        sections: [{ kind: 'message', label: 'Query', role: 'user', content: query }],
        metadata: this.eventMetadata(tags, metadata, undefined),
      });
    });
  }

  async handleRetrieverEnd(documents: DocumentInterface[], runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      const docs = documents ?? [];
      this.activeSession.closeSpan(runId, {
        status: 'success',
        sections: [
          {
            kind: 'metadata',
            label: `Documents (${docs.length})`,
            content: docs.map((doc) => ({
              pageContent: doc?.pageContent,
              metadata: doc?.metadata,
            })),
          },
        ],
      });
      await this.maybeEndSession(runId, 'success');
    });
  }

  async handleRetrieverError(error: unknown, runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      this.activeSession.closeSpan(runId, { status: 'error', error });
      await this.maybeEndSession(runId, 'error', error);
    });
  }

  // ── Chains / graph nodes ───────────────────────────────────────────────

  /**
   * THE ARGUMENT ORDER BELOW IS THE RUNTIME ORDER, NOT THE DECLARED ONE.
   *
   * `@langchain/core` >= 1.1.15 ships a `.d.ts` declaring
   * `(chain, inputs, runId, runType?, tags?, metadata?, runName?, parentRunId?)`
   * while `CallbackManager` actually dispatches
   * `(chain, inputs, runId, parentRunId, tags, metadata, runType, runName, extra)`.
   * Both orders are structurally `(…, string?, string[]?, object?, string?,
   * string?)`, so TypeScript accepts either and the mistake is invisible:
   * trusting the declaration reads `parentRunId` as `runType` and collapses the
   * entire chain hierarchy into a flat list.
   */
  handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    _runType?: string,
    runName?: string,
  ): void {
    this.guard(() => {
      const node = stringOrUndefined(metadata?.langgraph_node);
      const name = node ?? runName ?? serializedName(chain) ?? 'chain';

      // The outermost chain opens the session even when chain capture is off,
      // so the run still has a root to hang model and tool events under.
      const session = this.ensureSession(runId, metadata);
      if (!this.captureChains || NOISE_CHAIN_NAMES.has(name) || isHidden(tags)) {
        this.skipped.set(runId, this.parentKey(parentRunId));
        return;
      }

      session.openSpan(runId, {
        type: 'span',
        label: name,
        parentKey: this.parentKey(parentRunId),
        actor: { scope: 'agent', name },
        sections: [{ kind: 'metadata', label: 'Input', content: chainIo(inputs) }],
        metadata: this.eventMetadata(tags, metadata, undefined),
      });
    });
  }

  async handleChainEnd(outputs: ChainValues, runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      if (this.activeSession.hasSpan(runId)) {
        this.activeSession.closeSpan(runId, {
          status: 'success',
          sections: [{ kind: 'metadata', label: 'Output', content: chainIo(outputs) }],
        });
      }
      await this.maybeEndSession(runId, 'success');
    });
  }

  async handleChainError(error: unknown, runId: string): Promise<void> {
    await this.guardAsync(async () => {
      if (!this.activeSession) return;
      // A LangGraph interrupt is normal control flow, not a failure.
      const interrupted = isControlFlowError(error);
      const status = interrupted ? 'success' : 'error';

      if (this.activeSession.hasSpan(runId)) {
        this.activeSession.closeSpan(runId, {
          status,
          ...(interrupted
            ? {
                sections: [
                  { kind: 'metadata', label: 'Interrupted', content: describeError(error) },
                ],
              }
            : { error }),
        });
      }
      await this.maybeEndSession(runId, status, interrupted ? undefined : error);
    });
  }

  // ── Session plumbing ───────────────────────────────────────────────────

  private ensureSession(runId: string, metadata?: Record<string, unknown>): TraceSession {
    // Remember the thread id once seen: a follow-up tool or model run started
    // outside the graph carries no `configurable`, and its session should still
    // land in the same conversation.
    this.threadId ??= threadIdFrom(metadata);

    // The first run seen owns the per-run bookkeeping, whether or not we
    // opened the session ourselves. With a caller-supplied session this is the
    // ONLY thing that ever arms `rootRun`, and without it the skip map grows
    // by one entry per plumbing runnable for the handler's entire life.
    this.rootRun ??= runId;

    if (!this.activeSession) {
      this.activeSession = this.client.startSession({
        sessionId: this.fixedSessionId,
        threadId: this.threadId,
        agent: this.agent,
      });
    }
    return this.activeSession;
  }

  /**
   * Nearest ancestor that was actually recorded. Plumbing runs are skipped, so
   * a child's literal `parentRunId` can name a span that was never sent;
   * walking the skip map keeps the tree connected instead of orphaning it.
   */
  private parentKey(parentRunId?: string): string | undefined {
    let key = parentRunId;
    const seen = new Set<string>();
    while (key !== undefined && this.skipped.has(key) && !seen.has(key)) {
      seen.add(key);
      key = this.skipped.get(key);
    }
    return key;
  }

  /**
   * Close the session when the run that opened it finishes, then forget it —
   * so reusing one handler across several `invoke()` calls produces one session
   * per call, which is what the Console's Threads view expects.
   *
   * Forgetting it matters even when `sessionId` is fixed: an ended session
   * rejects further events, so keeping the object would silently drop every
   * event of the next run. A fresh object under the same id is what makes
   * "append to one long-lived session" work — the server's reopen path merges
   * it and keeps the running totals.
   */
  private async maybeEndSession(runId: string, status: string, error?: unknown): Promise<void> {
    if (this.rootRun !== runId) return;

    // Per-run bookkeeping is released whoever owns the session — a handler
    // given an external one still accumulates a skip-map entry per plumbing
    // runnable per run, and nothing else ever clears it.
    this.rootRun = undefined;
    this.skipped.clear();
    this.models.clear();

    const session = this.activeSession;
    // Never end a session we did not create; the caller owns its lifetime.
    if (!this.ownsSession || !session) return;
    this.activeSession = undefined;

    // Deliberately NOT awaited. This handler opts into `_awaitHandler` so that
    // serverless runtimes do not freeze before the export leaves — but that
    // makes LangChain await the handler body, so awaiting delivery here would
    // put the whole retry/backoff sequence on the traced application's own
    // critical path: measured at ~24 s of added latency against an unreachable
    // Console with the default three retries. `end()` enqueues everything
    // synchronously before its first suspension, so nothing is lost by letting
    // it drain in the background; `flush()` below is how a short-lived process
    // waits for it.
    this.closing = session.end({ status, error });
  }

  private eventMetadata(
    tags: string[] | undefined,
    metadata: Record<string, unknown> | undefined,
    params: Record<string, unknown> | undefined,
  ): Record<string, unknown> {
    const out: Record<string, unknown> = { ...this.extraMetadata };
    if (tags?.length) out.tags = [...tags];
    for (const key of [
      'langgraph_node',
      'langgraph_step',
      'langgraph_path',
      'checkpoint_ns',
      'ls_provider',
    ]) {
      const value = metadata?.[key];
      if (value !== undefined) out[key] = value;
    }
    for (const key of ['temperature', 'max_tokens', 'top_p', 'stop']) {
      const value = params?.[key];
      if (value !== undefined) out[key] = value;
    }
    return out;
  }

  /**
   * LangChain logs and swallows handler exceptions, which would make a broken
   * exporter look like "tracing just doesn't work". Contain the failure here
   * instead so it can never reach user code either way.
   */
  private guard(fn: () => void): void {
    try {
      fn();
    } catch (error) {
      this.client.config.logger.warn('langchain handler error:', error);
    }
  }

  private async guardAsync(fn: () => Promise<void>): Promise<void> {
    try {
      await fn();
    } catch (error) {
      this.client.config.logger.warn('langchain handler error:', error);
    }
  }
}

// ── Configuration helpers ────────────────────────────────────────────────

/** The part of `CallbackManager` this module needs, without importing it. */
interface CallbackManagerLike {
  addHandler(handler: BaseCallbackHandler, inherit?: boolean): void;
}

export interface CognipeerConfigOptions extends CognipeerCallbackHandlerOptions {
  /** Existing config to extend. Its callbacks are preserved. */
  base?: RunnableConfig;
  /** Reuse a handler instead of creating one. */
  handler?: CognipeerCallbackHandler;
}

/**
 * Build a `RunnableConfig` that traces correctly on every version.
 *
 * `threadId` is written to BOTH `configurable` (where LangGraph's checkpointer
 * needs it) and `metadata` (where a third-party callback handler can still see
 * it). Recent `@langchain/core` releases stopped copying `configurable` into
 * `metadata` for anything that is not a LangSmith tracer, so without this the
 * handler silently loses conversation grouping.
 *
 * ```ts
 * await agent.invoke(state, cognipeerConfig('conv-42'));
 * ```
 */
export function cognipeerConfig(
  threadId?: string,
  options: CognipeerConfigOptions = {},
): RunnableConfig {
  const { base, handler, ...handlerOptions } = options;
  const config: RunnableConfig = { ...base };
  const ours = handler ?? new CognipeerCallbackHandler({ threadId, ...handlerOptions });

  // `callbacks` is either an array of handlers or an already-built
  // CallbackManager. Overwriting the latter with an array would silently drop
  // every handler the caller had configured, so add to it in place instead.
  const existing = config.callbacks as unknown;
  if (existing && !Array.isArray(existing) && typeof (existing as CallbackManagerLike).addHandler === 'function') {
    (existing as CallbackManagerLike).addHandler(ours, true);
  } else {
    config.callbacks = [...(Array.isArray(existing) ? existing : []), ours];
  }

  if (threadId) {
    config.configurable = { ...config.configurable, thread_id: threadId };
    config.metadata = { ...config.metadata, thread_id: threadId };
  }
  return config;
}

/**
 * Attach a handler to every LangChain run in this process, using
 * `@langchain/core`'s global configure hook — no per-invoke config needed.
 *
 * Two caveats worth knowing:
 *
 * - every run in the process shares ONE handler, which is right for a script
 *   and wrong for a server handling concurrent unrelated conversations; there,
 *   pass a per-invocation handler via {@link cognipeerConfig} instead;
 * - the hook is backed by `AsyncLocalStorage`, so it must be installed before
 *   the work that should be traced is started.
 *
 * The `@langchain/core/context` entry point only exists from 0.3.0; on older
 * versions this warns and returns the handler unattached.
 */
export async function installLangChainTracing(
  options: CognipeerCallbackHandlerOptions = {},
): Promise<CognipeerCallbackHandler> {
  const handler = new CognipeerCallbackHandler(options);
  const client = options.client ?? getClient();

  try {
    const context = (await import('@langchain/core/context')) as {
      registerConfigureHook: (config: { contextVar: string; inheritable?: boolean }) => void;
      setContextVariable: (name: string, value: unknown) => void;
    };
    context.registerConfigureHook({ contextVar: CONTEXT_VAR, inheritable: true });
    context.setContextVariable(CONTEXT_VAR, handler);
  } catch {
    client.config.logger.warn(
      'this @langchain/core version has no global callback hook; pass the handler via ' +
        "config={ callbacks: [...] } (or cognipeerConfig()) instead",
    );
  }
  return handler;
}

const CONTEXT_VAR = 'cognipeer_handler';

// ── Extraction helpers ───────────────────────────────────────────────────

function invocationParams(extraParams?: Record<string, unknown>): Record<string, unknown> | undefined {
  const params = extraParams?.invocation_params;
  return isRecord(params) ? params : undefined;
}

function modelName(
  params?: Record<string, unknown>,
  metadata?: Record<string, unknown>,
): string | undefined {
  const ls = stringOrUndefined(metadata?.ls_model_name);
  if (ls) return ls;
  for (const key of ['model', 'model_name', 'modelName', 'model_id', 'deployment_name']) {
    const value = stringOrUndefined(params?.[key]);
    if (value) return value;
  }
  return undefined;
}

function responseModel(output: LLMResult): string | undefined {
  const llmOutput = (output as { llmOutput?: Record<string, unknown> }).llmOutput;
  for (const key of ['model_name', 'model', 'modelName']) {
    const value = stringOrUndefined(llmOutput?.[key]);
    if (value) return value;
  }
  for (const generation of firstGenerations(output)) {
    const metadata = (generation as { message?: { response_metadata?: Record<string, unknown> } })
      .message?.response_metadata;
    for (const key of ['model_name', 'model']) {
      const value = stringOrUndefined(metadata?.[key]);
      if (value) return value;
    }
  }
  return undefined;
}

/**
 * Token usage, newest source first.
 *
 * `usage_metadata` is normalized across providers and is the only source
 * carrying the cache-read breakdown. Its `input_tokens` already INCLUDES cached
 * tokens — which is exactly what the Console's pricing expects, since it treats
 * cached tokens as a subset of the input count.
 *
 * Every field stays `undefined` when the provider reported nothing. Streaming
 * only reports usage when explicitly enabled (`streamUsage: true` on
 * ChatOpenAI, and some providers never do), and recording that as zero would
 * silently under-report spend rather than showing the gap.
 */
function extractUsage(output: LLMResult): {
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  totalTokens?: number;
} {
  for (const generation of firstGenerations(output)) {
    const usage = (generation as { message?: { usage_metadata?: Record<string, unknown> } }).message
      ?.usage_metadata;
    if (isRecord(usage)) {
      const details = isRecord(usage.input_token_details) ? usage.input_token_details : undefined;
      return {
        inputTokens: numberOrUndefined(usage.input_tokens),
        outputTokens: numberOrUndefined(usage.output_tokens),
        totalTokens: numberOrUndefined(usage.total_tokens),
        cachedInputTokens: numberOrUndefined(details?.cache_read),
      };
    }
  }

  const llmOutput = (output as { llmOutput?: Record<string, unknown> }).llmOutput;
  const legacy = [llmOutput?.tokenUsage, llmOutput?.estimatedTokenUsage, llmOutput?.usage].find(
    isRecord,
  );
  if (legacy) {
    const details = isRecord(legacy.prompt_tokens_details) ? legacy.prompt_tokens_details : undefined;
    return {
      inputTokens: numberOrUndefined(legacy.promptTokens ?? legacy.prompt_tokens ?? legacy.input_tokens),
      outputTokens: numberOrUndefined(
        legacy.completionTokens ?? legacy.completion_tokens ?? legacy.output_tokens,
      ),
      totalTokens: numberOrUndefined(legacy.totalTokens ?? legacy.total_tokens),
      cachedInputTokens: numberOrUndefined(
        details?.cached_tokens ?? legacy.cache_read_input_tokens,
      ),
    };
  }

  return {};
}

/**
 * The tool menu offered to the model on THIS call.
 *
 * `bindTools()` puts the provider-formatted schemas into `invocation_params`.
 * They are stripped from the *tracing* metadata LangChain builds, but the
 * unfiltered object still reaches handlers — so this is the only place a
 * third-party integration can see them.
 */
function toolDefinitions(extraParams?: Record<string, unknown>): TraceToolDefinition[] | undefined {
  const params = invocationParams(extraParams);
  const options = isRecord(extraParams?.options) ? extraParams.options : undefined;
  const raw =
    params?.tools ?? params?.functions ?? params?.tool_schemas ?? options?.tools ?? options?.functions;
  if (!Array.isArray(raw)) return undefined;

  const out: TraceToolDefinition[] = [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    const fn = isRecord(entry.function) ? entry.function : entry;
    const name = stringOrUndefined(fn.name);
    if (!name) {
      // Hosted/provider tools carry only a type (web_search, file_search, …).
      const type = stringOrUndefined(entry.type);
      if (type) out.push({ name: type });
      continue;
    }
    const description = stringOrUndefined(fn.description);
    const parameters = [fn.parameters, fn.input_schema, fn.inputSchema].find(isRecord);
    out.push({
      name,
      ...(description ? { description } : {}),
      ...(parameters ? { parameters } : {}),
    });
  }
  return out.length > 0 ? out : undefined;
}

/**
 * The structured-output contract LangChain bound for this call.
 *
 * `withStructuredOutput()` / a `response_format` call option ends up in the
 * same `invocation_params` bag the tool schemas do, so it is visible here and
 * nowhere else. Without it a trace cannot distinguish a model that CHOSE prose
 * from one that was never asked for JSON.
 */
function responseFormat(extraParams?: Record<string, unknown>): TraceResponseFormat | undefined {
  const params = invocationParams(extraParams);
  const options = isRecord(extraParams?.options) ? extraParams.options : undefined;
  const raw = [
    params?.response_format,
    params?.responseFormat,
    options?.response_format,
    options?.responseFormat,
  ].find(isRecord);
  if (!raw) return undefined;

  const type = stringOrUndefined(raw.type);
  const jsonSchema = [raw.json_schema, raw.jsonSchema].find(isRecord);
  const source = jsonSchema ?? raw;
  const schema = isRecord(source.schema) ? source.schema : undefined;
  if (!type && !schema) return undefined;

  return {
    type: type ?? 'json_schema',
    ...(stringOrUndefined(source.name) ? { name: stringOrUndefined(source.name) as string } : {}),
    ...(typeof source.strict === 'boolean' ? { strict: source.strict } : {}),
    ...(schema ? { schema } : {}),
  };
}

function messagesToSections(messages: BaseMessage[]): TraceSection[] {
  const sections: TraceSection[] = [];
  for (const message of messages ?? []) {
    const role = messageRole(message);
    sections.push({
      kind: 'message',
      label: `${capitalize(role)} message`,
      role,
      content: (message as { content?: unknown }).content,
    });

    const toolCalls = (message as { tool_calls?: Array<Record<string, unknown>> }).tool_calls;
    for (const call of toolCalls ?? []) {
      const name = stringOrUndefined(call?.name) ?? 'tool';
      sections.push({
        kind: 'tool_call',
        label: `Tool call: ${name}`,
        tool: name,
        content: call?.args,
      });
    }
  }
  return sections;
}

function messageRole(message: BaseMessage): string {
  // `getType()` is the modern accessor; `_getType()` is the 0.1/0.2 spelling.
  const candidate = message as {
    getType?: () => string;
    _getType?: () => string;
    role?: string;
    type?: string;
  };
  const type = candidate.getType?.() ?? candidate._getType?.() ?? candidate.role ?? candidate.type;
  if (typeof type === 'string') return ROLE_BY_MESSAGE_TYPE[type] ?? type;

  const className = message?.constructor?.name ?? '';
  if (className.startsWith('Human')) return 'user';
  if (className.startsWith('AI')) return 'assistant';
  if (className.startsWith('System')) return 'system';
  if (className.startsWith('Tool') || className.startsWith('Function')) return 'tool';
  return 'user';
}

function generationsToSections(output: LLMResult): TraceSection[] {
  const sections: TraceSection[] = [];
  for (const batch of output?.generations ?? []) {
    for (const generation of batch ?? []) {
      const message = (generation as { message?: BaseMessage }).message;
      if (message) {
        sections.push(...messagesToSections([message]));
        continue;
      }
      const text = (generation as { text?: string }).text;
      if (text) {
        sections.push({
          kind: 'message',
          label: 'Assistant message',
          role: 'assistant',
          content: text,
        });
      }
    }
  }
  return sections;
}

function firstGenerations(output: LLMResult): unknown[] {
  return (output?.generations ?? []).flat();
}

/**
 * Render chain / graph-node inputs and outputs.
 *
 * LangGraph state carries the entire message list on every node, at every
 * step — rendering those as full message sections would repeat the whole
 * conversation once per node. Flatten them to a compact role/content list; the
 * model events already show the real prompts.
 */
function chainIo(payload: unknown): unknown {
  if (Array.isArray(payload)) return payload.map(chainIo);
  if (!isRecord(payload)) return payload;

  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(payload)) {
    if (key === 'messages' && Array.isArray(value)) {
      out[key] = value.map((message) => ({
        role: messageRole(message as BaseMessage),
        content: (message as { content?: unknown })?.content ?? message,
      }));
    } else {
      out[key] = value;
    }
  }
  return out;
}

/** A ToolMessage wraps the real payload — unwrap so the UI shows content. */
function toolOutput(output: unknown): unknown {
  const content = (output as { content?: unknown } | undefined)?.content;
  return content !== undefined ? content : output;
}

function threadIdFrom(metadata?: Record<string, unknown>): string | undefined {
  for (const key of ['thread_id', 'session_id', 'conversation_id']) {
    const value = metadata?.[key];
    if (value !== undefined && value !== null && value !== '') return String(value);
  }
  return undefined;
}

/** LangChain marks internal runs it does not want surfaced with this tag. */
function isHidden(tags?: string[]): boolean {
  return tags?.includes('langsmith:hidden') ?? false;
}

function isControlFlowError(error: unknown): boolean {
  const names = [
    (error as { name?: string })?.name,
    (error as object)?.constructor?.name,
  ].filter((name): name is string => typeof name === 'string');
  return names.some((name) => CONTROL_FLOW_ERROR_NAMES.has(name));
}

function describeError(error: unknown): unknown {
  if (error instanceof Error) return { name: error.name, message: error.message };
  return error;
}

function serializedName(serialized?: Serialized | null): string | undefined {
  // `serialized` is frequently `{}`, `null`, or a not_implemented stub, so it
  // is a label source of last resort — never an identity.
  const record = serialized as Record<string, unknown> | undefined | null;
  const name = stringOrUndefined(record?.name);
  if (name) return name;
  const id = record?.id;
  if (Array.isArray(id) && id.length > 0) return String(id[id.length - 1]);
  return undefined;
}

function maybeJson(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
