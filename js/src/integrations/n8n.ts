/**
 * n8n integration.
 *
 * n8n has no seam for injecting a callback into its AI Agent node, so this
 * integration works from its **execution run data** instead — which turns out
 * to be the richest source it has: prompts, completions, per-call token usage,
 * tool inputs and outputs, per-node durations and errors all live there.
 *
 * Two ways in, same mapper behind both:
 *
 * 1. **Bridge** (`N8nBridge`) — polls the public REST API
 *    (`GET /api/v1/executions?includeData=true`). Works on n8n Cloud and on
 *    self-hosted Community, needs nothing but an n8n API key, and is the
 *    fastest path to a customer's first trace.
 * 2. **External hook** (`createN8nExternalHook`) — a `workflow.postExecute`
 *    hook file mounted via `EXTERNAL_HOOK_FILES`. Self-hosted only, but it is
 *    push-based, so traces land the moment a run finishes.
 *
 * n8n's own OpenTelemetry export (`N8N_OTEL_*`) is a third option and is NOT
 * recommended for agent observability: the workflow/node spans carry no
 * prompts, completions or model names, LLM token attributes are gated behind
 * an Enterprise licence, and the exporter speaks OTLP protobuf only while the
 * Console's `/api/client/v1/traces` accepts OTLP JSON — so it needs a
 * collector in between. See the guide for that setup.
 *
 * Known limits of the run-data route, stated plainly because they affect what
 * the Console can show:
 * - **No cached-token counts.** n8n's LLM tracing records only prompt and
 *   completion tokens, so `cachedInputTokens` is always absent.
 * - **Estimated tokens are common.** When a provider reports no usage
 *   (streaming, cancellation), n8n substitutes a tiktoken estimate. Those
 *   events are marked `metadata.tokensEstimated` so a cost report can exclude
 *   them.
 * - **Tool definitions are reconstructed** from the workflow JSON, not from
 *   what the model literally received; such events carry
 *   `metadata.toolDefinitionsSource: 'n8n-workflow-json'`.
 */

import { getClient, type CognipeerObservability } from '../core/client';
import { newSpanId, spanIdFrom, traceIdFrom } from '../core/ids';
import type {
  TraceEvent,
  TraceSection,
  TraceSessionPayload,
  TraceToolDefinition,
} from '../core/types';

/* ------------------------------------------------------------------ */
/*  n8n shapes (structural — no dependency on `n8n-workflow`)          */
/* ------------------------------------------------------------------ */

/** Connection types n8n uses to wire sub-nodes into an agent node. */
export const AI_CONNECTIONS = {
  languageModel: 'ai_languageModel',
  tool: 'ai_tool',
  memory: 'ai_memory',
  embedding: 'ai_embedding',
  vectorStore: 'ai_vectorStore',
  retriever: 'ai_retriever',
  documentLoader: 'ai_document',
  outputParser: 'ai_outputParser',
  textSplitter: 'ai_textSplitter',
} as const;

/** Sub-node connection type → Console event type. */
const EVENT_TYPE_BY_CONNECTION: Record<string, TraceEvent['type']> = {
  [AI_CONNECTIONS.languageModel]: 'ai_call',
  [AI_CONNECTIONS.tool]: 'tool_call',
  [AI_CONNECTIONS.embedding]: 'embedding',
  [AI_CONNECTIONS.vectorStore]: 'retrieval',
  [AI_CONNECTIONS.retriever]: 'retrieval',
  [AI_CONNECTIONS.memory]: 'span',
};

export interface N8nNodeExecutionData {
  json?: Record<string, unknown>;
  [key: string]: unknown;
}

export type N8nTaskDataConnections = Record<string, Array<N8nNodeExecutionData[] | null>>;

export interface N8nTaskData {
  startTime?: number;
  executionTime?: number;
  executionIndex?: number;
  executionStatus?: string;
  source?: Array<
    { previousNode?: string; previousNodeOutput?: number; previousNodeRun?: number } | null
  >;
  data?: N8nTaskDataConnections;
  inputOverride?: N8nTaskDataConnections;
  error?: { message?: string; description?: string; [key: string]: unknown };
  metadata?: {
    tokenUsage?: { inputTokens?: number; outputTokens?: number };
    subRun?: Array<{ node?: string; runIndex?: number }>;
    [key: string]: unknown;
  };
}

export interface N8nRunExecutionData {
  resultData?: {
    runData?: Record<string, N8nTaskData[]>;
    lastNodeExecuted?: string;
    error?: { message?: string; [key: string]: unknown };
  };
}

export interface N8nExecution {
  id?: number | string;
  workflowId?: string;
  status?: string;
  mode?: string;
  startedAt?: string;
  stoppedAt?: string;
  finished?: boolean;
  data?: N8nRunExecutionData;
  customData?: Record<string, unknown>;
  workflowData?: N8nWorkflow;
}

export interface N8nWorkflowNode {
  id?: string;
  name?: string;
  type?: string;
  parameters?: Record<string, unknown>;
}

export interface N8nWorkflow {
  id?: string;
  name?: string;
  nodes?: N8nWorkflowNode[];
  /** `{ [sourceNode]: { [connectionType]: Array<Array<{ node: string }>> } }` */
  connections?: Record<string, Record<string, Array<Array<{ node?: string }> | null>>>;
}

/* ------------------------------------------------------------------ */
/*  Mapper                                                            */
/* ------------------------------------------------------------------ */

export interface MapExecutionOptions {
  /** Workflow JSON. Supplies node types, models and tool schemas. */
  workflow?: N8nWorkflow;
  /** Conversation id. Defaults to the memory node's resolved session key. */
  threadId?: string;
  /** Agent name for the session. Defaults to the workflow name. */
  agentName?: string;
  /** Session id. Defaults to `n8n-exec-<executionId>`. */
  sessionId?: string;
}

/**
 * Convert one n8n execution into a Console session payload.
 *
 * Pure and synchronous — the bridge and the external hook both call it, and it
 * is the unit worth testing against a captured execution JSON.
 */
export function mapN8nExecution(
  execution: N8nExecution,
  options: MapExecutionOptions = {},
): TraceSessionPayload | undefined {
  const runData = execution.data?.resultData?.runData;
  if (!runData || Object.keys(runData).length === 0) return undefined;

  const workflow = options.workflow ?? execution.workflowData;
  const executionId = String(execution.id ?? '');
  const sessionId = options.sessionId ?? `n8n-exec-${executionId || newSpanId()}`;
  const traceId = traceIdFrom(`n8n:${executionId || sessionId}`);
  const rootSpanId = spanIdFrom(`n8n:root:${executionId || sessionId}`);

  const nodeTypes = new Map<string, string>();
  for (const node of workflow?.nodes ?? []) {
    if (node.name) nodeTypes.set(node.name, node.type ?? '');
  }
  const parents = buildParentMap(workflow);
  const toolDefinitions = buildToolDefinitions(workflow);

  const events: TraceEvent[] = [];
  let sequence = 0;
  let totalInputTokens = 0;
  let totalOutputTokens = 0;

  // Node order is not guaranteed by object key order; `executionIndex` (or
  // `startTime`) is what actually reflects the run.
  const tasks: Array<{ nodeName: string; runIndex: number; task: N8nTaskData }> = [];
  for (const [nodeName, runs] of Object.entries(runData)) {
    runs.forEach((task, runIndex) => tasks.push({ nodeName, runIndex, task }));
  }
  const runCounts = new Map(
    Object.entries(runData).map(([nodeName, runs]) => [nodeName, runs.length]),
  );
  const subRunParents = buildSubRunParentMap(tasks);
  tasks.sort((a, b) => {
    const byIndex = (a.task.executionIndex ?? Infinity) - (b.task.executionIndex ?? Infinity);
    if (byIndex !== 0 && Number.isFinite(byIndex)) return byIndex;
    return (a.task.startTime ?? 0) - (b.task.startTime ?? 0);
  });

  for (const { nodeName, runIndex, task } of tasks) {
    const connection = detectConnectionType(task);
    const usage = extractUsage(task, connection);
    const spanId = spanIdFrom(`n8n:${executionId}:${nodeName}:${runIndex}`);
    const parentRun = resolveParentRun(
      nodeName,
      runIndex,
      task,
      parents,
      subRunParents,
      runCounts,
    );
    const failed = task.executionStatus === 'error' || Boolean(task.error);

    const event: TraceEvent = {
      id: `${nodeName}#${runIndex}`,
      type: connection ? (EVENT_TYPE_BY_CONNECTION[connection] ?? 'span') : 'span',
      label: nodeName,
      sequence: ++sequence,
      traceId,
      spanId,
      parentSpanId: parentRun
        ? spanIdFrom(`n8n:${executionId}:${parentRun.node}:${parentRun.runIndex}`)
        : rootSpanId,
      timestamp: task.startTime ? new Date(task.startTime).toISOString() : undefined,
      durationMs: task.executionTime,
      status: failed ? 'error' : 'success',
      actor: buildActor(nodeName, connection),
      sections: buildSections(task, connection),
      metadata: {
        nodeType: nodeTypes.get(nodeName),
        runIndex,
        ...(usage.estimated ? { tokensEstimated: true } : {}),
      },
    };

    const model = resolveModel(task, workflow?.nodes?.find((node) => node.name === nodeName));
    if (model) event.model = model;
    if (connection === AI_CONNECTIONS.tool) event.toolName = nodeName;
    if (usage.inputTokens !== undefined) {
      event.inputTokens = usage.inputTokens;
      totalInputTokens += usage.inputTokens;
    }
    if (usage.outputTokens !== undefined) {
      event.outputTokens = usage.outputTokens;
      totalOutputTokens += usage.outputTokens;
    }
    if (usage.totalTokens !== undefined) event.totalTokens = usage.totalTokens;
    if (failed) {
      event.error = task.error?.message ?? task.error?.description ?? 'Node failed';
    }

    // The tool menu is reconstructed from the workflow definition rather than
    // observed on the call, so it is attached to the agent node that owns the
    // tools and labelled with its provenance.
    if (toolDefinitions.length > 0 && isAgentNode(nodeTypes.get(nodeName))) {
      event.toolDefinitions = toolDefinitions;
      event.metadata = {
        ...event.metadata,
        toolDefinitionsSource: 'n8n-workflow-json',
      };
    }

    events.push(event);
  }

  const startedAt = execution.startedAt;
  const endedAt = execution.stoppedAt;
  const durationMs =
    startedAt && endedAt
      ? Math.max(0, new Date(endedAt).getTime() - new Date(startedAt).getTime())
      : undefined;
  const executionError = execution.data?.resultData?.error;

  return {
    sessionId,
    threadId: options.threadId ?? resolveThreadId(execution, workflow),
    traceId,
    rootSpanId,
    agent: {
      name: options.agentName ?? workflow?.name ?? 'n8n workflow',
      provider: 'n8n',
    },
    config: {
      executionId,
      workflowId: execution.workflowId ?? workflow?.id,
      mode: execution.mode,
    },
    status: execution.status === 'success' ? 'success' : executionError ? 'error' : execution.status,
    startedAt,
    endedAt,
    durationMs,
    summary: {
      totalInputTokens,
      totalOutputTokens,
      ...(durationMs !== undefined ? { totalDurationMs: durationMs } : {}),
    },
    errors: executionError ? [executionError] : [],
    events,
  };
}

/** Which AI connection a task's payload belongs to, if any. */
function detectConnectionType(task: N8nTaskData): string | undefined {
  const buckets = { ...(task.inputOverride ?? {}), ...(task.data ?? {}) };
  for (const key of Object.keys(buckets)) {
    if (key.startsWith('ai_')) return key;
  }
  return undefined;
}

function buildActor(nodeName: string, connection: string | undefined): TraceEvent['actor'] {
  if (connection === AI_CONNECTIONS.tool) return { scope: 'tool', name: nodeName };
  if (connection === AI_CONNECTIONS.languageModel) return { scope: 'model', name: nodeName };
  return { scope: 'agent', name: nodeName };
}

function buildSections(task: N8nTaskData, connection: string | undefined): TraceSection[] {
  const sections: TraceSection[] = [];
  const inputBucket = connection ? task.inputOverride?.[connection] : task.inputOverride?.main;
  const outputBucket = connection ? task.data?.[connection] : task.data?.main;

  const input = firstJson(inputBucket);
  const output = firstJson(outputBucket);

  if (connection === AI_CONNECTIONS.languageModel) {
    // n8n's LLM tracing writes the rendered prompt as a string array.
    const messages = input?.messages;
    if (Array.isArray(messages)) {
      messages.forEach((message, index) => {
        sections.push({
          kind: 'message',
          label: `Prompt ${index + 1}`,
          role: 'user',
          content: message,
        });
      });
    } else if (input) {
      sections.push({ kind: 'message', label: 'Prompt', role: 'user', content: input });
    }

    const generations = (output?.response as Record<string, unknown> | undefined)?.generations;
    if (generations !== undefined) {
      sections.push({
        kind: 'message',
        label: 'Completion',
        role: 'assistant',
        content: generations,
      });
    } else if (output) {
      sections.push({ kind: 'message', label: 'Completion', role: 'assistant', content: output });
    }

    if (input?.options) {
      sections.push({ kind: 'metadata', label: 'Model settings', content: input.options });
    }
    return sections;
  }

  if (connection === AI_CONNECTIONS.tool) {
    if (input !== undefined) {
      sections.push({ kind: 'tool_call', label: 'Input', content: input });
    }
    if (output !== undefined) {
      sections.push({ kind: 'tool_result', label: 'Output', content: output });
    }
    return sections;
  }

  if (input !== undefined) sections.push({ kind: 'metadata', label: 'Input', content: input });
  if (output !== undefined) sections.push({ kind: 'metadata', label: 'Output', content: output });
  return sections;
}

function firstJson(bucket: Array<N8nNodeExecutionData[] | null> | undefined): Record<string, unknown> | undefined {
  const items = bucket?.[0];
  if (!items || items.length === 0) return undefined;
  const json = items[0]?.json;
  return json && typeof json === 'object' ? (json as Record<string, unknown>) : undefined;
}

interface ExtractedUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  /** n8n substitutes a tiktoken estimate when the provider reports nothing. */
  estimated: boolean;
}

function extractUsage(task: N8nTaskData, connection: string | undefined): ExtractedUsage {
  const output = connection ? firstJson(task.data?.[connection]) : undefined;
  const reported = asRecord(output?.tokenUsage);
  const estimate = asRecord(output?.tokenUsageEstimate);
  const source = reported ?? estimate;

  if (source) {
    return {
      inputTokens: num(source.promptTokens ?? source.inputTokens),
      outputTokens: num(source.completionTokens ?? source.outputTokens),
      totalTokens: num(source.totalTokens),
      estimated: !reported && Boolean(estimate),
    };
  }

  // `metadata.tokenUsage` is the newer, node-level mirror of the same numbers.
  const metadataUsage = task.metadata?.tokenUsage;
  if (metadataUsage) {
    return {
      inputTokens: num(metadataUsage.inputTokens),
      outputTokens: num(metadataUsage.outputTokens),
      estimated: false,
    };
  }

  return { estimated: false };
}

/** A specific run of a specific node — what a `parentSpanId` must address. */
interface ParentRun {
  node: string;
  runIndex: number;
}

/**
 * Map each sub-node run to the exact agent run that made it.
 *
 * n8n records this itself: an agent's task carries `metadata.subRun`, listing
 * the sub-node runs that belong to that agent run. Without it, a node that
 * executes more than once — a Loop Over Items or Split In Batches iteration —
 * would have every one of its model and tool calls re-parented onto its FIRST
 * run, mis-shaping the tree and the per-iteration token rollups.
 */
function buildSubRunParentMap(
  tasks: Array<{ nodeName: string; runIndex: number; task: N8nTaskData }>,
): Map<string, ParentRun> {
  const map = new Map<string, ParentRun>();
  for (const { nodeName, runIndex, task } of tasks) {
    for (const child of task.metadata?.subRun ?? []) {
      if (!child?.node || typeof child.runIndex !== 'number') continue;
      map.set(`${child.node}#${child.runIndex}`, { node: nodeName, runIndex });
    }
  }
  return map;
}

/**
 * Which run of which node produced this one.
 *
 * Three sources, most precise first. When none of them pins the parent's run
 * index — the parent ran several times and n8n told us nothing about which —
 * the event hangs off the session root instead: an honestly flat tree beats a
 * confidently wrong one.
 */
function resolveParentRun(
  nodeName: string,
  runIndex: number,
  task: N8nTaskData,
  parents: Map<string, string>,
  subRunParents: Map<string, ParentRun>,
  runCounts: Map<string, number>,
): ParentRun | undefined {
  const exact = subRunParents.get(`${nodeName}#${runIndex}`);
  if (exact) return exact;

  const parentNode = parents.get(nodeName) ?? task.source?.[0]?.previousNode;
  if (!parentNode) return undefined;

  // Main-chain links carry their own run index on newer n8n versions.
  const previousRun = task.source?.[0]?.previousNodeRun;
  if (typeof previousRun === 'number') return { node: parentNode, runIndex: previousRun };

  // Unambiguous when the parent ran exactly once, which is the common case.
  return (runCounts.get(parentNode) ?? 1) === 1
    ? { node: parentNode, runIndex: 0 }
    : undefined;
}

/**
 * Parent node for a sub-node.
 *
 * AI sub-nodes connect *outwards* to the node that consumes them, so the
 * agent is found by following the sub-node's own `ai_*` connection — the
 * opposite direction from a normal `main` connection.
 */
function buildParentMap(workflow: N8nWorkflow | undefined): Map<string, string> {
  const parents = new Map<string, string>();
  for (const [source, connections] of Object.entries(workflow?.connections ?? {})) {
    for (const [type, outputs] of Object.entries(connections)) {
      if (!type.startsWith('ai_')) continue;
      for (const output of outputs ?? []) {
        for (const target of output ?? []) {
          if (target?.node) parents.set(source, target.node);
        }
      }
    }
  }
  return parents;
}

/** True for the nodes that own a tool menu. */
function isAgentNode(nodeType: string | undefined): boolean {
  if (!nodeType) return false;
  return (
    nodeType.includes('agent') ||
    nodeType.endsWith('.chainLlm') ||
    nodeType.endsWith('.messageAnAgent')
  );
}

/**
 * Reconstruct the tool menu from the workflow's `ai_tool` nodes.
 *
 * This is the workflow's *declared* tool set, not the array the model was
 * handed on a given call — n8n does not record the latter. Callers label it
 * as such.
 */
function buildToolDefinitions(workflow: N8nWorkflow | undefined): TraceToolDefinition[] {
  const toolNodes = new Set<string>();
  for (const [source, connections] of Object.entries(workflow?.connections ?? {})) {
    if (Object.keys(connections).some((type) => type === AI_CONNECTIONS.tool)) {
      toolNodes.add(source);
    }
  }

  const definitions: TraceToolDefinition[] = [];
  for (const node of workflow?.nodes ?? []) {
    if (!node.name || !toolNodes.has(node.name)) continue;
    const parameters = node.parameters ?? {};
    const definition: TraceToolDefinition = { name: node.name };
    const description = parameters.toolDescription ?? parameters.description;
    if (typeof description === 'string' && description) definition.description = description;
    const schema =
      parseMaybeJson(parameters.inputSchema) ??
      parseMaybeJson(parameters.jsonSchemaExample) ??
      asRecord(parameters.workflowInputs);
    if (schema) definition.parameters = schema;
    definitions.push(definition);
  }
  return definitions;
}

/** Model name from the call's own options, else the node's configuration. */
function resolveModel(task: N8nTaskData, node: N8nWorkflowNode | undefined): string | undefined {
  const output = firstJson(task.data?.[AI_CONNECTIONS.languageModel]);
  const input = firstJson(task.inputOverride?.[AI_CONNECTIONS.languageModel]);
  const fromOptions = asRecord(input?.options)?.model ?? asRecord(output?.options)?.model;
  if (typeof fromOptions === 'string' && fromOptions) return fromOptions;

  const parameter = node?.parameters?.model ?? node?.parameters?.modelName;
  if (typeof parameter === 'string' && parameter) return parameter;
  // Resource-locator parameters wrap the value: `{ __rl: true, value: 'gpt-4.1' }`.
  const locatorValue = asRecord(parameter)?.value;
  return typeof locatorValue === 'string' && locatorValue ? locatorValue : undefined;
}

/**
 * Conversation id.
 *
 * n8n has no first-class one: `$execution.id` is per run. The closest thing is
 * the memory sub-node's `sessionKey`, which is usually an expression — so a
 * literal value is used when present, and callers who need something better
 * pass `threadId` explicitly.
 */
function resolveThreadId(
  execution: N8nExecution,
  workflow: N8nWorkflow | undefined,
): string | undefined {
  const custom = execution.customData;
  for (const key of ['threadId', 'sessionId', 'conversationId']) {
    const value = custom?.[key];
    if (typeof value === 'string' && value) return value;
  }
  for (const node of workflow?.nodes ?? []) {
    const sessionKey = node.parameters?.sessionKey;
    if (typeof sessionKey === 'string' && sessionKey && !sessionKey.includes('{{')) {
      return sessionKey;
    }
  }
  return undefined;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function parseMaybeJson(value: unknown): Record<string, unknown> | undefined {
  const record = asRecord(value);
  if (record) return record;
  if (typeof value === 'string' && value.trim().startsWith('{')) {
    try {
      return asRecord(JSON.parse(value));
    } catch {
      return undefined;
    }
  }
  return undefined;
}

/* ------------------------------------------------------------------ */
/*  Bridge — poll the n8n public API                                  */
/* ------------------------------------------------------------------ */

/** Ceiling on the de-duplication set — see `N8nBridge.remember`. */
const MAX_REMEMBERED_EXECUTIONS = 5_000;

export interface N8nBridgeOptions {
  /** n8n base URL, e.g. `https://n8n.acme.com`. */
  n8nUrl: string;
  /** n8n API key (Settings → n8n API). Sent as `X-N8N-API-KEY`. */
  n8nApiKey: string;
  /** Cognipeer client. Defaults to the process-wide one. */
  client?: CognipeerObservability;
  /** Only mirror executions of these workflow ids. Default: all. */
  workflowIds?: string[];
  /** Poll interval in ms (default 15 000). */
  intervalMs?: number;
  /** Executions fetched per poll (n8n caps at 250, default 50). */
  pageSize?: number;
  /** Ignore executions that finished before this instant. Default: now. */
  since?: Date;
  /** Custom fetch (tests, proxies). */
  fetch?: typeof fetch;
  /** Called for each mirrored execution — useful for logging progress. */
  onSession?: (payload: TraceSessionPayload) => void;
  /** Called on any bridge failure instead of throwing. */
  onError?: (error: Error) => void;
}

/**
 * Mirrors finished n8n executions into the Console.
 *
 * ```ts
 * const bridge = new N8nBridge({ n8nUrl, n8nApiKey });
 * await bridge.start();          // polls until stop()
 * ```
 *
 * Executions are de-duplicated by id, so restarting the bridge re-mirrors
 * nothing. Workflow JSON is fetched once per workflow and cached — that is
 * what supplies node types, model names and the tool menu.
 */
export class N8nBridge {
  private readonly options: Required<Pick<N8nBridgeOptions, 'intervalMs' | 'pageSize'>> &
    N8nBridgeOptions;
  private readonly seen = new Set<string>();
  private readonly workflows = new Map<string, N8nWorkflow>();
  private timer: ReturnType<typeof setTimeout> | undefined;
  /** Resolver for the in-progress inter-poll sleep, so `stop()` can cut it short. */
  private wake: (() => void) | undefined;
  private running = false;
  private cutoff: number;

  constructor(options: N8nBridgeOptions) {
    this.options = {
      intervalMs: 15_000,
      pageSize: 50,
      ...options,
      n8nUrl: options.n8nUrl.replace(/\/+$/, '').replace(/\/api\/v1$/, ''),
    };
    this.cutoff = (options.since ?? new Date()).getTime();
  }

  /** Poll once. Returns how many executions were mirrored. */
  async runOnce(): Promise<number> {
    let mirrored = 0;
    try {
      for (const execution of await this.fetchExecutions()) {
        const id = String(execution.id ?? '');
        if (!id || this.seen.has(id)) continue;

        // A still-running execution has no complete run data yet. Skipping it
        // WITHOUT marking it seen is the point: it will be picked up on a
        // later poll, once it has finished.
        if (!execution.stoppedAt) continue;
        if (new Date(execution.stoppedAt).getTime() < this.cutoff) {
          this.remember(id);
          continue;
        }

        const workflow = await this.loadWorkflow(execution.workflowId);
        const payload = mapN8nExecution(execution, { workflow });
        if (!payload) {
          // No run data at all — nothing will ever come of this one.
          this.remember(id);
          continue;
        }

        // Marked seen only after delivery, so a transport failure retries on
        // the next poll rather than losing the execution permanently.
        await this.ingest(payload);
        this.remember(id);
        this.options.onSession?.(payload);
        mirrored++;
      }
    } catch (error) {
      this.fail(error);
    }
    return mirrored;
  }

  /**
   * Remember a mirrored execution, bounded.
   *
   * n8n returns executions newest-first, so a bridge that has been up for
   * months only ever re-sees recent ids — an unbounded set would grow forever
   * for no benefit.
   */
  private remember(id: string): void {
    this.seen.add(id);
    if (this.seen.size <= MAX_REMEMBERED_EXECUTIONS) return;
    const excess = this.seen.size - MAX_REMEMBERED_EXECUTIONS;
    let dropped = 0;
    for (const seenId of this.seen) {
      if (dropped++ >= excess) break;
      this.seen.delete(seenId); // insertion-ordered: drops the oldest first
    }
  }

  /** Poll forever. Resolves once `stop()` is called. */
  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    while (this.running) {
      await this.runOnce();
      if (!this.running) break;
      await this.sleep();
    }
  }

  stop(): void {
    this.running = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    // Waking the sleeper is what makes `start()` resolve. Clearing the timer
    // alone cancels the very callback that would have settled the promise, so
    // a `stop()` landing mid-interval — the normal case at the 15 s default —
    // would leave `await bridge.start()` pending forever.
    this.wake?.();
    this.wake = undefined;
  }

  /** Wait out the poll interval, interruptibly. */
  private sleep(): Promise<void> {
    return new Promise<void>((resolve) => {
      this.wake = resolve;
      this.timer = setTimeout(resolve, this.options.intervalMs);
    });
  }

  private async ingest(payload: TraceSessionPayload): Promise<void> {
    // Executions arrive complete, so the batch endpoint is the right shape:
    // one request per run, and re-posting the same id is idempotent.
    await (this.options.client ?? getClient()).getTransport().ingestSession(payload);
  }

  private async fetchExecutions(): Promise<N8nExecution[]> {
    const params = new URLSearchParams({
      includeData: 'true',
      limit: String(this.options.pageSize),
    });
    const workflowIds = this.options.workflowIds ?? [];
    if (workflowIds.length === 1) params.set('workflowId', workflowIds[0]);

    const body = await this.request<{ data?: N8nExecution[] }>(`/api/v1/executions?${params}`);
    const executions = body.data ?? [];
    if (workflowIds.length > 1) {
      return executions.filter((execution) => workflowIds.includes(String(execution.workflowId)));
    }
    return executions;
  }

  private async loadWorkflow(workflowId: string | undefined): Promise<N8nWorkflow | undefined> {
    if (!workflowId) return undefined;
    const cached = this.workflows.get(workflowId);
    if (cached) return cached;
    try {
      const workflow = await this.request<N8nWorkflow>(
        `/api/v1/workflows/${encodeURIComponent(workflowId)}`,
      );
      this.workflows.set(workflowId, workflow);
      return workflow;
    } catch (error) {
      // A workflow the key cannot read still yields usable events, just
      // without node types, models or the tool menu.
      this.fail(error);
      return undefined;
    }
  }

  private async request<T>(path: string): Promise<T> {
    const fetchImpl = this.options.fetch ?? globalThis.fetch;
    const response = await fetchImpl(`${this.options.n8nUrl}${path}`, {
      headers: {
        'X-N8N-API-KEY': this.options.n8nApiKey,
        Accept: 'application/json',
      },
    });
    if (!response.ok) {
      throw new Error(`n8n API ${path} failed: ${response.status} ${await response.text()}`);
    }
    return (await response.json()) as T;
  }

  private fail(error: unknown): void {
    const wrapped = error instanceof Error ? error : new Error(String(error));
    if (this.options.onError) this.options.onError(wrapped);
    else console.warn('[cognipeer] n8n bridge:', wrapped.message);
  }
}

/* ------------------------------------------------------------------ */
/*  External hook — self-hosted, push-based                           */
/* ------------------------------------------------------------------ */

export interface N8nExternalHookOptions extends MapExecutionOptions {
  client?: CognipeerObservability;
  /** Skip manual/test runs. Default false, so a test run verifies the wiring. */
  productionOnly?: boolean;
}

/**
 * Build the object an `EXTERNAL_HOOK_FILES` module must export.
 *
 * ```js
 * // /opt/cognipeer/n8n-hook.js
 * const { init } = require('@cognipeer/observability');
 * const { createN8nExternalHook } = require('@cognipeer/observability/n8n');
 * init({ apiKey: process.env.COGNIPEER_API_KEY });
 * module.exports = createN8nExternalHook();
 * ```
 *
 * then `EXTERNAL_HOOK_FILES=/opt/cognipeer/n8n-hook.js`.
 */
export function createN8nExternalHook(options: N8nExternalHookOptions = {}) {
  return {
    workflow: {
      postExecute: [
        async function cognipeerPostExecute(
          fullRunData: { data?: N8nRunExecutionData; mode?: string; status?: string; startedAt?: Date | string; stoppedAt?: Date | string } | undefined,
          workflowData: N8nWorkflow,
          executionId: string,
        ): Promise<void> {
          if (!fullRunData) return;
          if (options.productionOnly && fullRunData.mode === 'manual') return;

          const payload = mapN8nExecution(
            {
              id: executionId,
              workflowId: workflowData?.id,
              status: fullRunData.status,
              mode: fullRunData.mode,
              startedAt: toIso(fullRunData.startedAt),
              stoppedAt: toIso(fullRunData.stoppedAt),
              data: fullRunData.data,
            },
            { ...options, workflow: workflowData },
          );
          if (!payload) return;

          // Hooks run inside n8n's own process: never let an export failure
          // surface as a workflow error.
          try {
            await (options.client ?? getClient()).getTransport().ingestSession(payload);
          } catch (error) {
            console.warn('[cognipeer] n8n hook export failed:', error);
          }
        },
      ],
    },
  };
}

function toIso(value: Date | string | undefined): string | undefined {
  if (!value) return undefined;
  return value instanceof Date ? value.toISOString() : String(value);
}
