/**
 * Semantic-convention normalizer.
 *
 * Three incompatible conventions describe the same LLM call, and a real
 * customer's traces will contain all of them — sometimes on one span:
 *
 *   - **OpenInference** (Arize/Phoenix) — `openinference.span.kind`, `llm.*`.
 *     Emitted by the 37 `openinference-instrumentation-*` packages, which is
 *     how CrewAI, LlamaIndex, smolagents, DSPy, Agno and friends get traced.
 *   - **OTel GenAI / OpenLLMetry >= 0.55** — `gen_ai.*` with the whole
 *     conversation packed into JSON string attributes. Emitted natively by
 *     Pydantic AI, Google ADK, AWS Strands, Semantic Kernel and the Microsoft
 *     Agent Framework.
 *   - **OpenLLMetry <= 0.54 (legacy)** — `gen_ai.prompt.N.role` style flat
 *     indexed attributes. OpenLLMetry changed conventions at 0.55.0 with no
 *     dual-emit and no deprecation window, and customers pin old versions for
 *     years, so both must be supported permanently.
 *
 * Everything here is pure: attributes in, one event out. The exporter in
 * `./index.ts` owns all the I/O.
 *
 * ────────────────────────── THE NORMALIZATION TABLE ──────────────────────────
 *
 * | our field           | OpenInference (Arize)                      | OTel GenAI / OpenLLMetry >=0.55        | OpenLLMetry <=0.54 (legacy)              |
 * |---------------------|--------------------------------------------|----------------------------------------|------------------------------------------|
 * | `type`              | `openinference.span.kind`                   | `gen_ai.operation.name`                | `traceloop.span.kind`, `llm.request.type`|
 * | `model`             | `llm.model_name`                            | `gen_ai.response.model` / `.request.model` | `gen_ai.request.model`               |
 * | `inputTokens`       | `llm.token_count.prompt`                    | `gen_ai.usage.input_tokens`            | `gen_ai.usage.prompt_tokens`             |
 * | `outputTokens`      | `llm.token_count.completion`                | `gen_ai.usage.output_tokens`           | `gen_ai.usage.completion_tokens`         |
 * | `cachedInputTokens` | `llm.token_count.prompt_details.cache_read` | `gen_ai.usage.cache_read.input_tokens` | `gen_ai.usage.cache_read_input_tokens`   |
 * | message sections    | `llm.input_messages.N.message.role/content` | `gen_ai.input.messages` (JSON `parts[]`) | `gen_ai.prompt.N.role/.content`        |
 * |                     | `llm.output_messages.N.message.*`           | `gen_ai.output.messages` + `gen_ai.system_instructions` | `gen_ai.completion.N.*`  |
 * | `toolDefinitions`   | `llm.tools.N.tool.json_schema`              | `gen_ai.tool.definitions` (JSON array) | `llm.request.functions.N.{name,…}`       |
 * | `responseFormat`    | `llm.invocation_parameters.response_format` | `gen_ai.output.type` + `gen_ai.request.structured_output_schema` | `llm.request.response_format` |
 * | tool_call sections  | `…message.tool_calls.M.tool_call.function.*`| `parts[type=tool_call]{id,name,arguments}` | `gen_ai.{prompt,completion}.N.tool_calls.M.*` |
 * | tool_result section | `output.value` on a TOOL span               | `parts[type=tool_call_response]`, `gen_ai.tool.call.result` | `traceloop.entity.output`|
 * | `toolName`          | `tool.name`                                 | `gen_ai.tool.name`                     | `traceloop.entity.name`                  |
 * | `label`             | `agent.name` / span name                    | `gen_ai.agent.name` / `gen_ai.workflow.name` | `traceloop.entity.name`            |
 * | thread id           | `session.id`                                | `gen_ai.conversation.id`               | `traceloop.association.properties.session_id` |
 * | user id             | `user.id`                                   | —                                      | `traceloop.association.properties.user_id`|
 * | `metadata`          | `metadata` (JSON), `tag.tags`               | `traceloop.association.properties.*`   | same                                     |
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

import type { TraceEvent, TraceResponseFormat, TraceSection, TraceToolDefinition } from '../core/types';

/** Flat OTel attribute bag. Values may be strings, numbers, booleans or arrays. */
export type Attributes = Record<string, unknown>;

/**
 * The convention(s) a span was written in. A span can genuinely carry two:
 * `OPENINFERENCE_ENABLE_GENAI_SEMCONV=true` makes OpenInference dual-emit.
 */
export type Convention = 'openinference' | 'genai' | 'openllmetry-legacy' | 'unknown';

/** A span reduced to what the normalizer needs. Built by the exporter. */
export interface OtelSpanLike {
  traceId: string;
  spanId: string;
  /** Absent or `''` for a root span — OTLP/JSON uses the empty string. */
  parentSpanId?: string;
  name: string;
  startTimeMs: number;
  endTimeMs: number;
  attributes: Attributes;
  /** OTel `StatusCode`: 0 UNSET, 1 OK, 2 ERROR. */
  status?: { code?: number; message?: string };
  events?: Array<{ name: string; attributes?: Attributes }>;
  /** Resource attributes — `service.name` is the agent-name fallback. */
  resource?: Attributes;
}

/**
 * One span, normalized.
 *
 * The wire event is nested rather than spread flat so that the session-level
 * hints — which are properties of the *conversation*, not of the event — can
 * never leak onto the ingest payload.
 */
export interface NormalizedSpan {
  /** Ready to hand to `TraceSession.record()`. */
  event: TraceEvent;
  /** Which conventions were recognised. Useful when debugging a customer. */
  conventions: Convention[];
  /** Conversation id, if the application injected one. */
  threadId?: string;
  /** Best available agent name for the session. */
  agentName?: string;
  /** End user, if the application injected one. */
  userId?: string;
  /** True when the span has no parent — the signal that a trace has finished. */
  isRoot: boolean;
}

/* ------------------------------------------------------------------ */
/*  Attribute keys                                                    */
/* ------------------------------------------------------------------ */

/**
 * OpenInference keys, inlined rather than imported from
 * `@arizeai/openinference-semantic-conventions`. The package would be a hard
 * dependency on a receiver that must also work when the customer runs
 * OpenLLMetry and has never heard of Arize.
 */
const OI = {
  SPAN_KIND: 'openinference.span.kind',
  MODEL_NAME: 'llm.model_name',
  PROVIDER: 'llm.provider',
  SYSTEM: 'llm.system',
  TOKEN_PROMPT: 'llm.token_count.prompt',
  TOKEN_COMPLETION: 'llm.token_count.completion',
  TOKEN_TOTAL: 'llm.token_count.total',
  TOKEN_CACHE_READ: 'llm.token_count.prompt_details.cache_read',
  INPUT_MESSAGES: 'llm.input_messages',
  OUTPUT_MESSAGES: 'llm.output_messages',
  TOOLS: 'llm.tools',
  INVOCATION_PARAMETERS: 'llm.invocation_parameters',
  TOOL_NAME: 'tool.name',
  TOOL_DESCRIPTION: 'tool.description',
  TOOL_PARAMETERS: 'tool.parameters',
  TOOL_CALL_ID: 'tool_call.id',
  INPUT_VALUE: 'input.value',
  OUTPUT_VALUE: 'output.value',
  SESSION_ID: 'session.id',
  USER_ID: 'user.id',
  METADATA: 'metadata',
  TAGS: 'tag.tags',
  AGENT_NAME: 'agent.name',
  RETRIEVAL_DOCUMENTS: 'retrieval.documents',
  EMBEDDING_MODEL_NAME: 'embedding.model_name',
  EMBEDDING_EMBEDDINGS: 'embedding.embeddings',
  GRAPH_NODE_ID: 'graph.node.id',
  GRAPH_NODE_PARENT_ID: 'graph.node.parent_id',
  COST_TOTAL: 'llm.cost.total',
} as const;

/**
 * The literal a redacting instrumentation substitutes for hidden content.
 * It is a string, not null and not absent, so it has to be filtered explicitly
 * or it ends up rendered as a prompt and — worse — harvested into eval
 * datasets as if it were one.
 */
export const REDACTION_SENTINEL = '__REDACTED__';

/** Our canonical event types, so the mapping tables stay honest. */
type EventType = TraceEvent['type'];

/* ------------------------------------------------------------------ */
/*  Primitive readers                                                 */
/* ------------------------------------------------------------------ */

/** First key that holds a usable string. Redaction sentinels do not count. */
function str(attributes: Attributes, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = attributes[key];
    if (typeof value === 'string' && value.length > 0 && value !== REDACTION_SENTINEL) {
      return value;
    }
  }
  return undefined;
}

/** First key that holds a number. OTLP/JSON renders int attributes as strings. */
function num(attributes: Attributes, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = attributes[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return undefined;
}

/**
 * Parse a JSON string attribute, or fall back.
 *
 * OTel truncates attribute values at `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT`, and
 * OpenLLMetry >= 0.55 packs an entire conversation into ONE attribute — so a
 * truncation destroys the whole conversation's JSON rather than one message.
 * A parse failure must therefore degrade to raw-text capture, never drop the
 * event.
 */
function parseJson<T>(raw: unknown, fallback: T): T {
  if (typeof raw !== 'string') return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

/** True when the value is a JSON string that did not parse — i.e. truncated. */
function isUnparseableJson(raw: unknown): boolean {
  if (typeof raw !== 'string' || raw.length === 0) return false;
  const trimmed = raw.trimStart();
  if (trimmed[0] !== '{' && trimmed[0] !== '[') return false;
  try {
    JSON.parse(raw);
    return false;
  } catch {
    return true;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Group flat indexed attributes (`prefix.0.field`) by index.
 *
 * These arrive in arbitrary order — `gen_ai.prompt.7.content` can precede
 * `gen_ai.prompt.0.role` — so accumulating sequentially scrambles the
 * conversation. Group by the numeric index, then sort by it.
 */
export function groupIndexed(attributes: Attributes, prefix: string): Map<number, Attributes> {
  const grouped = new Map<number, Attributes>();
  const pattern = new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\.(\\d+)\\.(.+)$`);

  for (const [key, value] of Object.entries(attributes)) {
    const match = key.match(pattern);
    if (!match) continue;
    const index = Number(match[1]);
    if (!Number.isFinite(index)) continue;
    let bucket = grouped.get(index);
    if (!bucket) {
      bucket = {};
      grouped.set(index, bucket);
    }
    bucket[match[2]] = value;
  }

  return new Map([...grouped.entries()].sort(([a], [b]) => a - b));
}

/* ------------------------------------------------------------------ */
/*  Convention detection                                              */
/* ------------------------------------------------------------------ */

/**
 * Which convention(s) a span speaks, decided purely by attribute presence.
 *
 * Never by a version header: OpenLLMetry's 0.55 break is invisible from
 * outside, spans can be enriched in flight, and one span may carry two
 * conventions at once.
 */
export function detectConventions(attributes: Attributes): Convention[] {
  const found: Convention[] = [];

  if (attributes[OI.SPAN_KIND] != null || attributes[OI.MODEL_NAME] != null) {
    found.push('openinference');
  }
  if (
    attributes['gen_ai.input.messages'] != null ||
    attributes['gen_ai.output.messages'] != null ||
    attributes['gen_ai.system_instructions'] != null ||
    attributes['gen_ai.tool.definitions'] != null ||
    attributes['gen_ai.usage.input_tokens'] != null ||
    attributes['gen_ai.operation.name'] != null
  ) {
    found.push('genai');
  }

  const keys = Object.keys(attributes);
  if (
    keys.some((key) => /^gen_ai\.(prompt|completion)\.\d+\./.test(key)) ||
    keys.some((key) => /^llm\.request\.functions\.\d+\./.test(key)) ||
    attributes['traceloop.span.kind'] != null
  ) {
    found.push('openllmetry-legacy');
  }

  return found.length > 0 ? found : ['unknown'];
}

/* ------------------------------------------------------------------ */
/*  Event type                                                        */
/* ------------------------------------------------------------------ */

/** OpenInference span kind → our event type. */
const TYPE_BY_OI_KIND: Record<string, EventType> = {
  LLM: 'ai_call',
  TOOL: 'tool_call',
  EMBEDDING: 'embedding',
  RETRIEVER: 'retrieval',
  // Reranking is a stage of the retrieval pipeline; the Console has no
  // separate bucket for it, and the raw kind stays in `metadata`.
  RERANKER: 'retrieval',
  GUARDRAIL: 'guardrail',
  AGENT: 'span',
  CHAIN: 'span',
  EVALUATOR: 'span',
  UNKNOWN: 'span',
};

/** OTel GenAI `gen_ai.operation.name` → our event type. */
const TYPE_BY_GENAI_OPERATION: Record<string, EventType> = {
  chat: 'ai_call',
  text_completion: 'ai_call',
  generate_content: 'ai_call',
  execute_tool: 'tool_call',
  embeddings: 'embedding',
  invoke_agent: 'span',
  create_agent: 'span',
};

/**
 * Resolve the event type, best convention first.
 *
 * This is the load-bearing decision of the whole file: only an `ai_call`
 * carries token counts (see `normalizeSpan`), so a mis-typed span either
 * loses its usage or double-counts a parent's aggregate.
 */
export function normalizeType(attributes: Attributes, spanName: string): EventType {
  const openInferenceKind = str(attributes, OI.SPAN_KIND);
  if (openInferenceKind) {
    return TYPE_BY_OI_KIND[openInferenceKind.toUpperCase()] ?? 'span';
  }

  const operation = str(attributes, 'gen_ai.operation.name');
  if (operation) {
    const mapped = TYPE_BY_GENAI_OPERATION[operation];
    if (mapped) return mapped;
    // Vendors ship unlisted variants — `chat.stream`, `generate_content_v2`.
    if (operation.startsWith('chat') || operation.startsWith('generate')) return 'ai_call';
    // Otherwise fall through rather than settling for 'span': the signals
    // below may still identify the call, and the type gates token attribution.
  }

  const traceloopKind = str(attributes, 'traceloop.span.kind');
  if (traceloopKind && traceloopKind !== 'unknown') {
    return traceloopKind === 'tool' ? 'tool_call' : 'span';
  }

  const requestType = str(attributes, 'llm.request.type');
  if (requestType) {
    if (requestType === 'chat' || requestType === 'completion') return 'ai_call';
    if (requestType === 'embedding') return 'embedding';
    if (requestType === 'rerank') return 'retrieval';
  }

  // Nothing declared a kind. Two independent signals say "model call", and
  // BOTH are honoured — they must, because usage is only attached to a
  // model-call event, so a mistype here silently discards tokens the span
  // explicitly reported. Kept identical to the Python normalizer.
  if (str(attributes, OI.MODEL_NAME, 'gen_ai.request.model', 'gen_ai.response.model')) {
    return 'ai_call';
  }
  if (num(attributes, 'gen_ai.usage.input_tokens', 'gen_ai.usage.prompt_tokens') !== undefined) {
    return 'ai_call';
  }
  return spanName.toLowerCase().includes('tool') ? 'tool_call' : 'span';
}

/* ------------------------------------------------------------------ */
/*  Tool definitions                                                  */
/* ------------------------------------------------------------------ */

/**
 * The tool menu offered to the model on this call.
 *
 * Three shapes with three different nesting depths — unwrapping one with
 * another's logic silently yields nameless entries:
 *  - OpenInference `llm.tools.N.tool.json_schema` — each value is the WHOLE
 *    tool object JSON-encoded, usually `{type:'function', function:{…}}`;
 *  - GenAI `gen_ai.tool.definitions` — ONE JSON array of already-unwrapped
 *    `{name, description, parameters}` dicts;
 *  - legacy `llm.request.functions.N.{name,description,parameters}` — flat and
 *    indexed, with `parameters` itself a JSON string.
 */
export function extractToolDefinitions(attributes: Attributes): TraceToolDefinition[] {
  const openInference = groupIndexed(attributes, OI.TOOLS);
  if (openInference.size > 0) {
    const definitions = [...openInference.values()]
      .map((tool) => {
        const parsed = parseJson<unknown>(tool['tool.json_schema'], null);
        // The extra `function` wrapper is present on the OpenAI envelope and
        // absent on bare schemas.
        const unwrapped = isRecord(parsed) && isRecord(parsed.function) ? parsed.function : parsed;
        return toToolDefinition(unwrapped);
      })
      .filter((tool): tool is TraceToolDefinition => tool !== undefined);
    if (definitions.length > 0) return definitions;
  }

  const genai = parseJson<unknown[]>(attributes['gen_ai.tool.definitions'], []);
  if (Array.isArray(genai) && genai.length > 0) {
    const definitions = genai
      .map((entry) => toToolDefinition(entry))
      .filter((tool): tool is TraceToolDefinition => tool !== undefined);
    if (definitions.length > 0) return definitions;
  }

  const legacy = groupIndexed(attributes, 'llm.request.functions');
  return [...legacy.values()]
    .map((fn) => toToolDefinition(fn))
    .filter((tool): tool is TraceToolDefinition => tool !== undefined);
}

/**
 * Extract the structured-output contract enforced on a model call.
 *
 * Three places it hides, in order of fidelity:
 *   - OpenInference stashes the whole request body in
 *     `llm.invocation_parameters`, `response_format` included verbatim;
 *   - some emitters attach the body directly as `llm.request.response_format`;
 *   - OTel GenAI splits it into the output MODE (`gen_ai.output.type`) and the
 *     schema (`gen_ai.request.structured_output_schema`).
 *
 * Returns undefined when a span says nothing about the contract — which is not
 * the same as "the call was unconstrained", and the Console shows the
 * difference by simply having no section.
 */
export function extractResponseFormat(attributes: Attributes): TraceResponseFormat | undefined {
  const invocation = parseJson<unknown>(attributes[OI.INVOCATION_PARAMETERS], null);
  const fromInvocation = isRecord(invocation)
    ? (invocation.response_format ?? invocation.responseFormat)
    : undefined;
  const direct =
    fromInvocation
    ?? parseJson<unknown>(attributes['llm.request.response_format'], null)
    ?? parseJson<unknown>(attributes['gen_ai.request.response_format'], null);
  const format = toResponseFormat(direct);
  if (format) return format;

  const schema = parseJson<unknown>(attributes['gen_ai.request.structured_output_schema'], null);
  if (isRecord(schema)) {
    const name = attributes['gen_ai.request.structured_output_name'];
    return {
      type: 'json_schema',
      ...(typeof name === 'string' && name.length > 0 ? { name } : {}),
      schema,
    };
  }

  // `json` with no schema is still a contract — the model was told to emit JSON.
  return attributes['gen_ai.output.type'] === 'json' ? { type: 'json_object' } : undefined;
}

/** Coerce an OpenAI-shaped `response_format` body into our flat form. */
function toResponseFormat(value: unknown): TraceResponseFormat | undefined {
  if (!isRecord(value)) return undefined;
  const type = typeof value.type === 'string' && value.type.length > 0 ? value.type : undefined;
  const jsonSchema = isRecord(value.json_schema)
    ? value.json_schema
    : (isRecord(value.jsonSchema) ? value.jsonSchema : undefined);
  const source = jsonSchema ?? value;
  const schema = isRecord(source.schema) ? source.schema : undefined;
  if (!type && !schema) return undefined;

  return {
    type: type ?? 'json_schema',
    ...(typeof source.name === 'string' && source.name.length > 0 ? { name: source.name } : {}),
    ...(typeof source.strict === 'boolean' ? { strict: source.strict } : {}),
    ...(schema ? { schema } : {}),
  };
}

/** Coerce any of the three shapes into our `{name, description?, parameters?}`. */
function toToolDefinition(value: unknown): TraceToolDefinition | undefined {
  if (!isRecord(value)) return undefined;
  const source = isRecord(value.function) ? value.function : value;
  const name = source.name;
  if (typeof name !== 'string' || name.length === 0) return undefined;

  const definition: TraceToolDefinition = { name };
  const description = source.description;
  if (typeof description === 'string' && description.length > 0) {
    definition.description = description;
  }

  // `parameters` is an object in the GenAI array and a JSON string in the
  // legacy indexed form; `input_schema` is the Anthropic-flavoured alias.
  const rawParameters = source.parameters ?? source.input_schema ?? source.inputSchema;
  const parameters = isRecord(rawParameters)
    ? rawParameters
    : parseJson<Record<string, unknown> | null>(rawParameters, null);
  if (isRecord(parameters)) definition.parameters = parameters;

  return definition;
}

/* ------------------------------------------------------------------ */
/*  Sections                                                          */
/* ------------------------------------------------------------------ */

function messageLabel(role: string | undefined): string {
  const normalized = (role ?? 'message').trim();
  return `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)} message`;
}

function pushMessage(
  sections: TraceSection[],
  role: string | undefined,
  content: unknown,
): void {
  if (content === undefined || content === null || content === '') return;
  if (content === REDACTION_SENTINEL) return;
  sections.push({ kind: 'message', label: messageLabel(role), role, content });
}

/**
 * Every renderable block on the span.
 *
 * The three message conventions are tried in PRECEDENCE order, richest first —
 * not concatenated. With `OPENINFERENCE_ENABLE_GENAI_SEMCONV=true` one span
 * carries two namespaces describing the same conversation, and appending both
 * renders every message twice, which reads as a doubled prompt. The Python
 * normalizer resolves it the same way; the two must agree, since the same run
 * traced from either SDK has to look identical.
 */
export function extractSections(attributes: Attributes): TraceSection[] {
  const sections: TraceSection[] =
    openInferenceMessages(attributes).length > 0
      ? openInferenceMessages(attributes)
      : genAiMessages(attributes).length > 0
        ? genAiMessages(attributes)
        : legacyMessages(attributes);

  return [...sections, ...singleValueSections(attributes, sections)];
}

/** OpenInference indexed message attributes. */
function openInferenceMessages(attributes: Attributes): TraceSection[] {
  const sections: TraceSection[] = [];
  for (const prefix of [OI.INPUT_MESSAGES, OI.OUTPUT_MESSAGES]) {
    for (const message of groupIndexed(attributes, prefix).values()) {
      const role = typeof message['message.role'] === 'string'
        ? (message['message.role'] as string)
        : undefined;
      pushMessage(sections, role, message['message.content']);

      for (const call of groupIndexed(message, 'message.tool_calls').values()) {
        const name = call['tool_call.function.name'];
        sections.push({
          kind: 'tool_call',
          label: `Tool call: ${typeof name === 'string' ? name : 'tool'}`,
          ...(typeof name === 'string' ? { tool: name } : {}),
          // Arguments are a JSON *string* here, unlike the GenAI parts form.
          content: call['tool_call.function.arguments'],
          ...(typeof call['tool_call.id'] === 'string'
            ? { toolCallId: call['tool_call.id'] }
            : {}),
        });
      }
    }
  }
  return sections;
}

/** OTel GenAI / OpenLLMetry >= 0.55: JSON blobs of `{role, parts[]}`. */
function genAiMessages(attributes: Attributes): TraceSection[] {
  const sections: TraceSection[] = [];
  //
  // The system prompt lives in its own attribute from 0.55 onwards. Reading
  // only `input.messages` silently loses it — and the system prompt is
  // routinely the largest single block of tokens in an agent, so losing it
  // corrupts cost attribution rather than merely hiding a message.
  for (const instruction of parseJson<unknown[]>(attributes['gen_ai.system_instructions'], [])) {
    if (isRecord(instruction)) {
      pushMessage(sections, 'system', instruction.content);
    } else if (typeof instruction === 'string') {
      pushMessage(sections, 'system', instruction);
    }
  }

  for (const key of ['gen_ai.input.messages', 'gen_ai.output.messages']) {
    const messages = parseJson<unknown[]>(attributes[key], []);
    if (!Array.isArray(messages)) continue;
    for (const message of messages) {
      if (!isRecord(message)) continue;
      const role = typeof message.role === 'string' ? message.role : undefined;
      const parts = Array.isArray(message.parts) ? message.parts : [];

      // Some emitters still put a plain `content` alongside `parts`.
      if (parts.length === 0) pushMessage(sections, role, message.content);

      for (const part of parts) {
        if (!isRecord(part)) continue;
        switch (part.type) {
          case 'text':
          case 'reasoning':
          case 'refusal':
            pushMessage(sections, role, part.content);
            break;
          case 'tool_call': {
            const name = typeof part.name === 'string' ? part.name : undefined;
            sections.push({
              kind: 'tool_call',
              label: `Tool call: ${name ?? 'tool'}`,
              ...(name ? { tool: name } : {}),
              // Arguments arrive PARSED here — the opposite of OpenInference.
              content: part.arguments,
              ...(typeof part.id === 'string' ? { toolCallId: part.id } : {}),
            });
            break;
          }
          case 'tool_call_response':
            sections.push({
              kind: 'tool_result',
              label: 'Tool result',
              role: 'tool',
              content: part.response,
              ...(typeof part.id === 'string' ? { toolCallId: part.id } : {}),
            });
            break;
          default:
            break;
        }
      }
    }
  }
  return sections;
}

/** OpenLLMetry <= 0.54: flat indexed prompt/completion. */
function legacyMessages(attributes: Attributes): TraceSection[] {
  const sections: TraceSection[] = [];
  for (const prefix of ['gen_ai.prompt', 'gen_ai.completion']) {
    for (const message of groupIndexed(attributes, prefix).values()) {
      const role = typeof message.role === 'string' ? (message.role as string) : undefined;
      pushMessage(sections, role, message.content);

      for (const call of groupIndexed(message, 'tool_calls').values()) {
        const name = call.name;
        sections.push({
          kind: 'tool_call',
          label: `Tool call: ${typeof name === 'string' ? name : 'tool'}`,
          ...(typeof name === 'string' ? { tool: name } : {}),
          content: call.arguments,
          ...(typeof call.id === 'string' ? { toolCallId: call.id } : {}),
        });
      }
    }
  }
  return sections;
}

/**
 * Single-value fallbacks: TOOL spans and traceloop entities.
 *
 * `messages` is what the message conventions already produced, so a fallback
 * only fires when it would not duplicate them.
 */
function singleValueSections(attributes: Attributes, messages: TraceSection[]): TraceSection[] {
  const sections: TraceSection[] = [];
  //
  // A TOOL span has no message list at all: its arguments and its result are
  // the whole payload, carried as `input.value` / `output.value` (OpenInference)
  // or `traceloop.entity.input` / `.output`. Those must be read as a tool call
  // rather than as a message, and the input must not be conditional on the
  // output being absent — a tool span almost always has both.
  const isToolSpan =
    str(attributes, OI.SPAN_KIND)?.toUpperCase() === 'TOOL' ||
    str(attributes, 'gen_ai.operation.name') === 'execute_tool' ||
    str(attributes, 'traceloop.span.kind') === 'tool' ||
    attributes['gen_ai.tool.name'] != null;

  const genericInput = str(attributes, OI.INPUT_VALUE, 'traceloop.entity.input');
  // A span with content disabled (`TRACELOOP_TRACE_CONTENT=false`, or any of
  // the OPENINFERENCE_HIDE_* switches) is perfectly valid — it still carries
  // tokens and latency — so this is a fallback, not a requirement.
  if (genericInput && !messages.some((section) => section.kind === 'tool_call')) {
    if (isToolSpan) {
      sections.push({ kind: 'tool_call', label: 'Input', content: genericInput });
    } else if (!messages.some((section) => section.kind === 'message')) {
      pushMessage(sections, 'user', genericInput);
    }
  }

  const toolResult = str(
    attributes,
    'gen_ai.tool.call.result',
    'traceloop.entity.output',
    OI.OUTPUT_VALUE,
  );
  if (toolResult && !messages.some((section) => section.kind === 'tool_result')) {
    sections.push({
      kind: 'tool_result',
      label: isToolSpan ? 'Tool result' : 'Output',
      ...(isToolSpan ? { role: 'tool' } : {}),
      content: toolResult,
    });
  }

  sections.push(...extractRetrievalSections(attributes));
  sections.push(...truncationNotices(attributes));

  return sections;
}

/** Retrieved documents, which OpenInference emits as indexed attributes. */
function extractRetrievalSections(attributes: Attributes): TraceSection[] {
  const documents = groupIndexed(attributes, OI.RETRIEVAL_DOCUMENTS);
  if (documents.size === 0) return [];

  return [
    {
      kind: 'metadata',
      label: `Documents (${documents.size})`,
      content: [...documents.values()].map((document) => ({
        id: document['document.id'],
        score: document['document.score'],
        content: document['document.content'],
        metadata: parseJson<unknown>(document['document.metadata'], document['document.metadata']),
      })),
    },
  ];
}

/**
 * Preserve the raw text of an attribute whose JSON did not parse.
 *
 * This is what "degrade, do not drop" means in practice: a truncated
 * conversation still reaches the Console as readable text, flagged, instead of
 * vanishing.
 */
function truncationNotices(attributes: Attributes): TraceSection[] {
  const notices: TraceSection[] = [];
  for (const key of [
    'gen_ai.input.messages',
    'gen_ai.output.messages',
    'gen_ai.system_instructions',
    'gen_ai.tool.definitions',
  ]) {
    if (!isUnparseableJson(attributes[key])) continue;
    notices.push({
      kind: 'metadata',
      label: `${key} (truncated)`,
      content: attributes[key],
      truncated: true,
    });
  }
  return notices;
}

/* ------------------------------------------------------------------ */
/*  Session-level identity                                            */
/* ------------------------------------------------------------------ */

/**
 * Conversation id.
 *
 * No instrumentation emits this by itself — every branch requires the
 * application to opt in (`using_session()` / `setSession()` in OpenInference,
 * `set_association_properties()` / `set_conversation_id()` in OpenLLMetry).
 * Without it every trace arrives as its own orphan session, which is exactly
 * what the guide tells customers to avoid.
 */
export function extractThreadId(attributes: Attributes): string | undefined {
  return str(
    attributes,
    OI.SESSION_ID,
    'gen_ai.conversation.id',
    'traceloop.association.properties.session_id',
    'traceloop.association.properties.thread_id',
    'traceloop.association.properties.conversation_id',
  );
}

/** Agent name, best source first, falling back to the resource service name. */
export function extractAgentName(
  attributes: Attributes,
  resource: Attributes = {},
): string | undefined {
  return (
    str(attributes, 'gen_ai.agent.name', OI.AGENT_NAME, 'traceloop.workflow.name', 'gen_ai.workflow.name') ??
    str(resource, 'service.name')
  );
}

/* ------------------------------------------------------------------ */
/*  Metadata                                                          */
/* ------------------------------------------------------------------ */

/** Everything worth keeping that is not a first-class field. */
function extractMetadata(span: OtelSpanLike, conventions: Convention[]): Record<string, unknown> {
  const attributes = span.attributes;
  const metadata: Record<string, unknown> = { conventions };

  const application = parseJson<unknown>(attributes[OI.METADATA], attributes[OI.METADATA]);
  if (application !== undefined && application !== null) metadata.metadata = application;

  const tags = attributes[OI.TAGS];
  if (tags !== undefined) metadata.tags = tags;

  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith('traceloop.association.properties.')) {
      metadata[key.slice('traceloop.association.properties.'.length)] = value;
    }
  }

  const passthrough: Array<[string, string]> = [
    ['spanKind', OI.SPAN_KIND],
    ['provider', OI.PROVIDER],
    ['system', 'gen_ai.system'],
    ['operation', 'gen_ai.operation.name'],
    ['requestModel', 'gen_ai.request.model'],
    ['responseModel', 'gen_ai.response.model'],
    ['finishReasons', 'gen_ai.response.finish_reasons'],
    ['invocationParameters', OI.INVOCATION_PARAMETERS],
    // A separate LOGICAL hierarchy from OTel parentage: useful for agent-graph
    // visualisation, never for building the span tree.
    ['graphNodeId', OI.GRAPH_NODE_ID],
    ['graphNodeParentId', OI.GRAPH_NODE_PARENT_ID],
  ];
  for (const [name, key] of passthrough) {
    const value = attributes[key];
    if (value !== undefined) metadata[name] = value;
  }

  // OpenInference is the only convention that reports cost. Ours is computed
  // Console-side from tokens, so this is kept for reconciliation only.
  const cost = num(attributes, OI.COST_TOTAL);
  if (cost !== undefined) metadata.reportedCostUsd = cost;

  return metadata;
}

/* ------------------------------------------------------------------ */
/*  The event                                                         */
/* ------------------------------------------------------------------ */

/** Turn one span into one Console event, plus the session hints it carries. */
export function normalizeSpan(span: OtelSpanLike): NormalizedSpan {
  const attributes = span.attributes ?? {};
  const resource = span.resource ?? {};
  const conventions = detectConventions(attributes);
  const type = normalizeType(attributes, span.name);
  const isError = span.status?.code === 2;

  const event: TraceEvent = {
    id: `${span.traceId}:${span.spanId}`,
    type,
    label:
      str(attributes, 'gen_ai.agent.name', OI.AGENT_NAME, 'traceloop.entity.name', OI.TOOL_NAME) ??
      span.name,
    traceId: span.traceId,
    spanId: span.spanId,
    // `''` is a root in OTLP/JSON; the exporter re-parents those onto the
    // session root so the tree has a single trunk.
    parentSpanId: span.parentSpanId || undefined,
    timestamp: new Date(span.startTimeMs).toISOString(),
    durationMs: Math.max(0, Math.round(span.endTimeMs - span.startTimeMs)),
    status: isError ? 'error' : 'success',
    sections: extractSections(attributes),
    metadata: extractMetadata(span, conventions),
  };

  const model = str(
    attributes,
    OI.MODEL_NAME,
    'gen_ai.response.model',
    'gen_ai.request.model',
    OI.EMBEDDING_MODEL_NAME,
  );
  if (model) event.model = model;

  // Tokens ONLY on model calls. Three sources double-count otherwise: a
  // framework instrumentation and a client instrumentation both wrapping the
  // same call, an agent-run span repeating its children's aggregate, and
  // OpenInference dual-emit. Restricting usage to `ai_call` leaves exactly one
  // span per real call carrying it.
  if (type === 'ai_call' || type === 'embedding') {
    const inputTokens = num(
      attributes,
      OI.TOKEN_PROMPT,
      'gen_ai.usage.input_tokens',
      'gen_ai.usage.prompt_tokens',
    );
    const outputTokens = num(
      attributes,
      OI.TOKEN_COMPLETION,
      'gen_ai.usage.output_tokens',
      'gen_ai.usage.completion_tokens',
    );
    const totalTokens = num(attributes, OI.TOKEN_TOTAL, 'gen_ai.usage.total_tokens');
    // cache_read only. `cache_write` / `cache_creation` is a different number
    // billed at a different rate, and the Console treats cached tokens as a
    // SUBSET of the input count — folding writes in here would misprice.
    const cachedInputTokens = num(
      attributes,
      OI.TOKEN_CACHE_READ,
      'gen_ai.usage.cache_read.input_tokens',
      'gen_ai.usage.cache_read_input_tokens',
    );

    // Absent, never zero: a streaming call without `include_usage` reports
    // nothing, and a zero would silently under-report spend.
    if (inputTokens !== undefined) event.inputTokens = inputTokens;
    if (outputTokens !== undefined) event.outputTokens = outputTokens;
    if (totalTokens !== undefined) event.totalTokens = totalTokens;
    if (cachedInputTokens !== undefined) event.cachedInputTokens = cachedInputTokens;
  }

  if (type === 'ai_call') {
    const toolDefinitions = extractToolDefinitions(attributes);
    if (toolDefinitions.length > 0) event.toolDefinitions = toolDefinitions;
    const responseFormat = extractResponseFormat(attributes);
    if (responseFormat) event.responseFormat = responseFormat;
  }

  if (type === 'tool_call') {
    const toolName = str(attributes, OI.TOOL_NAME, 'gen_ai.tool.name', 'traceloop.entity.name');
    if (toolName) {
      event.toolName = toolName;
      event.actor = { scope: 'tool', name: toolName };
    }
    const toolCallId = str(attributes, 'gen_ai.tool.call.id', OI.TOOL_CALL_ID);
    if (toolCallId) event.toolExecutionId = toolCallId;
  } else if (type === 'ai_call') {
    event.actor = { scope: 'model', name: model ?? 'model' };
  } else {
    const agentName = str(attributes, 'gen_ai.agent.name', OI.AGENT_NAME);
    if (agentName) event.actor = { scope: 'agent', name: agentName };
  }

  if (isError) {
    event.error =
      span.status?.message ??
      str(attributes, 'exception.message') ??
      exceptionFromEvents(span) ??
      'Span reported an error';
  }

  return {
    event,
    conventions,
    threadId: extractThreadId(attributes),
    agentName: extractAgentName(attributes, resource),
    userId: str(attributes, OI.USER_ID, 'traceloop.association.properties.user_id'),
    isRoot: !span.parentSpanId,
  };
}

/** OTel records thrown errors as a span event rather than an attribute. */
function exceptionFromEvents(span: OtelSpanLike): string | undefined {
  for (const event of span.events ?? []) {
    if (event.name !== 'exception') continue;
    const message = str(event.attributes ?? {}, 'exception.message', 'exception.type');
    if (message) return message;
  }
  return undefined;
}
