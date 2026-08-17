/**
 * Integration and OpenTelemetry regressions.
 *
 * Every test here pins a defect that an adversarial review found and an
 * independent reviewer reproduced. Where a behaviour exists to match the Python
 * SDK, the test says so — the same run traced from either language has to
 * produce the same event.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CognipeerObservability, resetClient } from '../src/core/client';
import { spanIdFrom } from '../src/core/ids';
import type { TraceSessionPayload } from '../src/core/types';
import { CognipeerCallbackHandler } from '../src/integrations/langchain';
import { mapN8nExecution, N8nBridge, type N8nExecution } from '../src/integrations/n8n';
import { extractSections, normalizeType } from '../src/otel/normalize';
import { CognipeerAISDKTelemetry, withCognipeerTracing } from '../src/integrations/vercel-ai';

interface Sent {
  path: string;
  body: Record<string, unknown>;
}

function testClient(overrides: Record<string, unknown> = {}) {
  const sent: Sent[] = [];
  const client = new CognipeerObservability({
    apiKey: 'cpeer_test_key_000000000000',
    baseUrl: 'https://console.test',
    mode: 'batch',
    fetch: (async (url: string, options: { body: string }) => {
      sent.push({ path: new URL(url).pathname, body: JSON.parse(options.body) });
      return { ok: true, status: 200, text: async () => '' } as Response;
    }) as unknown as typeof fetch,
    ...overrides,
  });
  return { client, sent };
}

/** One `invoke()`-shaped run through the callback surface. */
async function runChain(handler: CognipeerCallbackHandler, suffix: string): Promise<void> {
  handler.handleChainStart({} as never, {}, `root-${suffix}`, undefined, [], {}, undefined, 'Agent');
  handler.handleChainStart({} as never, {}, `noise-${suffix}`, `root-${suffix}`, [], {}, undefined, 'ChannelWrite');
  await handler.handleChainEnd({}, `noise-${suffix}`);
  await handler.handleChainEnd({}, `root-${suffix}`);
}

beforeEach(() => resetClient());
afterEach(() => vi.restoreAllMocks());

describe('LangChain handler', () => {
  it('returns promptly even when every export attempt fails', async () => {
    // The handler opts into `_awaitHandler`, so LangChain awaits its body. If
    // that body awaited delivery, the retry/backoff sequence would land on the
    // traced application's critical path — measured at ~24 s against an
    // unreachable Console.
    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      fetch: (async () => {
        throw new Error('unreachable');
      }) as unknown as typeof fetch,
    });

    const handler = new CognipeerCallbackHandler({ client });
    const started = Date.now();
    await runChain(handler, '1');
    const elapsed = Date.now() - started;

    // The first retry alone sleeps ~250 ms; three retries take seconds.
    expect(elapsed).toBeLessThan(200);
  });

  it('clears per-run bookkeeping when the session is externally owned', async () => {
    // `maybeEndSession` used to return before the clears whenever the handler
    // did not own the session, so the skip map grew by one entry per plumbing
    // runnable per run, for the life of the handler.
    const { client } = testClient();
    const session = client.startSession();
    const handler = new CognipeerCallbackHandler({ client, session });

    for (let run = 0; run < 20; run++) await runChain(handler, String(run));

    const skipped = (handler as unknown as { skipped: Map<string, unknown> }).skipped;
    expect(skipped.size).toBe(0);
    await session.end();
  });

  it('keeps recording across runs when a fixed session id is reused', async () => {
    // An ended session rejects further events, so retaining the object made
    // every run after the first silently produce nothing.
    const { client, sent } = testClient({ mode: 'stream' });
    const handler = new CognipeerCallbackHandler({ client, sessionId: 'fixed-1' });

    await runChain(handler, 'a');
    await runChain(handler, 'b');
    await handler.flush();

    const events = sent.filter((request) => request.path.endsWith('/events'));
    expect(events.length).toBeGreaterThanOrEqual(2);
    expect(sent.every((request) => request.path.includes('fixed-1'))).toBe(true);
  });
});

describe('n8n bridge', () => {
  it('resolves start() when stop() lands mid-interval', async () => {
    // `stop()` cleared the sleep timer, which WAS the promise's resolver, so
    // the poll loop could never re-check its condition.
    const bridge = new N8nBridge({
      n8nUrl: 'https://n8n.test',
      n8nApiKey: 'k',
      intervalMs: 60_000,
      fetch: (async () =>
        ({ ok: true, status: 200, json: async () => ({ data: [] }) }) as unknown as Response) as
        unknown as typeof fetch,
    });

    const running = bridge.start();
    await new Promise((resolve) => setTimeout(resolve, 20));
    bridge.stop();

    await expect(
      Promise.race([
        running.then(() => 'resolved'),
        new Promise((resolve) => setTimeout(() => resolve('pending'), 500)),
      ]),
    ).resolves.toBe('resolved');
  });
});

describe('n8n mapper', () => {
  /** A loop that runs the agent twice, each with its own model call. */
  const execution: N8nExecution = {
    id: 42,
    workflowId: 'wf-1',
    status: 'success',
    startedAt: '2026-01-15T10:00:00Z',
    stoppedAt: '2026-01-15T10:00:03Z',
    data: {
      resultData: {
        runData: {
          Agent: [
            { executionIndex: 0, metadata: { subRun: [{ node: 'Model', runIndex: 0 }] } },
            { executionIndex: 2, metadata: { subRun: [{ node: 'Model', runIndex: 1 }] } },
          ],
          Model: [
            { executionIndex: 1, data: { ai_languageModel: [[{ json: {} }]] } },
            { executionIndex: 3, data: { ai_languageModel: [[{ json: {} }]] } },
          ],
        },
      },
    },
    workflowData: {
      id: 'wf-1',
      name: 'Looping agent',
      nodes: [
        { name: 'Agent', type: 'n8n-nodes-langchain.agent' },
        { name: 'Model', type: 'n8n-nodes-langchain.lmChatOpenAi' },
      ],
      connections: { Model: { ai_languageModel: [[{ node: 'Agent' }]] } },
    },
  };

  it('parents each sub-node run onto the agent run that made it', () => {
    // The parent span id used to be built with a hard-coded run index of 0, so
    // every iteration's model call hung off the FIRST agent run — mis-shaping
    // the tree and the per-iteration token rollups.
    const payload = mapN8nExecution(execution);
    const byId = Object.fromEntries((payload?.events ?? []).map((event) => [event.id, event]));

    expect(byId['Model#0'].parentSpanId).toBe(byId['Agent#0'].spanId);
    expect(byId['Model#1'].parentSpanId).toBe(byId['Agent#1'].spanId);
    expect(byId['Agent#0'].spanId).not.toBe(byId['Agent#1'].spanId);
  });

  it('hangs a sub-node off the session root when the parent run is ambiguous', () => {
    // No `subRun` and a parent that ran twice: guessing run 0 would be a lie,
    // so the event goes to the root instead.
    const ambiguous = structuredClone(execution);
    ambiguous.data!.resultData!.runData!.Agent!.forEach((task) => delete task.metadata);
    const payload = mapN8nExecution(ambiguous);
    const model = payload?.events?.find((event) => event.id === 'Model#1');

    expect(model?.parentSpanId).toBe(payload?.rootSpanId);
  });
});

describe('OTel normalizer', () => {
  it('renders one conversation once when a span carries two conventions', () => {
    const sections = extractSections({
      'openinference.span.kind': 'LLM',
      'llm.input_messages.0.message.role': 'user',
      'llm.input_messages.0.message.content': 'book me a flight',
      'gen_ai.input.messages':
        '[{"role":"user","parts":[{"type":"text","content":"book me a flight"}]}]',
    });

    expect(sections.filter((section) => section.kind === 'message')).toHaveLength(1);
  });

  it('treats a model attribute as a model call, matching Python', () => {
    // Both "this is a model call" signals are honoured in both languages,
    // because the event type gates token attribution.
    expect(normalizeType({ 'gen_ai.request.model': 'gpt-4.1-mini' }, 'span')).toBe('ai_call');
    expect(normalizeType({ 'gen_ai.usage.input_tokens': 33 }, 'span')).toBe('ai_call');
    expect(normalizeType({ 'gen_ai.operation.name': 'invoke_agent' }, 'span')).toBe('span');
  });
});

describe('Vercel AI SDK', () => {
  it('does not break a model that uses private fields', () => {
    // The proxy used to forward itself as the `Reflect.get` receiver, so every
    // accessor ran with `this === proxy` and any `#field` read threw into the
    // traced call before a request was even made.
    class PrivateModel {
      readonly #provider = 'acme';
      specificationVersion = 'v2';
      modelId = 'm';
      get provider(): string {
        return this.#provider;
      }
      doGenerate(): unknown {
        return {};
      }
      doStream(): unknown {
        return {};
      }
    }

    const wrapped = withCognipeerTracing(new PrivateModel());
    expect(() => wrapped.provider).not.toThrow();
    expect(wrapped.provider).toBe('acme');
  });

  it('does not break a frozen model literal', () => {
    // Binding a method rewrites the value, which the Proxy `get` invariant
    // forbids for a non-configurable, non-writable own property.
    const frozen = Object.freeze({
      specificationVersion: 'v2',
      modelId: 'm',
      provider: 'p',
      doEmbed: () => ({}),
      doGenerate: () => ({}),
      doStream: () => ({}),
    });

    const wrapped = withCognipeerTracing(frozen);
    expect(() => wrapped.doEmbed).not.toThrow();
  });

  it('fails only the call that errored, not every concurrent one', async () => {
    // One integration is installed process-wide, so failing every open call
    // marked unrelated runs — on a server, other users' runs — as errored.
    const { client, sent } = testClient();
    const telemetry = new CognipeerAISDKTelemetry({ client });

    telemetry.onStart({ callId: 'a' });
    telemetry.onStart({ callId: 'b' });
    telemetry.onError({ callId: 'a', error: new Error('boom from a') });
    telemetry.onEnd({ callId: 'b' });
    await client.flush();

    const payloads = sent.map((request) => request.body as unknown as TraceSessionPayload);
    const statuses = payloads.map((payload) => payload.status);
    expect(statuses.filter((status) => status === 'error')).toHaveLength(1);
    expect(statuses.filter((status) => status === 'success')).toHaveLength(1);
  });

  it('pairs a tool start and end that carry no toolCallId', async () => {
    // The two sides used different fallbacks, so the arguments and the result
    // landed on two unrelated events.
    const { client, sent } = testClient();
    const telemetry = new CognipeerAISDKTelemetry({ client });

    telemetry.onStart({ callId: 'c1' });
    telemetry.onToolExecutionStart({ callId: 'c1', toolName: 'search', input: { q: 'rome' } });
    telemetry.onToolExecutionEnd({ callId: 'c1', toolName: 'search', output: 'RESULT' });
    telemetry.onEnd({ callId: 'c1' });
    await client.flush();

    const payload = sent[0].body as unknown as TraceSessionPayload;
    const toolEvents = (payload.events ?? []).filter((event) => event.type === 'tool_call');
    expect(toolEvents).toHaveLength(1);
    const kinds = (toolEvents[0].sections ?? []).map((section) => section.kind);
    expect(kinds).toContain('tool_call');
    expect(kinds).toContain('tool_result');
  });

  it('gives two anonymous calls of one tool their own spans', () => {
    const { client } = testClient();
    const telemetry = new CognipeerAISDKTelemetry({ client });
    telemetry.onStart({ callId: 'c1' });
    telemetry.onToolExecutionStart({ callId: 'c1', toolName: 'search', input: 'A' });
    telemetry.onToolExecutionStart({ callId: 'c1', toolName: 'search', input: 'B' });

    const state = (telemetry as unknown as { calls: Map<string, { anonymousToolSpans: Map<string, string[]> }> })
      .calls.get('c1');
    expect(state?.anonymousToolSpans.get('search')).toHaveLength(2);
    // Distinct spans, unlike the old shared `spanIdFrom(name)` fallback.
    expect(new Set(state?.anonymousToolSpans.get('search')).size).toBe(2);
    expect(spanIdFrom('search')).not.toBe(state?.anonymousToolSpans.get('search')?.[0]);
  });
});
