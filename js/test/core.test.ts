/**
 * Core behaviour: configuration, identifiers, redaction, session delivery.
 *
 * These assert the guarantees the README makes — never throw, never block,
 * absent-not-zero, secrets do not leave the process — because those are what
 * make the package safe to put in front of a production agent.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CognipeerObservability,
  init,
  liveClientCount,
  resetClient,
} from '../src/core/client';
import { normalizeBaseUrl, resolveConfig } from '../src/core/config';
import { getCurrentSession } from '../src/core/context';
import { spanIdFrom, traceIdFrom } from '../src/core/ids';
import { observe, trace } from '../src/core/observe';
import {
  redactString,
  sanitizeSections,
  stringifyContent,
  UNSERIALIZABLE,
} from '../src/core/redact';
import type { StreamStartPayload, TraceSessionPayload } from '../src/core/types';

interface Sent {
  path: string;
  body: Record<string, unknown>;
}

/** A client whose transport is a list. Nothing here touches the network. */
function testClient(overrides: Record<string, unknown> = {}) {
  const sent: Sent[] = [];
  const fetchImpl = (async (url: string, options: { body: string }) => {
    sent.push({ path: new URL(url).pathname, body: JSON.parse(options.body) });
    return { ok: true, status: 200, text: async () => '' } as Response;
  }) as unknown as typeof fetch;

  const client = new CognipeerObservability({
    apiKey: 'cpeer_test_key_000000000000',
    baseUrl: 'https://console.test',
    fetch: fetchImpl,
    mode: 'batch',
    ...overrides,
  });
  return { client, sent };
}

beforeEach(() => resetClient());
afterEach(() => vi.restoreAllMocks());

describe('configuration', () => {
  it('disables itself instead of throwing when no API key is present', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const config = resolveConfig({ apiKey: '' });
    expect(config.enabled).toBe(false);
    expect(warn).toHaveBeenCalled();
  });

  it('accepts the legacy fully-qualified ingest path as a base URL', () => {
    expect(normalizeBaseUrl('https://console.acme.com/api/client/v1/')).toBe(
      'https://console.acme.com',
    );
  });

  it('treats capture: none as disabled', () => {
    expect(resolveConfig({ apiKey: 'k', capture: 'none' }).enabled).toBe(false);
  });
});

describe('identifiers', () => {
  it('does not collide for UUIDv7 run ids', () => {
    // UUIDv7's first 16 hex digits are a millisecond timestamp plus 12 bits of
    // entropy, so truncating would collide for runs started in the same
    // millisecond — which is most of them.
    const a = spanIdFrom('01a00571-cf48-7a60-8000-000000000001');
    const b = spanIdFrom('01a00571-cf48-7a60-8000-000000000002');
    expect(a).not.toBe(b);
    expect(a).toMatch(/^[0-9a-f]{16}$/);
  });

  it('is deterministic and passes real span ids through', () => {
    expect(spanIdFrom('run-1')).toBe(spanIdFrom('run-1'));
    expect(spanIdFrom('00f067aa0ba902b7')).toBe('00f067aa0ba902b7');
    expect(traceIdFrom('4bf92f3577b34da6a3ce929d0e0e4736')).toBe(
      '4bf92f3577b34da6a3ce929d0e0e4736',
    );
    expect(traceIdFrom('conversation-1')).toMatch(/^[0-9a-f]{32}$/);
  });
});

describe('redaction', () => {
  it('redacts credential shapes', () => {
    const out = redactString('use sk-abcdefghijklmnopqrstuvwx and AKIAIOSFODNN7EXAMPLE');
    expect(out).not.toContain('sk-abcdefghijklmnopqrstuvwx');
    expect(out).not.toContain('AKIAIOSFODNN7EXAMPLE');
  });

  it('leaves ordinary prose alone', () => {
    const prose = 'The flight to Rome costs 240 EUR and departs at 09:15.';
    expect(redactString(prose)).toBe(prose);
  });

  it('strips base64 data URLs', () => {
    const out = redactString(`data:image/png;base64,${'A'.repeat(500)}`);
    expect(out).not.toContain('AAAA');
    expect(out).toContain('stripped');
  });

  it('keeps structure but drops bodies under metadata capture', () => {
    const config = resolveConfig({ apiKey: 'k', capture: 'metadata' });
    const sections = sanitizeSections(
      [{ kind: 'message', role: 'user', content: 'secret question' }],
      config,
    );
    expect(sections?.[0].kind).toBe('message');
    expect(sections?.[0].content).toBeUndefined();
  });

  it('caps oversized content', () => {
    const config = resolveConfig({ apiKey: 'k', maxContentChars: 100 });
    const sections = sanitizeSections([{ kind: 'message', content: 'x'.repeat(5000) }], config);
    expect(String(sections?.[0].content).length).toBeLessThan(300);
    expect(sections?.[0].truncated).toBe(true);
  });
});

describe('session delivery', () => {
  it('sends one request carrying every event in batch mode', async () => {
    const { client, sent } = testClient();
    const session = client.startSession({ threadId: 'conv-1', agent: { name: 'test-agent' } });
    session.record({ type: 'ai_call', model: 'gpt-4.1-mini', inputTokens: 10, outputTokens: 5 });
    session.record({ type: 'tool_call', toolName: 'search' });
    await session.end();

    expect(sent).toHaveLength(1);
    const payload = sent[0].body as unknown as TraceSessionPayload;
    expect(sent[0].path).toBe('/api/client/v1/tracing/sessions');
    expect(payload.threadId).toBe('conv-1');
    expect(payload.events?.map((event) => event.type)).toEqual(['ai_call', 'tool_call']);
    expect(payload.summary?.totalInputTokens).toBe(10);
    expect(payload.summary?.eventCounts).toEqual({ ai_call: 1, tool_call: 1 });
  });

  it('opens before it appends in stream mode', async () => {
    const { client, sent } = testClient({ mode: 'stream' });
    const session = client.startSession();
    session.record({ type: 'ai_call' });
    await session.end();

    expect(sent[0].path).toMatch(/\/start$/);
    expect(sent[1].path).toMatch(/\/events$/);
    expect(sent[sent.length - 1].path).toMatch(/\/end$/);
  });

  it('merges client-default and per-session metadata into the batch payload', async () => {
    const { client, sent } = testClient({ metadata: { env: 'prod' } });
    const session = client.startSession({ metadata: { complexity: 'complex' } });
    session.record({ type: 'ai_call' });
    await session.end();

    const payload = sent[0].body as unknown as TraceSessionPayload;
    expect(payload.metadata).toEqual({ env: 'prod', complexity: 'complex' });
  });

  it('carries session metadata on the streaming /start payload', async () => {
    const { client, sent } = testClient({ mode: 'stream' });
    const session = client.startSession({ metadata: { complexity: 'simple' } });
    session.record({ type: 'ai_call' });
    await session.end();

    const start = sent.find((entry) => entry.path.endsWith('/start'));
    expect((start?.body as unknown as StreamStartPayload).metadata).toEqual({ complexity: 'simple' });
  });

  it('pairs a span open/close into one event with a parent link', async () => {
    const { client, sent } = testClient();
    const session = client.startSession();
    session.openSpan('run-1', { type: 'ai_call', label: 'gpt-4.1-mini', model: 'gpt-4.1-mini' });
    session.openSpan('run-2', { type: 'tool_call', parentKey: 'run-1', toolName: 'search' });
    session.closeSpan('run-2', { sections: [{ kind: 'tool_result', content: 'ok' }] });
    session.closeSpan('run-1', { inputTokens: 100, outputTokens: 20, cachedInputTokens: 64 });
    await session.end();

    const events = (sent[0].body as unknown as TraceSessionPayload).events ?? [];
    expect(events).toHaveLength(2);
    const [tool, modelCall] = events;
    expect(tool.type).toBe('tool_call');
    expect(modelCall.cachedInputTokens).toBe(64);
    expect(tool.parentSpanId).toBe(modelCall.spanId);
  });

  it('leaves unreported usage absent rather than zero', async () => {
    const { client, sent } = testClient();
    const session = client.startSession();
    session.openSpan('run-1', { type: 'ai_call' });
    session.closeSpan('run-1');
    await session.end();

    const [event] = (sent[0].body as unknown as TraceSessionPayload).events ?? [];
    expect(event.inputTokens).toBeUndefined();
    expect(event.outputTokens).toBeUndefined();
  });

  it('closes spans left open when the session ends', async () => {
    const { client, sent } = testClient();
    const session = client.startSession();
    session.openSpan('never-closed', { type: 'tool_call' });
    await session.end();

    expect((sent[0].body as unknown as TraceSessionPayload).events).toHaveLength(1);
  });

  it('marks the session failed when an event errors', async () => {
    const { client, sent } = testClient();
    const session = client.startSession();
    session.openSpan('run-1', { type: 'tool_call' });
    session.closeSpan('run-1', { error: new Error('boom') });
    await session.end();

    const payload = sent[0].body as unknown as TraceSessionPayload;
    expect(payload.status).toBe('error');
    expect(payload.errors?.[0]).toMatchObject({ message: 'boom' });
  });

  it('retries a retryable status and gives up on a client error', async () => {
    const statuses: number[] = [];
    const responses = [503, 503, 200];
    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      maxRetries: 3,
      fetch: (async () => {
        const status = responses.shift() ?? 200;
        statuses.push(status);
        return { ok: status === 200, status, text: async () => '' } as Response;
      }) as unknown as typeof fetch,
    });

    const session = client.startSession();
    session.record({ type: 'ai_call' });
    await session.end();
    expect(statuses).toEqual([503, 503, 200]);

    // A 4xx is the caller's fault; retrying it just wastes the budget.
    const clientErrors: number[] = [];
    const onError = vi.fn();
    const strict = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      maxRetries: 3,
      onError,
      fetch: (async () => {
        clientErrors.push(400);
        return { ok: false, status: 400, text: async () => 'bad request' } as Response;
      }) as unknown as typeof fetch,
    });
    const failing = strict.startSession();
    failing.record({ type: 'ai_call' });
    await failing.end();
    expect(clientErrors).toHaveLength(1);
    expect(onError).toHaveBeenCalledOnce();
  });

  it('never throws when the transport fails', async () => {
    const onError = vi.fn();
    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      maxRetries: 0,
      onError,
      fetch: (async () => {
        throw new Error('network down');
      }) as unknown as typeof fetch,
    });

    const session = client.startSession();
    session.record({ type: 'ai_call' });
    await expect(session.end()).resolves.toBeUndefined();
    expect(onError).toHaveBeenCalled();
  });

  it('records nothing and throws nothing when disabled', async () => {
    const { sent } = testClient();
    const disabled = new CognipeerObservability({ apiKey: '', enabled: false });
    const session = disabled.startSession();
    expect(session.disabled).toBe(true);
    session.record({ type: 'ai_call' });
    await session.end();
    expect(sent).toHaveLength(0);
  });
});

describe('manual instrumentation', () => {
  it('nests observed calls and creates a root session', async () => {
    const { client, sent } = testClient();
    init({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      fetch: client.config.fetch,
    });

    const inner = observe(async (value: number) => value * 2, {
      name: 'inner',
      type: 'tool_call',
      toolName: 'double',
    });
    const outer = observe(async () => inner(21), { name: 'outer' });

    await expect(outer()).resolves.toBe(42);

    const events = (sent[0].body as unknown as TraceSessionPayload).events ?? [];
    const byLabel = Object.fromEntries(events.map((event) => [event.label, event]));
    expect(Object.keys(byLabel).sort()).toEqual(['inner', 'outer']);
    expect(byLabel.inner.parentSpanId).toBe(byLabel.outer.spanId);
    expect(byLabel.inner.type).toBe('tool_call');
  });

  it('keeps one session across await boundaries', async () => {
    // The failure this pins: the async-context fallback restores its frame the
    // moment a function's synchronous part returns, so work after the first
    // `await` used to escape the session and open a second one.
    const { client, sent } = testClient();
    init({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      fetch: client.config.fetch,
    });

    const plan = observe(async () => 'planned', { name: 'plan' });
    const summarise = observe(async () => 'summarised', { name: 'summarise' });

    await trace({ name: 'probe-agent', threadId: 'conv-42' }, async () => {
      await plan();
      await new Promise((resolve) => setTimeout(resolve, 5));
      await summarise();
      expect(getCurrentSession()).toBeDefined();
    });

    expect(sent).toHaveLength(1);
    const payload = sent[0].body as unknown as TraceSessionPayload;
    expect(payload.threadId).toBe('conv-42');
    expect(payload.events?.map((event) => event.label)).toEqual(['plan', 'summarise']);
  });

  it('marks the session failed and rethrows', async () => {
    const { client, sent } = testClient();
    init({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      fetch: client.config.fetch,
    });

    await expect(
      trace({ name: 'failing-agent' }, async () => {
        throw new Error('nope');
      }),
    ).rejects.toThrow('nope');

    const payload = sent[0].body as unknown as TraceSessionPayload;
    expect(payload.status).toBe('error');
    expect(payload.agent?.name).toBe('failing-agent');
  });
});

describe('hostile and shared payloads', () => {
  it('renders a shared object in full instead of calling it circular', () => {
    // Cycle detection needs an ancestor stack, not a visited-set: agent
    // payloads routinely reference one object twice (the same system message
    // in two turns, one tool definition in two menus) and marking the second
    // occurrence `[circular]` silently destroys content that is not circular.
    const shared = { name: 'search_flights', description: 'Search flights by city' };
    const out = stringifyContent({ first: { tools: [shared] }, second: { tools: [shared] } });

    expect(out).not.toContain('[circular]');
    expect(out.match(/search_flights/g)).toHaveLength(2);
  });

  it('still detects a real cycle', () => {
    const node: Record<string, unknown> = { name: 'root' };
    node.self = node;
    expect(stringifyContent(node)).toContain('[circular]');
  });

  it('never throws on a value that cannot be stringified at all', () => {
    // JSON.stringify throws AND String() throws: a null-prototype object with
    // a throwing getter is the shape that used to escape into the caller.
    const hostile = Object.create(null) as Record<string, unknown>;
    Object.defineProperty(hostile, 'boom', {
      enumerable: true,
      get() {
        throw new Error('nope');
      },
    });

    expect(() => stringifyContent(hostile)).not.toThrow();
    expect(stringifyContent(hostile)).toBe(UNSERIALIZABLE);
  });

  it('records a hostile payload without throwing into the caller', async () => {
    const { client, sent } = testClient();
    const hostile = Object.create(null) as Record<string, unknown>;
    Object.defineProperty(hostile, 'boom', {
      enumerable: true,
      get() {
        throw new Error('nope');
      },
    });

    const session = client.startSession();
    expect(() =>
      session.record({
        type: 'ai_call',
        sections: [{ kind: 'message', content: hostile }],
        metadata: { hostile },
      }),
    ).not.toThrow();
    await session.end();

    expect(sent).toHaveLength(1);
  });
});

describe('metadata and config sanitisation', () => {
  it('redacts secrets and caps blobs in event metadata', async () => {
    const { client, sent } = testClient({ maxContentChars: 200 });
    const session = client.startSession();
    session.record({
      type: 'ai_call',
      metadata: {
        apiKey: 'sk-abcdefghijklmnopqrstuvwx',
        nested: { token: 'cpeer_abcdefghijklmnopqrstuv' },
        blob: 'x'.repeat(50_000),
      },
    });
    await session.end();

    const wire = JSON.stringify(sent[0].body);
    expect(wire).not.toContain('sk-abcdefghijklmnopqrstuvwx');
    expect(wire).not.toContain('cpeer_abcdefghijklmnopqrstuv');
    expect(wire).toContain('[redacted]');
    // The blob is capped rather than shipped whole.
    expect(wire.length).toBeLessThan(10_000);
  });

  it('preserves metadata structure rather than flattening it', async () => {
    const { client, sent } = testClient();
    const session = client.startSession();
    session.record({
      type: 'span',
      metadata: { langgraph_node: 'agent', langgraph_step: 3, nested: { ok: true } },
    });
    await session.end();

    const [event] = (sent[0].body as unknown as TraceSessionPayload).events ?? [];
    expect(event.metadata).toEqual({
      langgraph_node: 'agent',
      langgraph_step: 3,
      nested: { ok: true },
    });
  });

  it('redacts the session config too', async () => {
    const { client, sent } = testClient();
    const session = client.startSession({
      config: { systemPrompt: 'internal key sk-abcdefghijklmnopqrstuvwx' },
    });
    session.record({ type: 'ai_call' });
    await session.end();

    const wire = JSON.stringify(sent[0].body);
    expect(wire).not.toContain('sk-abcdefghijklmnopqrstuvwx');
    expect(wire).toContain('[redacted]');
  });
});

describe('failure isolation', () => {
  it('survives an onError handler that itself throws', async () => {
    // A throwing handler used to poison the delivery chain, which nobody
    // awaits while a session is open — enough to kill the process under
    // Node's default unhandled-rejection policy.
    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      maxRetries: 0,
      onError: () => {
        throw new Error('sentry not initialised');
      },
      fetch: (async () => {
        throw new Error('network down');
      }) as unknown as typeof fetch,
    });

    const session = client.startSession();
    session.record({ type: 'ai_call' });
    await expect(session.end()).resolves.toBeUndefined();
  });

  it('does not replace the application error when the app itself throws', async () => {
    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      maxRetries: 0,
      onError: () => {
        throw new Error('sentry not initialised');
      },
      fetch: (async () => {
        throw new Error('network down');
      }) as unknown as typeof fetch,
    });

    await expect(
      client.trace({ agent: { name: 'agent' } }, async () => {
        throw new Error('THE REAL APPLICATION ERROR');
      }),
    ).rejects.toThrow('THE REAL APPLICATION ERROR');
  });

  it('deregisters a client on shutdown', async () => {
    const before = liveClientCount();
    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      fetch: (async () => ({ ok: true, status: 200, text: async () => '' })) as unknown as typeof fetch,
    });
    expect(liveClientCount()).toBe(before + 1);
    await client.shutdown();
    expect(liveClientCount()).toBe(before);
  });
});

describe('flush covers in-flight deliveries', () => {
  it('waits for a session that has already ended', async () => {
    // `end()` returns before the network settles, so the session leaves the
    // client's live set while its last request is still on the wire. A script
    // following the documented `await flush()` before exit must still get it.
    let resolveRequest!: () => void;
    const arrived = new Promise<void>((resolve) => {
      resolveRequest = resolve;
    });
    let delivered = false;

    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      fetch: (async () => {
        resolveRequest();
        await new Promise((resolve) => setTimeout(resolve, 30));
        delivered = true;
        return { ok: true, status: 200, text: async () => '' } as Response;
      }) as unknown as typeof fetch,
    });

    const session = client.startSession();
    session.record({ type: 'ai_call' });
    void session.end();
    await arrived;
    expect(delivered).toBe(false);

    await client.flush();
    expect(delivered).toBe(true);
  });
});
