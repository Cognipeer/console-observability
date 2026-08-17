/**
 * `responseFormat` — the structured-output contract carried per model call.
 *
 * Every producer has to land on the same flat `{type, name?, strict?, schema?}`
 * shape, because the Console normalizes exactly that into its
 * `response_format` section. A trace that omits it cannot distinguish a model
 * that CHOSE prose from one that was never asked for JSON, and a replay of the
 * call runs under a looser contract than production did.
 */

import { describe, expect, it } from 'vitest';

import { extractResponseFormat } from '../src/otel/normalize';
import { normalizeAISDKResponseFormat } from '../src/integrations/vercel-ai';
import { CognipeerCallbackHandler } from '../src/integrations/langchain';
import { CognipeerObservability } from '../src/core/client';
import type { TraceSessionPayload } from '../src/core/types';

const SCHEMA = { type: 'object', properties: { total: { type: 'number' } }, required: ['total'] };

describe('OTLP attribute extraction', () => {
  it('reads response_format out of OpenInference invocation parameters', () => {
    const format = extractResponseFormat({
      'llm.invocation_parameters': JSON.stringify({
        temperature: 0,
        response_format: { type: 'json_schema', json_schema: { name: 'invoice', strict: true, schema: SCHEMA } },
      }),
    });
    expect(format).toEqual({ type: 'json_schema', name: 'invoice', strict: true, schema: SCHEMA });
  });

  it('reads a directly attached response_format body', () => {
    expect(extractResponseFormat({ 'llm.request.response_format': JSON.stringify({ type: 'json_object' }) }))
      .toEqual({ type: 'json_object' });
  });

  it('reassembles the GenAI split form (output type + schema)', () => {
    expect(extractResponseFormat({
      'gen_ai.output.type': 'json',
      'gen_ai.request.structured_output_schema': JSON.stringify(SCHEMA),
      'gen_ai.request.structured_output_name': 'invoice',
    })).toEqual({ type: 'json_schema', name: 'invoice', schema: SCHEMA });
  });

  it('treats JSON mode without a schema as a contract of its own', () => {
    expect(extractResponseFormat({ 'gen_ai.output.type': 'json' })).toEqual({ type: 'json_object' });
  });

  it('says nothing when the span says nothing about the contract', () => {
    expect(extractResponseFormat({})).toBeUndefined();
    expect(extractResponseFormat({ 'gen_ai.output.type': 'text' })).toBeUndefined();
    expect(extractResponseFormat({ 'llm.invocation_parameters': 'not json' })).toBeUndefined();
  });
});

describe('AI SDK normalization', () => {
  it("maps the SDK's `json` mode with a schema onto json_schema", () => {
    expect(normalizeAISDKResponseFormat({ type: 'json', name: 'invoice', schema: SCHEMA }))
      .toEqual({ type: 'json_schema', name: 'invoice', schema: SCHEMA });
  });

  it("maps `json` without a schema onto json_object", () => {
    expect(normalizeAISDKResponseFormat({ type: 'json' })).toEqual({ type: 'json_object' });
  });

  it('falls back to generateObject telemetry attributes', () => {
    expect(normalizeAISDKResponseFormat(undefined, {
      'ai.schema': JSON.stringify(SCHEMA),
      'ai.schema.name': 'invoice',
    })).toEqual({ type: 'json_schema', name: 'invoice', schema: SCHEMA });
  });

  it('treats no-schema object output as JSON mode', () => {
    expect(normalizeAISDKResponseFormat(undefined, { 'ai.settings.output': 'no-schema' }))
      .toEqual({ type: 'json_object' });
  });

  it('stays silent for an unconstrained text call', () => {
    expect(normalizeAISDKResponseFormat(undefined, {})).toBeUndefined();
    expect(normalizeAISDKResponseFormat({ type: 'text' })).toEqual({ type: 'text' });
  });
});

describe('LangChain callback handler', () => {
  function testClient() {
    const sent: Array<Record<string, unknown>> = [];
    const client = new CognipeerObservability({
      apiKey: 'cpeer_test_key_000000000000',
      baseUrl: 'https://console.test',
      mode: 'batch',
      fetch: (async (_url: string, init: { body?: string }) => {
        sent.push(JSON.parse(init.body ?? '{}'));
        return { ok: true, status: 200, text: async () => '' };
      }) as unknown as typeof fetch,
    });
    return { client, sent };
  }

  it('records the response_format bound via invocation params', async () => {
    const { client, sent } = testClient();
    const handler = new CognipeerCallbackHandler({ client, agent: { name: 'test' } });

    handler.handleLLMStart(
      { id: ['ChatOpenAI'] } as never,
      ['summarize this'],
      'run-1',
      undefined,
      {
        invocation_params: {
          model: 'gpt-4.1-mini',
          response_format: { type: 'json_schema', json_schema: { name: 'invoice', strict: true, schema: SCHEMA } },
        },
      },
    );
    await handler.handleLLMEnd({ generations: [[{ text: '{"total":1}' }]] } as never, 'run-1');
    await client.flush();

    const events = (sent[0] as unknown as TraceSessionPayload).events ?? [];
    const call = events.find((event) => event.type === 'ai_call');
    expect(call?.responseFormat).toEqual({ type: 'json_schema', name: 'invoice', strict: true, schema: SCHEMA });
  });

  it('leaves the field off an unconstrained call', async () => {
    const { client, sent } = testClient();
    const handler = new CognipeerCallbackHandler({ client, agent: { name: 'test' } });

    handler.handleLLMStart(
      { id: ['ChatOpenAI'] } as never,
      ['summarize this'],
      'run-2',
      undefined,
      { invocation_params: { model: 'gpt-4.1-mini' } },
    );
    await handler.handleLLMEnd({ generations: [[{ text: 'hello' }]] } as never, 'run-2');
    await client.flush();

    const events = (sent[0] as unknown as TraceSessionPayload).events ?? [];
    expect(events.find((event) => event.type === 'ai_call')?.responseFormat).toBeUndefined();
  });
});
