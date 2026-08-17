/**
 * OpenAI Agents SDK, traced through its own tracing processor.
 *
 * Demonstrates:
 *
 *   - `installOpenAIAgentsTracing()` — one call, and every run in the process
 *     is traced. By default it REPLACES the SDK's processor list, so traces
 *     stop going to OpenAI's dashboard and no prompt data leaves for a second
 *     destination. Pass `keepOpenAIExporter: true` to export to both.
 *   - `groupId` — the SDK's only conversation identifier, and nothing sets it
 *     for you. Without it every run is an orphan session.
 *   - function tools arriving as nested events under the agent span.
 *
 * ⚠️ Calls the OpenAI API — this one costs money (a few tenths of a cent).
 *
 *   npm install @cognipeer/observability @openai/agents zod
 *   export OPENAI_API_KEY=sk-…
 *   npx tsx examples/js/openai-agents.ts
 *
 * In Console: Tracing → Threads → "conv-42". The `ai_call` event carries the
 * full tool schemas, because this agent runs on the Responses API — on the
 * Chat Completions path the SDK reports tool *names* only.
 */

import { flush, init } from '@cognipeer/observability';
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────
//  Wire tracing in — this is the whole integration
// ─────────────────────────────────────────────────────────────────────
import { installOpenAIAgentsTracing } from '@cognipeer/observability/openai-agents';

for (const required of ['COGNIPEER_API_KEY', 'OPENAI_API_KEY']) {
  if (!process.env[required]) {
    console.error(`Set ${required} before running this example.`);
    process.exit(1);
  }
}

init({ agent: { name: 'support-bot', version: '1.0.0' } });
// `install…` is async here (unlike Python) because it dynamically imports the
// SDK, which is an optional peer dependency.
await installOpenAIAgentsTracing();
// ─────────────────────────────────────────────────────────────────────

const ORDERS: Record<string, Record<string, unknown>> = {
  'A-1001': { status: 'shipped', carrier: 'DHL', eta: '2026-08-18' },
  'A-1002': { status: 'processing', carrier: null, eta: null },
};

const lookupOrder = tool({
  name: 'lookup_order',
  description: 'Look up the delivery status of an order by its id.',
  parameters: z.object({ orderId: z.string() }),
  execute: async ({ orderId }) => ORDERS[orderId] ?? { status: 'not_found' },
});

const agent = new Agent({
  name: 'support-bot',
  instructions: 'You are a concise customer support agent.',
  model: 'gpt-4.1-mini',
  tools: [lookupOrder],
});

// `groupId` is what groups this run with the rest of the conversation in
// Tracing → Threads.
const result = await run(agent, 'Where is order A-1001?', { groupId: 'conv-42' });
console.log(result.finalOutput);

// Short-lived process: without this it can exit before the last export lands.
await flush();
console.log("\nLook in Console → Tracing → Threads → 'conv-42'");
