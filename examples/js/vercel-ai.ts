/**
 * Vercel AI SDK, traced through its native telemetry integration.
 *
 * The SDK moved its telemetry seam twice in two majors, so the integration
 * ships three routes. This example shows the recommended one:
 *
 *   - **`installVercelAITracing()`** (ai@6+) — the native `Telemetry`
 *     interface. The only seam that hands over structured tool definitions
 *     with full JSON Schema, flat usage with a cache-read breakdown, and
 *     per-tool timings. `ai@7` deleted OpenTelemetry from the core package, so
 *     on current versions this is also the only telemetry seam left.
 *
 * The other two, for reference:
 *
 *   - `cognipeerTelemetry()` — injects a tracer into `experimental_telemetry`
 *     on ai@3–5, where OTel was still the seam.
 *   - `withCognipeerTracing(model)` — middleware, portable across every major,
 *     sees the exact wire payload. It wraps ONE model call and knows nothing
 *     about the agent loop, so pair it with `trace()` to group a run.
 *
 * ⚠️ Calls the OpenAI API — this one costs money (a few tenths of a cent).
 *
 *   npm install @cognipeer/observability ai @ai-sdk/openai zod
 *   export OPENAI_API_KEY=sk-…
 *   npx tsx examples/js/vercel-ai.ts
 *
 * In Console: Tracing → Threads → "conv-42", one session per `generateText`.
 */

import { openai } from '@ai-sdk/openai';
import { flush, init, trace } from '@cognipeer/observability';
import { generateText, tool } from 'ai';
import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────
//  Wire tracing in — this is the whole integration
// ─────────────────────────────────────────────────────────────────────
import {
  installVercelAITracing,
  withCognipeerTracing,
} from '@cognipeer/observability/vercel-ai';

for (const required of ['COGNIPEER_API_KEY', 'OPENAI_API_KEY']) {
  if (!process.env[required]) {
    console.error(`Set ${required} before running this example.`);
    process.exit(1);
  }
}

init({ agent: { name: 'support-bot', version: '1.0.0' } });
await installVercelAITracing({ threadId: 'conv-42' });
// ─────────────────────────────────────────────────────────────────────

const ORDERS: Record<string, Record<string, unknown>> = {
  'A-1001': { status: 'shipped', carrier: 'DHL', eta: '2026-08-18' },
  'A-1002': { status: 'processing', carrier: null, eta: null },
};

const lookupOrder = tool({
  description: 'Look up the delivery status of an order by its id.',
  inputSchema: z.object({ orderId: z.string() }),
  execute: async ({ orderId }) => ORDERS[orderId] ?? { status: 'not_found' },
});

// ── Route 1 (recommended): the global telemetry integration ──────────
const { text } = await generateText({
  model: openai('gpt-4.1-mini'),
  prompt: 'Where is order A-1001?',
  tools: { lookupOrder },
  // `functionId` becomes the event label, so give it something you would want
  // to read in a timeline.
  telemetry: { functionId: 'order-status-turn' },
});
console.log(text);

// ── Route 2 (portable): middleware, no global registration ───────────
// Works identically on every `ai` major. `trace()` around it is what groups
// several model calls into one session — the middleware alone sees one call.
await trace({ name: 'support-bot', threadId: 'conv-42' }, async () => {
  const { text: summary } = await generateText({
    model: withCognipeerTracing(openai('gpt-4.1-mini')),
    prompt: `Summarise in one sentence: ${text}`,
  });
  console.log(summary);
});

// Short-lived process: without this it can exit before the last export lands.
await flush();
console.log("\nLook in Console → Tracing → Threads → 'conv-42'");
