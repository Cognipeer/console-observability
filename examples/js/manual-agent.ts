/**
 * A hand-rolled agent loop, traced with no framework at all.
 *
 * This is the template to copy when your agent is your own code. It shows the
 * three things that matter:
 *
 *   - `trace()` opens the session — one agent run, one row in Tracing → Sessions.
 *   - `observe()` wraps any function into an event, nested automatically under
 *     whatever called it.
 *   - `type: 'tool_call'` / `'ai_call'` make an event render as a tool
 *     invocation or a model call rather than a generic step.
 *
 * Runs entirely offline — the "model" is a scripted stub — so it costs nothing
 * and is safe to run in CI.
 *
 *   npx tsx examples/js/manual-agent.ts
 *
 * In Console: Tracing → Sessions → "order-bot". The timeline should read
 * plan → searchOrders → summarise, with `searchOrders` nested under `plan`.
 */

import { flush, init, observe, trace, type TraceSession } from '@cognipeer/observability';

// ─────────────────────────────────────────────────────────────────────
//  Wire tracing in — this is the whole integration
// ─────────────────────────────────────────────────────────────────────
if (!process.env.COGNIPEER_API_KEY) {
  console.error('Set COGNIPEER_API_KEY (Settings → API Tokens, with `tracing` enabled).');
  process.exit(1);
}

init({ agent: { name: 'order-bot', version: '1.0.0' } });
// ─────────────────────────────────────────────────────────────────────

const ORDERS: Record<string, Record<string, unknown>> = {
  'A-1001': { status: 'shipped', carrier: 'DHL', eta: '2026-08-18' },
  'A-1002': { status: 'processing', carrier: null, eta: null },
};

/** A tool. Arguments and return value become the event's two sections. */
const searchOrders = observe(
  async (orderId: string) => ORDERS[orderId] ?? { status: 'not_found' },
  { name: 'searchOrders', type: 'tool_call', toolName: 'search_orders' },
);

/** A step that calls a tool. The tool event nests under this one. */
const plan = observe(
  async (question: string) => searchOrders(question.split(' ').pop() as string),
  { name: 'plan' },
);

const summarise = observe(
  async (order: Record<string, unknown>) =>
    order.status === 'not_found'
      ? 'I could not find that order.'
      : `Your order is ${order.status} (carrier: ${order.carrier}).`,
  { name: 'summarise' },
);

/**
 * A model call, recorded by hand.
 *
 * `observe({ type: 'ai_call' })` would also work and is one line — but a model
 * call is worth recording explicitly because four extra fields unlock the rest
 * of Console:
 *
 *   - `model` — the provider's model id, which is what cost resolution matches
 *     against Model Hub. A nickname prices at zero.
 *   - token counts — the provider's own numbers. Never estimate them, and omit
 *     them entirely rather than sending zeros; an absent value reads as
 *     unknown, a zero silently under-reports spend.
 *   - `cachedInputTokens` — a SUBSET of `inputTokens`, priced at the cached rate.
 *   - `toolDefinitions` — the tool menu this call was offered. It changes
 *     between turns and is often the biggest line item in the prompt bill,
 *     which is why it belongs on the event rather than the session.
 */
function callModel(session: TraceSession, prompt: string): string {
  const completion = `(pretend completion for: ${prompt.slice(0, 40)})`;

  session.record({
    type: 'ai_call',
    label: 'gpt-4.1-mini',
    model: 'gpt-4.1-mini',
    inputTokens: 412,
    outputTokens: 58,
    cachedInputTokens: 256,
    sections: [
      { kind: 'message', role: 'user', content: prompt },
      { kind: 'message', role: 'assistant', content: completion },
    ],
    toolDefinitions: [
      {
        name: 'search_orders',
        description: 'Look up an order by id',
        parameters: {
          type: 'object',
          properties: { order_id: { type: 'string' } },
          required: ['order_id'],
        },
      },
    ],
  });

  return completion;
}

async function main(): Promise<void> {
  const question = 'where is order A-1001';

  // `threadId` groups this run with the rest of the conversation in
  // Tracing → Threads. Use whatever key your app already has.
  await trace({ name: 'order-bot', threadId: 'conv-42' }, async (session) => {
    const order = await plan(question);
    callModel(session, `Summarise this order: ${JSON.stringify(order)}`);
    console.log(await summarise(order));
  });

  // Short-lived process: without this it can exit before the last export
  // lands. A long-running server does not need it.
  await flush();
  console.log("\nLook in Console → Tracing → Sessions → agent 'order-bot'");
}

void main();
