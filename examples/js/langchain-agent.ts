/**
 * A real tool-calling LangChain agent, traced with one callback handler.
 *
 * Demonstrates what the handler recovers from an ordinary agent run:
 *
 *   - the model call, with the provider's model id and token counts;
 *   - **the tool menu the model was offered**, read from `invocation_params`
 *     (LangChain strips `tools` from the metadata it hands to tracers, so the
 *     LangSmith/OTel export path shows none of this);
 *   - each tool invocation with its arguments and result;
 *   - the whole thing as one session, nested by parent run.
 *
 * ⚠️ Calls the OpenAI API — this one costs money (a few tenths of a cent).
 *
 *   npm install @cognipeer/observability @langchain/core @langchain/openai @langchain/langgraph zod
 *   export OPENAI_API_KEY=sk-…
 *   npx tsx examples/js/langchain-agent.ts
 *
 * In Console: Tracing → Sessions → "support-bot". Open the `ai_call` event and
 * expand its Tool Definitions section — that is the menu, captured per call.
 */

import { flush, init } from '@cognipeer/observability';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────
//  Wire tracing in — this is the whole integration
// ─────────────────────────────────────────────────────────────────────
import { CognipeerCallbackHandler } from '@cognipeer/observability/langchain';

for (const required of ['COGNIPEER_API_KEY', 'OPENAI_API_KEY']) {
  if (!process.env[required]) {
    console.error(`Set ${required} before running this example.`);
    process.exit(1);
  }
}

init({ agent: { name: 'support-bot', version: '1.0.0' } });
// ─────────────────────────────────────────────────────────────────────

const ORDERS: Record<string, Record<string, unknown>> = {
  'A-1001': { status: 'shipped', carrier: 'DHL', eta: '2026-08-18' },
  'A-1002': { status: 'processing', carrier: null, eta: null },
};

const lookupOrder = tool(
  async ({ orderId }) => JSON.stringify(ORDERS[orderId] ?? { status: 'not_found' }),
  {
    name: 'lookup_order',
    description: 'Look up the delivery status of an order by its id.',
    schema: z.object({ orderId: z.string().describe('The order id, e.g. A-1001') }),
  },
);

async function main(): Promise<void> {
  const agent = createReactAgent({
    llm: new ChatOpenAI({ model: 'gpt-4.1-mini', temperature: 0 }),
    tools: [lookupOrder],
  });

  const result = await agent.invoke(
    { messages: [{ role: 'user', content: 'Where is order A-1001?' }] },
    // `threadId` groups this run with the rest of the conversation in
    // Tracing → Threads. Everything else is default.
    { callbacks: [new CognipeerCallbackHandler({ threadId: 'conv-42' })] },
  );

  const messages = result.messages;
  console.log(messages[messages.length - 1].content);

  // Short-lived process: without this it can exit before the last export
  // lands. JS callbacks are also fire-and-forget by default, which is why the
  // handler opts into awaiting — but the final flush is still yours to call.
  await flush();
  console.log("\nLook in Console → Tracing → Sessions → agent 'support-bot'");
}

void main();
