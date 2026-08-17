import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'integrations/langchain': 'src/integrations/langchain.ts',
    'integrations/langgraph': 'src/integrations/langgraph.ts',
    'integrations/openai-agents': 'src/integrations/openai-agents.ts',
    'integrations/claude-agent-sdk': 'src/integrations/claude-agent-sdk.ts',
    'integrations/vercel-ai': 'src/integrations/vercel-ai.ts',
    'integrations/n8n': 'src/integrations/n8n.ts',
    'otel/index': 'src/otel/index.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  target: 'node18',
  splitting: false,
  treeshake: true,
  // Framework packages are optional peers: a user who only installs the
  // LangChain integration must not have @openai/agents resolved at build time.
  external: [
    '@anthropic-ai/claude-agent-sdk',
    '@langchain/core',
    '@openai/agents',
    '@opentelemetry/api',
    '@opentelemetry/sdk-trace-base',
    'ai',
  ],
});
