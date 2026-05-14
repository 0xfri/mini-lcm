/**
 * Mini LCM - Lightweight Lossless Context Management
 * 
 * OpenClaw context engine plugin with:
 * - CJK-aware token estimation
 * - SQLite message storage
 * - FTS5 full-text search
 * - Vector semantic search (Alibaba text-embedding-v3)
 * - Cross-session memory
 * - Automatic compaction with LLM summarization
 */

import { MiniLcmEngine } from './context-engine.js';

export default function register(api: any) {
  const config = api.config?.plugins?.entries?.['mini-lcm']?.config || {};
  
  // Get the LLM completion function from the host
  const llmComplete = async (params: any) => {
    return api.llm?.complete?.(params) || api.runtime?.llm?.complete?.(params);
  };

  // Get API key for embedding
  const apiKey = process.env.DASHSCOPE_API_KEY 
    || api.config?.plugins?.entries?.['mini-lcm']?.config?.apiKey
    || '';

  api.registerContextEngine('mini-lcm', (ctx: any) => {
    return new MiniLcmEngine({
      ...config,
      llmComplete,
      apiKey,
    });
  });

  console.log('[mini-lcm] Context engine registered');
}

export { MiniLcmEngine } from './context-engine.js';
export { estimateTokens, estimateMessageTokens } from './token-estimator.js';
export { MemoryStore } from './memory-store.js';
export { DashscopeEmbedding, cosineSimilarity } from './embedding.js';
export { MiniLcmDb } from './db.js';
