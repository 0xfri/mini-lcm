/**
 * Mini LCM Context Engine
 * 
 * Implements OpenClaw's ContextEngine interface:
 * - bootstrap: Initialize DB for session
 * - ingest/ingestBatch: Store messages
 * - assemble: Build context with summaries + memories + recent messages
 * - compact: Compress old messages into summaries
 * - afterTurn: Post-turn maintenance
 */

import type {
  ContextEngine,
  ContextEngineInfo,
  AssembleResult,
  CompactResult,
  IngestResult,
  IngestBatchResult,
  BootstrapResult,
} from 'openclaw/plugin-sdk';
import type { AgentMessage } from '@mariozechner/pi-agent-core';

import { MiniLcmDb } from './db.js';
import { estimateTokens, estimateMessageTokens } from './token-estimator.js';
import { MemoryStore } from './memory-store.js';
import { DashscopeEmbedding } from './embedding.js';

// Default config values
const DEFAULTS = {
  freshTailCount: 64,
  contextThreshold: 0.75,
  summaryModel: 'xiaomi-tokenplan/mimo-v2.5-pro',
  embeddingModel: 'dashscope/text-embedding-v3',
  embeddingDim: 1024,
  dbPath: '~/.openclaw/mini-lcm.db',
};

export class MiniLcmEngine implements ContextEngine {
  readonly info: ContextEngineInfo = {
    id: 'mini-lcm',
    name: 'Mini LCM',
    version: '2026.5.14.001',
    ownsCompaction: true,
  };

  private db: MiniLcmDb;
  private memoryStore: MemoryStore;
  private config: typeof DEFAULTS;
  private llmComplete: ((params: any) => Promise<any>) | null = null;

  constructor(config: Partial<typeof DEFAULTS> & { llmComplete?: any, apiKey?: string }) {
    this.config = { ...DEFAULTS, ...config };
    this.db = new MiniLcmDb(this.config.dbPath);

    // Initialize embedding provider
    const embedder = new DashscopeEmbedding({
      apiKey: config.apiKey || process.env.DASHSCOPE_API_KEY || '',
      model: this.config.embeddingModel.split('/')[1] || 'text-embedding-v3',
      dim: this.config.embeddingDim,
    });

    // LLM wrapper for memory store (conflict detection etc.)
    const llmWrapper = async (prompt: string): Promise<string> => {
      return this.callLlm(prompt);
    };

    this.memoryStore = new MemoryStore(this.db, embedder, llmWrapper);
    this.llmComplete = config.llmComplete || null;
  }

  // ===== Lifecycle =====

  async bootstrap(params: { sessionId: string; sessionKey?: string; sessionFile: string }): Promise<BootstrapResult> {
    // Ensure tables exist (already done in constructor via migrate)
    const msgCount = this.db.getMessageCount(params.sessionId);
    return {
      bootstrapped: true,
      importedMessages: msgCount,
    };
  }

  async ingest(params: { sessionId: string; sessionKey?: string; message: AgentMessage; isHeartbeat?: boolean }): Promise<IngestResult> {
    const { sessionId, message } = params;
    if (!message.content) return { ingested: false };

    const content = typeof message.content === 'string'
      ? message.content
      : JSON.stringify(message.content);

    // BUG #4 FIX: Use db method for atomic seq generation
    const seq = this.db.getNextMessageSeq(sessionId);
    const tokenCount = estimateMessageTokens({ role: message.role, content: message.content });

    this.db.insertMessage(sessionId, seq, message.role, content, tokenCount);
    return { ingested: true };
  }

  async ingestBatch(params: { sessionId: string; sessionKey?: string; messages: AgentMessage[]; isHeartbeat?: boolean }): Promise<IngestBatchResult> {
    let count = 0;
    for (const msg of params.messages) {
      const result = await this.ingest({ sessionId: params.sessionId, message: msg, isHeartbeat: params.isHeartbeat });
      if (result.ingested) count++;
    }
    return { ingestedCount: count };
  }

  // ===== Context Assembly =====

  async assemble(params: {
    sessionId: string;
    sessionKey?: string;
    messages: AgentMessage[];
    tokenBudget?: number;
    availableTools?: Set<string>;
    model?: string;
    prompt?: string;
  }): Promise<AssembleResult> {
    const { sessionId, tokenBudget = 128000 } = params;
    
    // 1. Get fresh tail (recent raw messages)
    const freshTail = this.db.getMessages(sessionId, this.config.freshTailCount);
    const freshTailTokens = freshTail.reduce((sum, m) => sum + m.token_count, 0);
    
    // 2. Get summaries
    const summaries = this.db.getSummaries(sessionId);
    
    // Token budget allocation: 60% reserved for fresh tail, rest for summaries + memories
    const reservedForTail = Math.floor(tokenBudget * 0.6);
    const availableForContext = Math.max(0, tokenBudget - freshTailTokens - reservedForTail);

    // 3. Get relevant cross-session memories
    let memoryBlock = '';
    let memoryTokens = 0;
    try {
      const memories = await this.memoryStore.getRelevantMemories(
        freshTail.map(m => ({ role: m.role, content: m.content })),
        5,
      );
      if (memories.length > 0) {
        memoryBlock = '## Relevant memories from previous sessions:\n' +
          memories.map(m => `- [${m.type}] ${m.title}: ${m.content}`).join('\n');
        memoryTokens = estimateTokens(memoryBlock);
      }
    } catch (err) {
      console.warn('mini-lcm: memory retrieval failed:', err);
    }

    // Trim summaries to fit remaining budget (newest first, drop oldest if needed)
    let runningTokens = memoryTokens;
    const trimmedSummaries: typeof summaries = [];
    for (const summary of summaries.reverse()) {
      if (runningTokens + summary.token_count > availableForContext) break;
      trimmedSummaries.unshift(summary);
      runningTokens += summary.token_count;
    }
    const summaryTokens = trimmedSummaries.reduce((sum, s) => sum + s.token_count, 0);

    // 4. Build assembled messages
    const assembled: AgentMessage[] = [];

    // Add memory block as system message if present
    if (memoryBlock) {
      assembled.push({
        role: 'system',
        content: memoryBlock,
      } as AgentMessage);
    }

    // Add summaries as system context
    if (trimmedSummaries.length > 0) {
      const summaryText = '## Previous conversation context (compressed):\n' +
        trimmedSummaries.map(s => s.content).join('\n\n');
      assembled.push({
        role: 'system',
        content: summaryText,
      } as AgentMessage);
    }

    // Add fresh tail messages
    for (const msg of freshTail) {
      assembled.push({
        role: msg.role as any,
        content: msg.content,
      } as AgentMessage);
    }

    const totalTokens = memoryTokens + summaryTokens + freshTailTokens;

    return {
      messages: assembled,
      estimatedTokens: totalTokens,
    };
  }

  // ===== Compaction =====

  async compact(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
  }): Promise<CompactResult> {
    const { sessionId, tokenBudget = 128000, force = false } = params;
    
    const totalTokens = this.db.getMessageTokenSum(sessionId);
    const threshold = tokenBudget * this.config.contextThreshold;
    
    if (!force && totalTokens < threshold) {
      return { ok: true, compacted: false, reason: 'below threshold' };
    }

    // BUG #3 FIX: Only compress messages after last compacted seq
    const compactedUpto = this.db.getCompactedUptoSeq(sessionId);
    const allMessages = this.db.getMessages(sessionId);
    const freshCount = this.config.freshTailCount;

    if (allMessages.length <= freshCount) {
      return { ok: true, compacted: false, reason: 'not enough messages' };
    }

    // Only compress messages that are: after compactedUpto AND outside fresh tail
    const compressCandidates = allMessages.filter(m => m.seq > compactedUpto);
    const toCompress = compressCandidates.slice(0, Math.max(0, compressCandidates.length - freshCount));

    if (toCompress.length === 0) {
      return { ok: true, compacted: false, reason: 'nothing new to compress' };
    }

    const tokensBefore = totalTokens;
    
    try {
      // Build compression prompt
      const conversationText = toCompress
        .map(m => `[${m.role}] ${m.content}`)
        .join('\n');

      const prompt = `Compress the following conversation into a concise summary. Preserve all important information, decisions, facts, and context. Use the same language as the original conversation.

Conversation:
${conversationText}

Summary:`;

      // Call LLM to generate summary
      const summary = await this.callLlm(prompt);
      const summaryTokens = estimateTokens(summary);
      
      // Store summary
      this.db.insertSummary(
        sessionId,
        summary,
        summaryTokens,
        toCompress[0].created_at,
        toCompress[toCompress.length - 1].created_at,
        toCompress[0].seq,
        toCompress[toCompress.length - 1].seq,
      );

      // BUG #3 FIX: Track last compacted seq so we don't re-compress
      this.db.setCompactedUptoSeq(sessionId, toCompress[toCompress.length - 1].seq);

      // Extract and store memories from compressed content
      await this.extractMemories(sessionId, conversationText);

      const tokensAfter = this.db.getMessageTokenSum(sessionId) + this.db.getSummaryTokenSum(sessionId);

      return {
        ok: true,
        compacted: true,
        result: {
          summary,
          tokensBefore,
          tokensAfter,
        },
      };
    } catch (err) {
      return {
        ok: false,
        compacted: false,
        reason: `compaction failed: ${err}`,
      };
    }
  }

  // ===== Post-turn =====

  async afterTurn(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    messages: AgentMessage[];
    tokenBudget?: number;
  }): Promise<void> {
    // Check if compaction is needed
    const totalTokens = this.db.getMessageTokenSum(params.sessionId);
    const budget = params.tokenBudget || 128000;
    const threshold = budget * this.config.contextThreshold;

    if (totalTokens >= threshold) {
      await this.compact({
        sessionId: params.sessionId,
        sessionKey: params.sessionKey,
        sessionFile: params.sessionFile,
        tokenBudget: budget,
      });
    }
  }

  // ===== Memory extraction =====

  private async extractMemories(sessionId: string, text: string): Promise<void> {
    try {
      const prompt = `Analyze this conversation and extract important memories worth remembering across sessions.

For each memory, provide:
- type: one of "decision", "learning", "config", "bugfix", "preference", "fact"
- title: one-line title
- content: detailed description
- concepts: array of relevant tags
- importance: 1-5 (5 = must remember forever)
- confidence: 0-1 (how certain are you this was actually said/decided, not inferred)
- evidence: quote the exact text from the conversation that supports this memory

RULES:
- ONLY extract things explicitly stated or decided in the conversation
- DO NOT infer or assume things not directly mentioned
- DO NOT extract API keys, tokens, passwords, or credentials
- If confidence < 0.7, do not include the memory
- Evidence must be a direct quote from the conversation

Return a JSON array. If nothing worth remembering, return [].

Conversation:
${text}

Memories (JSON array):`;

      const response = await this.callLlm(prompt);

      // Try to parse JSON from response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (!jsonMatch) return;

      const memories = JSON.parse(jsonMatch[0]);
      if (!Array.isArray(memories)) return;

      for (const mem of memories) {
        if (mem.type && mem.title && mem.content) {
          const result = await this.memoryStore.store({
            type: mem.type || 'general',
            title: mem.title,
            content: mem.content,
            concepts: Array.isArray(mem.concepts) ? mem.concepts : [],
            sourceSession: sessionId,
            importance: Math.min(5, Math.max(1, mem.importance || 3)),
            confidence: typeof mem.confidence === 'number' ? mem.confidence : 0.8,
            evidence: mem.evidence || null,
            verified: false,
          });

          if (result === null) {
            console.warn('mini-lcm: memory blocked by safety filter:', mem.title?.slice(0, 50));
          }
        }
      }
    } catch (err) {
      // Non-fatal: memory extraction is best-effort
      console.warn('mini-lcm: memory extraction failed:', err);
    }
  }

  // ===== LLM calling =====

  private async callLlm(prompt: string): Promise<string> {
    if (!this.llmComplete) {
      throw new Error('LLM not configured');
    }

    const result = await this.llmComplete({
      messages: [{ role: 'user', content: prompt }],
      model: this.config.summaryModel,
    });

    if (result?.content) {
      return typeof result.content === 'string' ? result.content : JSON.stringify(result.content);
    }
    if (result?.text) return result.text;
    if (typeof result === 'string') return result;
    
    throw new Error('Unexpected LLM response format');
  }

  // ===== Cleanup =====

  async dispose(): Promise<void> {
    this.db.close();
  }
}
