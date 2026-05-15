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
    version: '2026.05.15.004',
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
    // FIX: Import historical messages from session file if DB is empty
    // Handles OpenClaw's actual JSONL format: { type: 'message', message: { role, content } }
    let importedMessages = 0;
    const existingCount = this.db.getMessageCount(params.sessionId);

    if (existingCount === 0 && params.sessionFile) {
      try {
        const { readFileSync, existsSync } = await import('fs');
        if (existsSync(params.sessionFile)) {
          const lines = readFileSync(params.sessionFile, 'utf-8').split('\n').filter(Boolean);
          for (const line of lines) {
            try {
              const entry = JSON.parse(line);

              // Handle OpenClaw session format: { type: 'message', message: { role, content } }
              if (entry.type === 'message' && entry.message?.role) {
                const msg = entry.message;
                let textContent: string;

                if (typeof msg.content === 'string') {
                  textContent = msg.content;
                } else if (Array.isArray(msg.content)) {
                  // Extract text from content parts, skip thinking/tool_use
                  textContent = msg.content
                    .filter((p: any) => p.type === 'text' && p.text)
                    .map((p: any) => p.text)
                    .join('\n');
                } else {
                  textContent = JSON.stringify(msg.content);
                }

                if (textContent && textContent.length > 0) {
                  const seq = this.db.getNextMessageSeq(params.sessionId);
                  const tokenCount = estimateTokens(textContent);
                  this.db.insertMessage(params.sessionId, seq, msg.role, textContent, tokenCount);
                  importedMessages++;
                }
              }
              // Also handle simple format: { role, content }
              else if (entry.role && entry.content) {
                const content = typeof entry.content === 'string'
                  ? entry.content : JSON.stringify(entry.content);
                if (content.length > 0) {
                  const seq = this.db.getNextMessageSeq(params.sessionId);
                  const tokenCount = estimateMessageTokens({ role: entry.role, content: entry.content });
                  this.db.insertMessage(params.sessionId, seq, entry.role, content, tokenCount);
                  importedMessages++;
                }
              }
            } catch {}
          }
        }
      } catch (err) {
        console.warn('mini-lcm: bootstrap import failed:', err);
      }
    }

    return {
      bootstrapped: true,
      importedMessages,
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

    // FIX: Use OpenClaw-provided messages as the primary source
    // DB is used for summaries, memories, and historical context only
    const incomingMessages = params.messages || [];

    // 1. Also try to get any additional messages from DB (for cross-session continuity)
    const dbMessages = this.db.getMessages(sessionId, this.config.freshTailCount);

    // 2. Build the fresh tail: prefer incoming messages, supplement with DB
    // Deduplicate by content to avoid repeats
    let freshTail: { role: string; content: string; token_count: number }[] = [];

    if (incomingMessages.length > 0) {
      // Use incoming messages as the source of truth
      freshTail = incomingMessages.map(m => {
        const content = typeof m.content === 'string'
          ? m.content
          : JSON.stringify(m.content);
        return {
          role: m.role,
          content,
          token_count: estimateMessageTokens({ role: m.role, content: m.content }),
        };
      });

      // Also ingest these into DB for future reference
      for (const msg of incomingMessages) {
        try {
          const content = typeof msg.content === 'string'
            ? msg.content
            : JSON.stringify(msg.content);
          // Check if already in DB (by content match on last few)
          const existing = this.db.getMessages(sessionId, 5);
          const alreadyExists = existing.some(e => e.content === content);
          if (!alreadyExists && content.length > 0) {
            const seq = this.db.getNextMessageSeq(sessionId);
            const tokenCount = estimateMessageTokens({ role: msg.role, content: msg.content });
            this.db.insertMessage(sessionId, seq, msg.role, content, tokenCount);
          }
        } catch {}
      }
    } else if (dbMessages.length > 0) {
      // Fallback to DB messages if no incoming messages
      freshTail = dbMessages.map(m => ({
        role: m.role,
        content: m.content,
        token_count: m.token_count,
      }));
    }

    const freshTailTokens = freshTail.reduce((sum, m) => sum + m.token_count, 0);

    // 3. Get summaries
    const summaries = this.db.getSummaries(sessionId);

    // Token budget allocation: fresh tail gets what it needs, rest for summaries + memories
    const availableForContext = Math.max(0, tokenBudget - freshTailTokens);

    // 4. Get relevant cross-session memories
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
    for (const summary of [...summaries].reverse()) {
      if (runningTokens + summary.token_count > availableForContext) break;
      trimmedSummaries.unshift(summary);
      runningTokens += summary.token_count;
    }
    const summaryTokens = trimmedSummaries.reduce((sum, s) => sum + s.token_count, 0);

    // 5. Build assembled messages
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

      // FIX: tokensAfter = fresh tail + all summaries (getSummaryTokenSum already includes the new one)
      const freshTailMsgs = this.db.getMessages(sessionId, this.config.freshTailCount);
      const freshTailTokens = freshTailMsgs.reduce((sum, m) => sum + m.token_count, 0);
      const allSummaryTokens = this.db.getSummaryTokenSum(sessionId);
      const tokensAfter = freshTailTokens + allSummaryTokens;

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

      // FIX #5: Try multiple JSON extraction strategies
      let memories: any[] | null = null;

      // Strategy 1: fenced code block
      const fencedMatch = response.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      if (fencedMatch) {
        try { memories = JSON.parse(fencedMatch[1]); } catch {}
      }

      // Strategy 2: raw JSON array
      if (!memories) {
        const rawMatch = response.match(/\[[\s\S]*\]/);
        if (rawMatch) {
          try { memories = JSON.parse(rawMatch[0]); } catch {}
        }
      }

      if (!memories || !Array.isArray(memories)) return;

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

    // FIX #5: Add timeout with AbortController to actually cancel the request
    const TIMEOUT_MS = 30000;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const result = await this.llmComplete({
        messages: [{ role: 'user', content: prompt }],
        model: this.config.summaryModel,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (result?.content) {
        return typeof result.content === 'string' ? result.content : JSON.stringify(result.content);
      }
      if (result?.text) return result.text;
      if (typeof result === 'string') return result;

      throw new Error('Unexpected LLM response format');
    } catch (err: any) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        throw new Error('LLM call timed out');
      }
      throw err;
    }
  }

  // ===== Cleanup =====

  async dispose(): Promise<void> {
    this.db.close();
  }
}
