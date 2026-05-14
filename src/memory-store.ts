/**
 * Cross-session memory store with hybrid retrieval
 * 
 * Combines:
 * 1. Vector search (semantic similarity)
 * 2. FTS5 (keyword matching)
 * 3. Time-based ranking (freshness)
 * 
 * Safety:
 * - Hallucination prevention: evidence anchoring + confidence scoring
 * - Deduplication: vector similarity + content hash
 * - Poison prevention: sensitive data filtering + trust decay + conflict detection
 */

import { createHash } from 'crypto';
import { MiniLcmDb, MemoryRow } from './db.js';
import { EmbeddingProvider, cosineSimilarity, vectorToBuffer, bufferToVector } from './embedding.js';

// ===== Safety: Sensitive data patterns =====
// NOTE: No 'g' flag — RegExp.test() with 'g' has lastIndex state副作用
const SENSITIVE_PATTERNS: RegExp[] = [
  /(?:api[_-]?key|token|secret|password|credential|bearer)\s*[:=]\s*\S+/i,
  /(?:sk-|pk-|ghp_|gho_|github_pat_|tvly-)\S+/,
  /-----BEGIN.*(?:KEY|CERTIFICATE|PRIVATE)-----/,
  /(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?/,  // IP addresses
  /Bearer\s+\S+/i,
];

const FORBIDDEN_CONCEPTS = new Set([
  'api_key', 'password', 'token', 'secret', 'credential',
  'private_key', 'auth_token', 'access_token',
]);

const DEDUP_THRESHOLD = 0.85;   // Above this similarity → merge
const CONFLICT_THRESHOLD = 0.7; // Check for conflict above this
const MIN_CONFIDENCE = 0.7;     // Minimum confidence to store
const TRUST_DECAY_DAYS = 90;    // Trust decays over 90 days
const MIN_TRUST_SCORE = 0.3;    // Minimum trust to return in search

export interface SearchResult {
  memory: MemoryRow;
  score: number;
  trustScore: number;
  source: 'vector' | 'fts' | 'hybrid';
}

export interface MemoryParams {
  type: string;
  title: string;
  content: string;
  concepts: string[];
  sourceSession: string;
  importance: number;
  confidence?: number;       // 0-1, LLM's confidence this is real
  evidence?: string;         // Quote from original text
  verified?: boolean;        // User confirmed?
}

export class MemoryStore {
  private db: MiniLcmDb;
  private embedder: EmbeddingProvider;
  private llmComplete: ((prompt: string) => Promise<string>) | null;

  // FIX #6: Simple LRU cache for query embeddings (TTL 5 min)
  private embedCache = new Map<string, { vector: Float32Array; ts: number }>();
  private readonly EMBED_CACHE_TTL = 5 * 60 * 1000;
  private readonly EMBED_CACHE_MAX = 50;

  // Weights for hybrid scoring
  private readonly W_VECTOR = 0.4;
  private readonly W_FTS = 0.3;
  private readonly W_TIME = 0.3;

  constructor(db: MiniLcmDb, embedder: EmbeddingProvider, llmComplete?: (prompt: string) => Promise<string>) {
    this.db = db;
    this.embedder = embedder;
    this.llmComplete = llmComplete || null;
  }

  // ===== Safety: Sensitive data filtering =====

  // FIX #6: Cached embed to avoid repeated API calls
  private async cachedEmbed(text: string): Promise<Float32Array> {
    const now = Date.now();
    const cached = this.embedCache.get(text);
    if (cached && (now - cached.ts) < this.EMBED_CACHE_TTL) {
      return cached.vector;
    }
    const vector = await this.embedder.embed(text);
    // Evict oldest if over limit
    if (this.embedCache.size >= this.EMBED_CACHE_MAX) {
      const oldest = this.embedCache.keys().next().value;
      if (oldest) this.embedCache.delete(oldest);
    }
    this.embedCache.set(text, { vector, ts: now });
    return vector;
  }

  private hasSensitiveData(text: string): boolean {
    return SENSITIVE_PATTERNS.some(p => p.test(text));
  }

  private sanitizeConcepts(concepts: string[]): string[] {
    return concepts.filter(c => !FORBIDDEN_CONCEPTS.has(c.toLowerCase()));
  }

  private contentHash(content: string): string {
    return createHash('sha256').update(content.trim()).digest('hex').slice(0, 16);
  }

  // ===== Safety: Trust score calculation =====

  calculateTrustScore(memory: MemoryRow): number {
    const age = Date.now() - new Date(memory.created_at).getTime();
    const maxAgeMs = TRUST_DECAY_DAYS * 24 * 60 * 60 * 1000;

    // Base score from importance
    let score = memory.importance / 5;

    // Time decay
    score *= Math.max(0.1, 1 - (age / maxAgeMs));

    // Recall count boost
    const recallCount = memory.recall_count || 0;
    score *= Math.min(1.5, 1 + recallCount * 0.1);

    // Unverified penalty
    if (!memory.verified) score *= 0.7;

    // Confidence penalty
    score *= (memory.confidence ?? 1.0);

    return Math.min(1, Math.max(0, score));
  }

  // ===== Core: Store with full safety pipeline =====

  async store(params: MemoryParams): Promise<number | null> {
    // FIX #8: Only check content (not title) for sensitive data to reduce false positives
    if (this.hasSensitiveData(params.content)) {
      console.warn('mini-lcm: blocked memory with sensitive data:', params.title.slice(0, 50));
      return null;
    }

    // Step 2: Sanitize concepts
    const concepts = this.sanitizeConcepts(params.concepts);

    // Step 3: Confidence gate
    const confidence = params.confidence ?? 1.0;
    if (confidence < MIN_CONFIDENCE) {
      console.warn('mini-lcm: blocked low-confidence memory:', params.title.slice(0, 50), 'confidence:', confidence);
      return null;
    }

    // Step 4: Dedup via vector similarity
    let vector: Float32Array | null = null;
    const textForEmbedding = `${params.title}\n${params.content}\n${concepts.join(' ')}`;
    try {
      vector = await this.embedder.embed(textForEmbedding);
      const existing = await this.findSimilarMemories(vector, DEDUP_THRESHOLD);

      if (existing.length > 0) {
        // Merge: keep longer content, boost importance, increment recall_count
        const best = existing[0];
        const longerContent = params.content.length > best.memory.content.length
          ? params.content : best.memory.content;
        const mergedTitle = params.title.length > best.memory.title.length
          ? params.title : best.memory.title;
        const mergedImportance = Math.min(5, best.memory.importance + 1);

        // BUG #2 FIX: Update embedding, content_hash, concepts, confidence on merge
        const newHash = this.contentHash(longerContent);
        const mergedConcepts = JSON.stringify([
          ...new Set([
            ...JSON.parse(best.memory.concepts || '[]'),
            ...params.concepts,
          ]),
        ]);
        const mergedConfidence = Math.max(
          best.memory.confidence ?? 1.0,
          params.confidence ?? 1.0,
        );

        // Atomic update: content + metadata + embedding in one SQL
        this.db.db.prepare(`
          UPDATE memories SET
            content = ?,
            title = ?,
            importance = ?,
            concepts = ?,
            confidence = ?,
            content_hash = ?,
            recall_count = COALESCE(recall_count, 0) + 1,
            last_recalled = datetime('now')
          WHERE id = ?
        `).run(longerContent, mergedTitle, mergedImportance,
          mergedConcepts, mergedConfidence, newHash, best.memory.id);

        // Update embedding for merged content
        if (vector) {
          try {
            this.db.storeEmbedding('memory', best.memory.id, vectorToBuffer(vector), 'text-embedding-v3');
          } catch (err) {
            console.warn('mini-lcm: embedding update on merge failed:', err);
          }
        }

        return best.memory.id;
      }
    } catch (err) {
      console.warn('mini-lcm: embedding/dedup failed, storing without vector:', err);
    }

    // Step 5: Conflict detection
    if (this.llmComplete && vector) {
      try {
        const conflicts = await this.findSimilarMemories(vector, CONFLICT_THRESHOLD);
        for (const c of conflicts) {
          const isConflict = await this.checkConflict(c.memory.content, params.content);
          if (isConflict) {
            console.warn('mini-lcm: conflict detected with memory', c.memory.id, ':', c.memory.title);
            // Flag for review instead of blocking
            this.db.db.prepare(`
              UPDATE memories SET conflict_flag = 1 WHERE id = ?
            `).run(c.memory.id);
          }
        }
      } catch (err) {
        console.warn('mini-lcm: conflict check failed:', err);
      }
    }

    // Step 6: Store the memory
    const memoryId = this.db.insertMemory(
      params.type,
      params.title,
      params.content,
      concepts,
      params.sourceSession,
      params.importance,
    );

    // Step 7: Store embedding
    if (vector) {
      try {
        this.db.storeEmbedding('memory', memoryId, vectorToBuffer(vector), 'text-embedding-v3');
      } catch (err) {
        console.warn('mini-lcm: embedding storage failed:', err);
      }
    }

    // Step 8: Store metadata (confidence, evidence, verified)
    // FIX #4: content_hash already computed in Step 4 dedup path; reuse here
    const finalHash = vector ? undefined : this.contentHash(params.content);
    this.db.db.prepare(`
      UPDATE memories SET
        confidence = ?,
        evidence = ?,
        verified = ?,
        content_hash = COALESCE(?, content_hash),
        recall_count = 0
      WHERE id = ?
    `).run(
      confidence,
      params.evidence || null,
      params.verified ? 1 : 0,
      finalHash,
      memoryId,
    );

    return memoryId;
  }

  // ===== Search with trust scoring =====

  async search(query: string, limit = 10): Promise<SearchResult[]> {
    const results = new Map<number, { memory: MemoryRow; vectorScore: number; ftsScore: number; timeScore: number }>();

    // 1. Vector search
    try {
      const queryVector = await this.cachedEmbed(query);
      const allEmbeddings = this.db.getAllMemoryEmbeddings();

      const vectorScores: { id: number; score: number }[] = [];
      for (const emb of allEmbeddings) {
        const vec = bufferToVector(emb.vector);
        const score = cosineSimilarity(queryVector, vec);
        if (score > 0.3) {
          vectorScores.push({ id: emb.id, score });
        }
      }
      vectorScores.sort((a, b) => b.score - a.score);

      for (const vs of vectorScores.slice(0, limit * 2)) {
        if (!results.has(vs.id)) {
          const memory = this.db.db.prepare('SELECT * FROM memories WHERE id = ?').get(vs.id) as MemoryRow;
          if (memory) {
            results.set(vs.id, { memory, vectorScore: vs.score, ftsScore: 0, timeScore: 0 });
          }
        }
      }
    } catch (err) {
      console.warn('mini-lcm: vector search failed:', err);
    }

    // 2. FTS search
    try {
      const ftsResults = this.db.searchMemoriesFts(query, limit * 2);
      const maxRank = Math.max(...ftsResults.map(r => Math.abs(r.rank)), 1);

      for (const fr of ftsResults) {
        const normalizedScore = 1 - (Math.abs(fr.rank) / maxRank);
        const existing = results.get(fr.id);
        if (existing) {
          existing.ftsScore = normalizedScore;
        } else {
          results.set(fr.id, { memory: fr, vectorScore: 0, ftsScore: normalizedScore, timeScore: 0 });
        }
      }
    } catch (err) {
      console.warn('mini-lcm: FTS search failed:', err);
    }

    // 3. Time scoring
    const now = Date.now();
    const maxAge = TRUST_DECAY_DAYS * 24 * 60 * 60 * 1000;
    for (const [, result] of results) {
      const created = new Date(result.memory.created_at).getTime();
      const age = now - created;
      result.timeScore = Math.max(0, 1 - (age / maxAge));
    }

    // 4. Combine scores + trust scoring
    const combined: SearchResult[] = [];
    for (const [, result] of results) {
      const hybridScore = this.W_VECTOR * result.vectorScore
                        + this.W_FTS * result.ftsScore
                        + this.W_TIME * result.timeScore;

      const trustScore = this.calculateTrustScore(result.memory);

      // Filter out low-trust memories
      if (trustScore < MIN_TRUST_SCORE) continue;

      // Final score = hybrid × trust
      const finalScore = hybridScore * trustScore;

      combined.push({
        memory: result.memory,
        score: finalScore,
        trustScore,
        source: result.vectorScore > 0 && result.ftsScore > 0 ? 'hybrid'
              : result.vectorScore > 0 ? 'vector' : 'fts',
      });
    }

    combined.sort((a, b) => b.score - a.score);
    return combined.slice(0, limit);
  }

  /**
   * Get memories relevant to the current context
   * Used during context assembly
   */
  async getRelevantMemories(messages: { role: string; content: string }[], limit = 5): Promise<MemoryRow[]> {
    const userMessages = messages
      .filter(m => m.role === 'user')
      .slice(-3)
      .map(m => typeof m.content === 'string' ? m.content : '')
      .filter(Boolean);

    if (userMessages.length === 0) {
      return this.db.getImportantMemories(limit);
    }

    const query = userMessages.join(' ');
    const results = await this.search(query, limit);

    if (results.length === 0) {
      return this.db.getImportantMemories(limit);
    }

    // Update recall_count for retrieved memories
    for (const r of results) {
      this.db.db.prepare(`
        UPDATE memories SET
          recall_count = COALESCE(recall_count, 0) + 1,
          last_recalled = datetime('now')
        WHERE id = ?
      `).run(r.memory.id);
    }

    return results.map(r => r.memory);
  }

  // ===== Safety helpers =====

  private async findSimilarMemories(vector: Float32Array, threshold: number): Promise<{ memory: MemoryRow; similarity: number }[]> {
    const allEmbeddings = this.db.getAllMemoryEmbeddings();
    const results: { memory: MemoryRow; similarity: number }[] = [];

    for (const emb of allEmbeddings) {
      const vec = bufferToVector(emb.vector);
      const sim = cosineSimilarity(vector, vec);
      if (sim >= threshold) {
        const memory = this.db.db.prepare('SELECT * FROM memories WHERE id = ?').get(emb.id) as MemoryRow;
        if (memory) {
          results.push({ memory, similarity: sim });
        }
      }
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results;
  }

  private async checkConflict(existingContent: string, newContent: string): Promise<boolean> {
    if (!this.llmComplete) return false;

    const prompt = `You are checking if two memories contradict each other.

Memory A: ${existingContent}
Memory B: ${newContent}

Do these memories contain contradictory information? Consider:
- Different values for the same setting
- Opposite preferences
- Contradictory facts

Answer ONLY "yes" or "no".`;

    try {
      const answer = await this.llmComplete(prompt);
      return answer.trim().toLowerCase().startsWith('yes');
    } catch {
      return false;
    }
  }
}
