/**
 * SQLite database with FTS5 and vector storage
 * 
 * Tables:
 * - messages: Per-session raw messages
 * - summaries: Per-session compressed summaries
 * - memories: Cross-session persistent memories
 * - memory_fts: FTS5 index for full-text search
 * - embeddings: Vector embeddings for semantic search
 */

import Database from 'better-sqlite3';
import { resolve } from 'path';
import { mkdirSync } from 'fs';
import { dirname } from 'path';

export interface MessageRow {
  id: number;
  session_id: string;
  seq: number;
  role: string;
  content: string;
  token_count: number;
  created_at: string;
}

export interface SummaryRow {
  id: number;
  session_id: string;
  content: string;
  token_count: number;
  earliest_at: string;
  latest_at: string;
  source_start_seq: number;
  source_end_seq: number;
  created_at: string;
}

export interface MemoryRow {
  id: number;
  type: string;
  title: string;
  content: string;
  concepts: string;       // JSON array
  source_session: string;
  importance: number;      // 1-5
  // FIX #9: Safety fields in interface
  confidence: number;
  evidence: string | null;
  verified: number;        // 0 or 1
  content_hash: string | null;
  recall_count: number;
  last_recalled: string | null;
  conflict_flag: number;   // 0 or 1
  created_at: string;
  expires_at: string | null;
}

export class MiniLcmDb {
  db: Database.Database;

  constructor(dbPath: string) {
    const resolvedPath = resolve(dbPath.replace(/^~/, process.env.HOME || '/root'));
    mkdirSync(dirname(resolvedPath), { recursive: true });
    
    this.db = new Database(resolvedPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.migrate();
  }

  private migrate() {
    this.db.exec(`
      -- Per-session messages
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(session_id, seq)
      );

      -- Engine state (tracks compaction progress per session)
      CREATE TABLE IF NOT EXISTS engine_state (
        session_id TEXT PRIMARY KEY,
        compacted_upto_seq INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      );

      -- Per-session summaries
      CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        earliest_at TEXT NOT NULL,
        latest_at TEXT NOT NULL,
        source_start_seq INTEGER NOT NULL,
        source_end_seq INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );

      -- Cross-session persistent memories
      CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL DEFAULT 'general',
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        concepts TEXT NOT NULL DEFAULT '[]',
        source_session TEXT,
        importance INTEGER NOT NULL DEFAULT 3 CHECK(importance BETWEEN 1 AND 5),
        embedding_id INTEGER,
        -- Safety fields
        confidence REAL NOT NULL DEFAULT 1.0,
        evidence TEXT,
        verified INTEGER NOT NULL DEFAULT 0,
        content_hash TEXT,
        recall_count INTEGER NOT NULL DEFAULT 0,
        last_recalled TEXT,
        conflict_flag INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        expires_at TEXT
      );

      -- FTS5 index for full-text search on memories
      CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        title,
        content,
        concepts,
        content=memories,
        content_rowid=id,
        tokenize='unicode61'
      );

      -- Triggers to keep FTS in sync
      CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memory_fts(rowid, title, content, concepts)
        VALUES (new.id, new.title, new.content, new.concepts);
      END;

      CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, title, content, concepts)
        VALUES ('delete', old.id, old.title, old.content, old.concepts);
      END;

      CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, title, content, concepts)
        VALUES ('delete', old.id, old.title, old.content, old.concepts);
        INSERT INTO memory_fts(rowid, title, content, concepts)
        VALUES (new.id, new.title, new.content, new.concepts);
      END;

      -- Embedding vectors
      CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ref_type TEXT NOT NULL CHECK(ref_type IN ('memory', 'message', 'summary')),
        ref_id INTEGER NOT NULL,
        vector BLOB NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(ref_type, ref_id)
      );

      -- Indexes
      CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
      CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);

      CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
      CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
      CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_embeddings_ref ON embeddings(ref_type, ref_id);
    `);
  }

  // ===== Engine state operations =====

  getCompactedUptoSeq(sessionId: string): number {
    const row = this.db.prepare('SELECT compacted_upto_seq FROM engine_state WHERE session_id = ?').get(sessionId) as { compacted_upto_seq: number } | undefined;
    return row?.compacted_upto_seq || 0;
  }

  setCompactedUptoSeq(sessionId: string, seq: number) {
    this.db.prepare(`
      INSERT INTO engine_state (session_id, compacted_upto_seq, updated_at)
      VALUES (?, ?, datetime('now'))
      ON CONFLICT(session_id) DO UPDATE SET
        compacted_upto_seq = excluded.compacted_upto_seq,
        updated_at = datetime('now')
    `).run(sessionId, seq);
  }

  getNextMessageSeq(sessionId: string): number {
    const row = this.db.prepare(
      'SELECT COALESCE(MAX(seq), 0) + 1 as next_seq FROM messages WHERE session_id = ?'
    ).get(sessionId) as { next_seq: number };
    return row.next_seq;
  }

  // ===== Message operations =====

  insertMessage(sessionId: string, seq: number, role: string, content: string, tokenCount: number) {
    this.db.prepare(`
      INSERT OR IGNORE INTO messages (session_id, seq, role, content, token_count)
      VALUES (?, ?, ?, ?, ?)
    `).run(sessionId, seq, role, content, tokenCount);
  }

  getMessages(sessionId: string, limit?: number): MessageRow[] {
    const sql = limit
      ? `SELECT id, session_id, seq, role, content, token_count, created_at FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?`
      : `SELECT id, session_id, seq, role, content, token_count, created_at FROM messages WHERE session_id = ? ORDER BY seq`;
    const rows = limit
      ? this.db.prepare(sql).all(sessionId, limit) as MessageRow[]
      : this.db.prepare(sql).all(sessionId) as MessageRow[];
    return limit ? rows.reverse() : rows;
  }

  getMessageCount(sessionId: string): number {
    const row = this.db.prepare('SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?').get(sessionId) as { cnt: number };
    return row.cnt;
  }

  getMessageTokenSum(sessionId: string, fromSeq?: number): number {
    let sql = 'SELECT COALESCE(SUM(token_count), 0) as total FROM messages WHERE session_id = ?';
    const params: any[] = [sessionId];
    if (fromSeq !== undefined) {
      sql += ' AND seq >= ?';
      params.push(fromSeq);
    }
    const row = this.db.prepare(sql).get(...params) as { total: number };
    return row.total;
  }

  // ===== Summary operations =====

  insertSummary(sessionId: string, content: string, tokenCount: number, earliestAt: string, latestAt: string, startSeq: number, endSeq: number) {
    this.db.prepare(`
      INSERT INTO summaries (session_id, content, token_count, earliest_at, latest_at, source_start_seq, source_end_seq)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(sessionId, content, tokenCount, earliestAt, latestAt, startSeq, endSeq);
  }

  getSummaries(sessionId: string): SummaryRow[] {
    return this.db.prepare('SELECT * FROM summaries WHERE session_id = ? ORDER BY earliest_at').all(sessionId) as SummaryRow[];
  }

  getSummaryTokenSum(sessionId: string): number {
    const row = this.db.prepare('SELECT COALESCE(SUM(token_count), 0) as total FROM summaries WHERE session_id = ?').get(sessionId) as { total: number };
    return row.total;
  }

  // ===== Memory operations =====

  insertMemory(type: string, title: string, content: string, concepts: string[], sourceSession: string, importance: number): number {
    const result = this.db.prepare(`
      INSERT INTO memories (type, title, content, concepts, source_session, importance)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(type, title, content, JSON.stringify(concepts), sourceSession, importance);
    return Number(result.lastInsertRowid);
  }

  searchMemoriesFts(query: string, limit = 10): (MemoryRow & { rank: number })[] {
    // Sanitize query for FTS5: wrap each word in quotes to prevent operator interpretation
    // FTS5 treats hyphens as NOT operators and can misparse column names
    const words = query.replace(/[\[\]"*:^()]/g, ' ').split(/\s+/).filter(Boolean);
    if (words.length === 0) return [];
    const sanitized = words.map(w => `"${w}"`).join(' ');

    return this.db.prepare(`
      SELECT m.*, COALESCE(bm25(memory_fts), 0) as rank
      FROM memory_fts
      JOIN memories m ON memory_fts.rowid = m.id
      WHERE memory_fts MATCH ?
        AND (m.expires_at IS NULL OR m.expires_at > datetime('now'))
      ORDER BY rank
      LIMIT ?
    `).all(sanitized, limit) as (MemoryRow & { rank: number })[];
  }

  getMemoriesByConcept(concept: string, limit = 10): MemoryRow[] {
    return this.db.prepare(`
      SELECT * FROM memories
      WHERE concepts LIKE ?
        AND (expires_at IS NULL OR expires_at > datetime('now'))
      ORDER BY importance DESC, created_at DESC
      LIMIT ?
    `).all(`%"${concept}"%`, limit) as MemoryRow[];
  }

  getRecentMemories(limit = 20): MemoryRow[] {
    return this.db.prepare(`
      SELECT * FROM memories
      WHERE expires_at IS NULL OR expires_at > datetime('now')
      ORDER BY created_at DESC
      LIMIT ?
    `).all(limit) as MemoryRow[];
  }

  getImportantMemories(limit = 10): MemoryRow[] {
    return this.db.prepare(`
      SELECT * FROM memories
      WHERE importance >= 4
        AND (expires_at IS NULL OR expires_at > datetime('now'))
      ORDER BY importance DESC, created_at DESC
      LIMIT ?
    `).all(limit) as MemoryRow[];
  }

  // ===== Embedding operations =====

  storeEmbedding(refType: string, refId: number, vector: Buffer, model: string) {
    this.db.prepare(`
      INSERT OR REPLACE INTO embeddings (ref_type, ref_id, vector, model)
      VALUES (?, ?, ?, ?)
    `).run(refType, refId, vector, model);
  }

  getEmbedding(refType: string, refId: number): { vector: Buffer; model: string } | null {
    return this.db.prepare('SELECT vector, model FROM embeddings WHERE ref_type = ? AND ref_id = ?')
      .get(refType, refId) as { vector: Buffer; model: string } | null;
  }

  getAllMemoryEmbeddings(): { id: number; vector: Buffer }[] {
    return this.db.prepare(`
      SELECT e.ref_id as id, e.vector
      FROM embeddings e
      JOIN memories m ON e.ref_id = m.id
      WHERE e.ref_type = 'memory'
        AND (m.expires_at IS NULL OR m.expires_at > datetime('now'))
    `).all() as { id: number; vector: Buffer }[];
  }

  close() {
    this.db.close();
  }
}
