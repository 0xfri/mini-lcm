/**
 * Integration test: Simulates OpenClaw's actual lifecycle
 * 
 * Flow: bootstrap → ingest → assemble → afterTurn
 * Tests the exact scenario that caused the "greeting-only" bug
 */

import { MiniLcmEngine } from './dist/context-engine.js';
import { writeFileSync, mkdirSync, rmSync } from 'fs';
import { join } from 'path';

const TEST_DIR = '/tmp/mini-lcm-test';
const TEST_DB = join(TEST_DIR, 'test.db');
const TEST_SESSION = join(TEST_DIR, 'session.jsonl');

let passed = 0;
let failed = 0;

function assert(condition, name) {
  if (condition) {
    console.log(`  ✅ ${name}`);
    passed++;
  } else {
    console.log(`  ❌ ${name}`);
    failed++;
  }
}

// Setup
mkdirSync(TEST_DIR, { recursive: true });

// Create a mock session file in OpenClaw's actual format
const sessionLines = [
  JSON.stringify({ type: 'message', id: '1', timestamp: '2026-05-15T00:00:00Z', message: { role: 'user', content: [{ type: 'text', text: '哈喽？' }] } }),
  JSON.stringify({ type: 'message', id: '2', timestamp: '2026-05-15T00:00:01Z', message: { role: 'assistant', content: [{ type: 'text', text: '嘿闫安，星期五在线 🦞' }] } }),
  JSON.stringify({ type: 'message', id: '3', timestamp: '2026-05-15T00:00:10Z', message: { role: 'user', content: [{ type: 'text', text: '1+1' }] } }),
  JSON.stringify({ type: 'message', id: '4', timestamp: '2026-05-15T00:00:11Z', message: { role: 'assistant', content: [{ type: 'text', text: '2' }] } }),
  JSON.stringify({ type: 'message', id: '5', timestamp: '2026-05-15T00:00:20Z', message: { role: 'user', content: [{ type: 'text', text: '去GitHub搜SenseNova-U1' }] } }),
  JSON.stringify({ type: 'message', id: '6', timestamp: '2026-05-15T00:00:21Z', message: { role: 'assistant', content: [{ type: 'thinking', thinking: 'let me search...' }, { type: 'text', text: '我来搜一下' }] } }),
  // Also include a tool message (should be skipped)
  JSON.stringify({ type: 'message', id: '7', timestamp: '2026-05-15T00:00:22Z', message: { role: 'tool', content: [{ type: 'text', text: 'search results...' }] } }),
];
writeFileSync(TEST_SESSION, sessionLines.join('\n'));

// ===== Test 1: Bootstrap imports session messages =====
console.log('\n🧪 Test 1: Bootstrap - import session file');

async function test1() {
  const engine = new MiniLcmEngine({
    dbPath: TEST_DB,
    apiKey: 'test-key-not-real',
  });

  const result = await engine.bootstrap({
    sessionId: 'test-session-1',
    sessionFile: TEST_SESSION,
  });

  assert(result.bootstrapped === true, 'bootstrapped = true');
  assert(result.importedMessages === 6, `importedMessages = 6 (skipped tool message, got ${result.importedMessages})`);

  // Verify messages are in DB by assembling with empty incoming
  const assembled = await engine.assemble({
    sessionId: 'test-session-1',
    messages: [],
    tokenBudget: 128000,
  });

  assert(assembled.messages.length > 0, `assemble() returns messages from DB (got ${assembled.messages.length})`);

  // Check content is correct
  const userMsgs = assembled.messages.filter(m => m.role === 'user');
  assert(userMsgs.length === 3, `3 user messages imported (got ${userMsgs.length})`);

  const hasHamlou = userMsgs.some(m => m.content.includes('哈喽'));
  assert(hasHamlou, 'contains "哈喽？" message');

  const hasSearch = userMsgs.some(m => m.content.includes('SenseNova'));
  assert(hasSearch, 'contains "去GitHub搜SenseNova-U1" message');

  // Verify thinking was filtered out
  const hasThinking = assembled.messages.some(m => m.content.includes('let me search'));
  assert(!hasThinking, 'thinking content was filtered out');

  await engine.dispose();
}

// ===== Test 2: Assemble uses incoming messages (the main bug fix) =====
console.log('\n🧪 Test 2: Assemble - uses OpenClaw-provided messages');

async function test2() {
  // Fresh engine with empty DB
  rmSync(TEST_DB, { force: true });
  const engine = new MiniLcmEngine({
    dbPath: TEST_DB,
    apiKey: 'test-key-not-real',
  });

  // Simulate OpenClaw calling assemble WITH messages (no prior ingest)
  const assembled = await engine.assemble({
    sessionId: 'test-session-2',
    messages: [
      { role: 'user', content: '去GitHub搜一下SenseNova-U1' },
      { role: 'assistant', content: '我来搜一下' },
      { role: 'user', content: '搜到了吗？' },
    ],
    tokenBudget: 128000,
  });

  assert(assembled.messages.length === 3, `assemble returns all 3 incoming messages (got ${assembled.messages.length})`);

  const hasUser = assembled.messages.some(m => m.content.includes('SenseNova'));
  assert(hasUser, 'contains SenseNova message');

  const hasFollowUp = assembled.messages.some(m => m.content.includes('搜到了吗'));
  assert(hasFollowUp, 'contains follow-up message');

  // Verify messages were auto-ingested into DB
  const engine2 = new MiniLcmEngine({ dbPath: TEST_DB, apiKey: 'test' });
  const assembled2 = await engine2.assemble({
    sessionId: 'test-session-2',
    messages: [], // Empty - should fall back to DB
    tokenBudget: 128000,
  });

  assert(assembled2.messages.length === 3, `DB fallback returns auto-ingested messages (got ${assembled2.messages.length})`);

  await engine.dispose();
  await engine2.dispose();
}

// ===== Test 3: No duplicate messages on repeated assemble calls =====
console.log('\n🧪 Test 3: Dedup - no duplicates on repeated assemble');

async function test3() {
  rmSync(TEST_DB, { force: true });
  const engine = new MiniLcmEngine({
    dbPath: TEST_DB,
    apiKey: 'test-key-not-real',
  });

  const messages = [
    { role: 'user', content: '你好' },
    { role: 'assistant', content: '你好闫安' },
  ];

  // Call assemble 3 times with same messages
  for (let i = 0; i < 3; i++) {
    await engine.assemble({
      sessionId: 'test-session-3',
      messages,
      tokenBudget: 128000,
    });
  }

  // Check DB has exactly 2 messages, not 6
  const db = engine.db;
  const count = db.getMessageCount('test-session-3');
  assert(count === 2, `DB has exactly 2 messages after 3 assemble calls (got ${count})`);

  await engine.dispose();
}

// ===== Test 4: Empty messages + empty DB = empty result =====
console.log('\n🧪 Test 4: Edge case - empty everything');

async function test4() {
  rmSync(TEST_DB, { force: true });
  const engine = new MiniLcmEngine({
    dbPath: TEST_DB,
    apiKey: 'test-key-not-real',
  });

  const assembled = await engine.assemble({
    sessionId: 'test-session-4',
    messages: [],
    tokenBudget: 128000,
  });

  assert(assembled.messages.length === 0, `empty input = empty output (got ${assembled.messages.length})`);
  assert(assembled.estimatedTokens === 0, `0 tokens for empty (got ${assembled.estimatedTokens})`);

  await engine.dispose();
}

// ===== Test 5: Content array format (OpenClaw's actual format) =====
console.log('\n🧪 Test 5: Content array format');

async function test5() {
  rmSync(TEST_DB, { force: true });
  const engine = new MiniLcmEngine({
    dbPath: TEST_DB,
    apiKey: 'test-key-not-real',
  });

  const assembled = await engine.assemble({
    sessionId: 'test-session-5',
    messages: [
      { role: 'user', content: [{ type: 'text', text: '你好世界' }] },
      { role: 'assistant', content: [{ type: 'thinking', thinking: 'thinking...' }, { type: 'text', text: '你好！' }] },
    ],
    tokenBudget: 128000,
  });

  assert(assembled.messages.length === 2, `handles content array (got ${assembled.messages.length})`);

  const userMsg = assembled.messages.find(m => m.role === 'user');
  assert(userMsg?.content === '你好世界', 'user content extracted correctly');

  const asstMsg = assembled.messages.find(m => m.role === 'assistant');
  assert(asstMsg?.content === '你好！', 'assistant text content extracted (thinking skipped)');

  await engine.dispose();
}

// ===== Test 6: Compact guard - LLM not configured =====
console.log('\n🧪 Test 6: Compact guard - no LLM');

async function test6() {
  rmSync(TEST_DB, { force: true });
  const engine = new MiniLcmEngine({
    dbPath: TEST_DB,
    apiKey: 'test-key-not-real',
    // No llmComplete provided
  });

  // Ingest enough messages to trigger compaction
  for (let i = 0; i < 100; i++) {
    await engine.ingest({
      sessionId: 'test-session-6',
      message: { role: i % 2 === 0 ? 'user' : 'assistant', content: `Message ${i} with some content to fill up the context window` },
    });
  }

  const result = await engine.compact({
    sessionId: 'test-session-6',
    sessionFile: TEST_SESSION,
    tokenBudget: 100, // Very small to force compaction
    force: true,
  });

  assert(result.ok === false, `compact fails gracefully without LLM (ok=${result.ok})`);
  assert(result.reason?.includes('LLM'), `reason mentions LLM (${result.reason})`);

  await engine.dispose();
}

// ===== Test 7: Full lifecycle simulation =====
console.log('\n🧪 Test 7: Full lifecycle - bootstrap → ingest → assemble → afterTurn');

async function test7() {
  rmSync(TEST_DB, { force: true });
  const engine = new MiniLcmEngine({
    dbPath: TEST_DB,
    apiKey: 'test-key-not-real',
  });

  // Step 1: Bootstrap
  const bootResult = await engine.bootstrap({
    sessionId: 'test-session-7',
    sessionFile: TEST_SESSION,
  });
  assert(bootResult.importedMessages === 6, `bootstrap imported 6 messages (got ${bootResult.importedMessages})`);

  // Step 2: New user message arrives
  const ingestResult = await engine.ingest({
    sessionId: 'test-session-7',
    message: { role: 'user', content: '做个自我介绍' },
  });
  assert(ingestResult.ingested, 'new message ingested');

  // Step 3: OpenClaw calls assemble with full context
  const assembled = await engine.assemble({
    sessionId: 'test-session-7',
    messages: [
      // OpenClaw passes the full conversation
      { role: 'user', content: '哈喽？' },
      { role: 'assistant', content: '嘿闫安，星期五在线 🦞' },
      { role: 'user', content: '1+1' },
      { role: 'assistant', content: '2' },
      { role: 'user', content: '去GitHub搜SenseNova-U1' },
      { role: 'assistant', content: '我来搜一下' },
      { role: 'user', content: '做个自我介绍' },
    ],
    tokenBudget: 128000,
  });

  assert(assembled.messages.length === 7, `assembled has 7 messages (got ${assembled.messages.length})`);

  const lastMsg = assembled.messages[assembled.messages.length - 1];
  assert(lastMsg.content === '做个自我介绍', 'last message is the new user message');

  // Verify no greeting-only behavior
  const userMsgs = assembled.messages.filter(m => m.role === 'user');
  assert(userMsgs.length === 4, `4 user messages total (got ${userMsgs.length})`);

  const hasSearch = assembled.messages.some(m => m.content.includes('SenseNova'));
  assert(hasSearch, 'SenseNova message preserved in context');

  await engine.dispose();
}

// Run all tests
async function runAll() {
  try {
    await test1();
    await test2();
    await test3();
    await test4();
    await test5();
    await test6();
    await test7();
  } catch (err) {
    console.error('\n💥 Test crashed:', err);
    failed++;
  }

  console.log(`\n${'='.repeat(50)}`);
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log(`${'='.repeat(50)}`);

  // Cleanup
  rmSync(TEST_DIR, { recursive: true, force: true });

  process.exit(failed > 0 ? 1 : 0);
}

runAll();
