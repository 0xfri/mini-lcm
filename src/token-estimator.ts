/**
 * CJK-aware token estimation
 * 
 * Problem: Default estimators treat CJK chars like ASCII (0.25 tok/char)
 * Reality: CJK chars are ~1.5 tokens each, Emoji ~2.0
 * 
 * Reference: cl100k_base encoding benchmarks
 */

// Character categories and their tokens-per-char rates
const RATES = {
  cjk: 1.5,       // Chinese, Japanese, Korean
  emoji: 2.0,     // Emoji and supplementary plane characters
  ascii: 0.25,    // ASCII / Latin
} as const;

// CJK Unified Ideographs + Extension A + Compatibility
const CJK_REGEX = /[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u{20000}-\u{2A6DF}]/u;

// Emoji: supplementary plane + common emoji ranges
const EMOJI_REGEX = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2702}-\u{27B0}]/u;

export function estimateTokens(text: string): number {
  if (!text) return 0;
  
  let tokens = 0;
  let asciiCount = 0;
  
  for (const char of text) {
    if (CJK_REGEX.test(char)) {
      // Flush ASCII count first
      if (asciiCount > 0) {
        tokens += Math.ceil(asciiCount * RATES.ascii);
        asciiCount = 0;
      }
      tokens += RATES.cjk;
    } else if (EMOJI_REGEX.test(char)) {
      if (asciiCount > 0) {
        tokens += Math.ceil(asciiCount * RATES.ascii);
        asciiCount = 0;
      }
      tokens += RATES.emoji;
    } else {
      asciiCount++;
    }
  }
  
  // Flush remaining ASCII
  if (asciiCount > 0) {
    tokens += Math.ceil(asciiCount * RATES.ascii);
  }
  
  return Math.ceil(tokens);
}

/**
 * Estimate token count for a message object
 */
export function estimateMessageTokens(msg: { role: string; content: string | any[] }): number {
  if (typeof msg.content === 'string') {
    return estimateTokens(msg.content);
  }
  if (Array.isArray(msg.content)) {
    let total = 0;
    for (const part of msg.content) {
      if (part.type === 'text' && part.text) {
        total += estimateTokens(part.text);
      } else if (part.type === 'tool_use' && part.input) {
        total += estimateTokens(JSON.stringify(part.input));
      } else if (part.type === 'tool_result' && part.content) {
        total += estimateTokens(typeof part.content === 'string' ? part.content : JSON.stringify(part.content));
      }
    }
    return total;
  }
  return 0;
}
