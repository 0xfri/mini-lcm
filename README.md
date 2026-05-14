# mini-lcm

轻量级无损上下文管理插件，专为中文场景优化，为 [OpenClaw](https://github.com/openclaw/openclaw) 设计。

## ✨ 特性

- **CJK Token 修正**：中文 1.5 tok/char，Emoji 2.0 tok/char（修正 6 倍偏差）
- **SQLite 持久化**：所有消息永不丢失
- **自动压缩**：context 快满时用 LLM 压缩旧消息成摘要
- **FTS5 全文检索**：关键词精确匹配记忆
- **向量语义搜索**：阿里 text-embedding-v3 (1024维)
- **跨会话记忆**：自动提取决策/学习/偏好，跨会话共享
- **安全防护**：防幻觉、防重复、防投毒

## 🏗️ 架构

```
┌──────────────────────────────────────────────┐
│                 mini-lcm                      │
├──────────────────────────────────────────────┤
│  消息层 (per-session)                         │
│  SQLite: messages + summaries + engine_state  │
├──────────────────────────────────────────────┤
│  记忆层 (cross-session)                       │
│  SQLite: memories + FTS5 + embeddings         │
├──────────────────────────────────────────────┤
│  检索层                                       │
│  向量搜索(0.4) + FTS5(0.3) + 时间排序(0.3)    │
├──────────────────────────────────────────────┤
│  安全层                                       │
│  敏感信息过滤 + 置信度评分 + 向量去重           │
│  + 冲突检测 + 可信度衰减                       │
├──────────────────────────────────────────────┤
│  组装层                                       │
│  [记忆] + [摘要(预算裁剪)] + [最近消息] → 模型 │
└──────────────────────────────────────────────┘
```

## 📦 安装

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/mini-lcm.git
cd mini-lcm

# 安装依赖
npm install

# 编译
npm run build

# 安装到 OpenClaw（link 模式）
openclaw plugins install -l .
```

## ⚙️ 配置

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "mini-lcm"
    },
    "entries": {
      "mini-lcm": {
        "enabled": true,
        "config": {
          "freshTailCount": 64,
          "contextThreshold": 0.75,
          "summaryModel": "xiaomi-tokenplan/mimo-v2.5-pro",
          "embeddingModel": "dashscope/text-embedding-v3",
          "embeddingDim": 1024,
          "dbPath": "~/.openclaw/mini-lcm.db"
        }
      }
    }
  }
}
```

需要环境变量 `DASHSCOPE_API_KEY`（阿里向量搜索）。

## 🧠 记忆类型

| type | 说明 |
|------|------|
| decision | 决策 |
| learning | 学到的知识 |
| config | 配置变更 |
| bugfix | Bug 修复 |
| preference | 用户偏好 |
| fact | 事实 |

## 🛡️ 安全机制

| 风险 | 机制 |
|------|------|
| 幻觉 | 原文锚定 + 置信度评分(≥0.7) + 人工确认 |
| 重复 | 向量相似度(≥0.85)合并 + content_hash |
| 投毒 | 敏感信息过滤 + 可信度衰减(90天) + 冲突检测 |

## 📁 文件结构

```
src/
├── index.ts            # 插件入口
├── token-estimator.ts  # CJK-aware token 估算
├── embedding.ts        # 向量嵌入接口
├── db.ts               # SQLite 数据库
├── memory-store.ts     # 跨会话记忆 + 安全防护
└── context-engine.ts   # 核心 ContextEngine 实现
```

## 📝 License

MIT
