# RAG + Agent 实训项目

小学期程序实训项目：**检索增强生成（RAG）与智能体（Agent）的结合**，
最终交付一个**智能问答助手的网页**。

核心目标：不依赖 LangChain / LlamaIndex 等现成框架，**纯 Python 自研**实现一个
能"先查资料、再回答问题"的智能体 —— Agent 根据问题自主决定是否调用检索工具，
把检索到的文档片段作为上下文交给大模型生成答案。

## 当前状态

> **已完成**：RAG 全链路（loader → chunker → embedder → vector_store → retriever）
> 与生成链路（llm → agent ReAct 循环 → Prompt + 幻觉控制）均已实现并跑通；
> 后端 FastAPI + SSE 流式 + 多轮历史 + 前端聊天网页已集成，可一键启动演示。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置模型 API（复制 .env.full 为 .env 并填入你的 DeepSeek Key）
cp .env.full .env

# 3. 命令行问答（作业三：生成链路）
python main.py

# 4. 网页版（作业四：系统集成）
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
#    浏览器打开 http://127.0.0.1:8000 即可聊天（SSE 流式 + 来源展示 + 多轮）
```

## 整体架构（三层）

```
┌─────────────────────────────────────────────────┐
│  frontend/  前端展示层（待定方案）                │
│  聊天界面：输入框 + 消息气泡 + 流式显示           │
└───────────────────────┬─────────────────────────┘
                        │  HTTP / SSE
┌───────────────────────▼─────────────────────────┐
│  backend/  后端服务层（待定框架）                 │
│  把核心逻辑包装成 HTTP API（如 POST /api/ask）   │
└───────────────────────┬─────────────────────────┘
                        │  函数调用
┌───────────────────────▼─────────────────────────┐
│  src/  核心逻辑层（纯 Python，与界面无关）        │
│  rag/（检索） + agent/（决策循环） + llm/（模型） │
└─────────────────────────────────────────────────┘
```

关键：`src/` 是纯逻辑，不依赖命令行或任何 Web 框架。前后端框架**待定**，
无论选 FastAPI / Flask、原生 JS / React，都只是在外层套壳，核心逻辑不改。

## 项目结构

```
rag-agent-training/
├── main.py               # 命令行入口（开发调试用，最终会换成网页）
├── config.py             # 配置加载（读取 .env）
├── requirements.txt      # 依赖
├── data/                 # 存放待检索的文档（txt/md，可扩展 pdf/docx）
├── class/                # 课程资料：笔记（原始课件 + 精简版总结）+ demo 示例代码
├── src/                  # 核心逻辑层（前后端共用）
│   ├── llm/              # 大模型抽象层（可插拔，支持后续切换模型）
│   │   └── base.py       # BaseLLM 接口 + 工厂函数
│   ├── rag/              # RAG 各环节，独立实现
│   │   ├── loader.py     # 文档加载
│   │   ├── chunker.py    # 文本切块
│   │   ├── embedder.py   # 文本向量化
│   │   ├── vector_store.py  # 向量索引与相似度检索
│   │   └── retriever.py  # 组装：问题 -> 相关文档片段
│   └── agent/            # Agent 循环
│       ├── tools.py      # 工具定义（检索工具等）
│       └── agent.py      # Agent 主循环（ReAct 风格）
├── backend/              # 后端服务层（待定框架，src 的 HTTP 壳）
└── frontend/             # 前端展示层（待定方案，问答网页）
```

## 设计要点

1. **RAG 与 Agent 解耦**：`src/rag` 只负责"文档 → 向量 → 检索"；`src/agent` 负责
   "决策 → 调用工具 → 生成答案"，通过 `tools.py` 里的检索工具衔接两者。
2. **LLM 可插拔**：`BaseLLM` 定义统一接口，具体模型实现放在 `src/llm/` 下，
   通过 `config.py` 的 `LLM_PROVIDER` 切换，后续确定用哪家模型只改一处。
3. **三层分离**：核心逻辑（src）/ 服务（backend）/ 展示（frontend）解耦，
   前后端技术栈可独立选定，互不影响。
4. **全程自研**：切块、向量化、相似度计算、Agent 的 ReAct 循环都自行实现，便于
   理解原理并在答辩/报告里讲清楚每一环。

## 快速开始（功能实现后）

> 以下命令需在对应模块实现后生效；当前为纯框架，直接运行会提示 NotImplementedError。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置模型 API（复制示例并填入 key）
cp .env.example .env
#    编辑 .env 填入你的 API Key

# 3. 把文档放进 data/ 目录

# 4. 运行（命令行模式，网页版待前后端框架确定后接入）
python main.py
```

## 路线图

- [ ] 文档加载与切块
- [ ] 向量化 + 向量存储与相似度检索
- [ ] LLM 抽象层 + 具体模型接入
- [ ] Agent 工具与 ReAct 循环
- [ ] 命令行交互入口
- [ ] 后端 API 服务（框架待定）
- [ ] 前端问答网页（方案待定）
- [ ] 前后端联调 + 流式输出
