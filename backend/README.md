# 后端服务层（FastAPI）

此目录把 `src/` 里的核心逻辑（RAG + Agent）包装成 HTTP API。

## 接口

- `GET /`            → 返回前端聊天页面
- `GET /health`      → 健康检查
- `POST /api/ask`    → 提交问题，返回 **SSE 流式**响应（打字机 + 来源）

## 运行

```bash
# 在 proj/ 目录下
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

启动时自动构建一次 Agent（知识库索引 + 模型加载，约 1 分钟），随后即可问答。

## SSE 事件格式（每条 `data: <json>\n\n`）

- `{"type":"tool","name":..,"arg":..}` 调用检索工具
- `{"type":"source","sources":[...]}` 参考来源
- `{"type":"token","text":..}` 答案增量（打字机）
- `{"type":"done","answer":..}` 结束

## 与核心层的关系

```
frontend (网页)  ──SSE──>  backend (API 壳)  ──调用──>  src/ (RAG+Agent 核心逻辑)
```

后端只负责接收请求、调用 `Agent.stream()`、转发事件，业务逻辑全在 `src/`。
