"""后端服务层（FastAPI）：把 src/ 里的 RAG + Agent 核心逻辑包装成 HTTP API。

接口：
- GET  /            → 返回前端聊天页面（frontend/index.html）
- GET  /health      → 健康检查
- POST /api/ask     → 提交问题，返回 SSE 流式响应（打字机 + 来源）

设计要点：
- 服务启动时只构建一次 Agent（知识库索引 + 模型加载较慢，约 1 分钟），
  所有请求复用同一个实例，避免重复索引。
- SSE 流式事件（每条一行 `data: <json>\n\n`）：
    {"type":"tool",   "name":..., "arg":...}
    {"type":"source", "sources":[...]}
    {"type":"token",  "text":...}      # 答案增量，打字机效果
    {"type":"done",   "answer":...}
- 多轮历史由前端随请求带上（history），后端透传给 Agent。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.agent import Agent
from src.agent.planner import Planner
from src.agent.prompts import ADVISOR_PROMPT
from src.agent.tools import make_retrieve_tool
from src.llm.base import get_llm
from src.rag.retriever import Retriever

app = FastAPI(title="留学全周期智能助手", version="1.0.0")

# 允许前端跨域（开发期可能用不同端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# 前端：Vue 构建产物目录（frontend/dist），由 `npm run build` 生成
FRONTEND_DIR = ROOT / "frontend" / "dist"

# 全局单例：启动时构建一次
_agent: Agent | None = None
_advisor: Agent | None = None
_planner: Planner | None = None
_retriever: Retriever | None = None


def _build_retriever() -> Retriever:
    """构建（或复用）全局检索器。"""
    global _retriever
    if _retriever is None:
        r = Retriever.from_config()
        r.index_directory(str(DATA_DIR))
        _retriever = r
    return _retriever


def get_agent() -> Agent:
    """构建（或复用）全局问答 Agent 实例（提问功能）。"""
    global _agent
    if _agent is None:
        retriever = _build_retriever()
        llm = get_llm()
        _agent = Agent(llm=llm, tools=[make_retrieve_tool(retriever)])
    return _agent


def get_advisor() -> Agent:
    """构建（或复用）全局规划引导 Agent 实例（没思路的规划功能）。

    复用 Agent 的 ReAct 循环，但 system prompt 换成 ADVISOR_PROMPT：
    通过多轮对话一步步引导用户理清目标。
    """
    global _advisor
    if _advisor is None:
        retriever = _build_retriever()
        llm = get_llm()
        _advisor = Agent(
            llm=llm,
            tools=[make_retrieve_tool(retriever)],
            system_prompt=ADVISOR_PROMPT,
        )
    return _advisor


def get_planner() -> Planner:
    """构建（或复用）全局规划助手实例（有目标的规划功能）。"""
    global _planner
    if _planner is None:
        retriever = _build_retriever()
        llm = get_llm()
        _planner = Planner(llm=llm, tools=[make_retrieve_tool(retriever)])
    return _planner


class AskRequest(BaseModel):
    """一次问答请求。"""
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)


class PlanRequest(BaseModel):
    """一次定制规划请求（引导流程收集到的结构化信息）。"""
    country: str = ""
    school: str = ""
    program: str = ""
    audience: str = ""


@app.on_event("startup")
def _startup() -> None:
    """启动时预热：构建 Agent / Advisor / Planner，避免首个请求等待 1 分钟。"""
    get_agent()
    get_advisor()
    get_planner()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    """提交问题，返回 SSE 流式响应。"""
    agent = get_agent()

    def event_stream():
        # 先发一个开始事件（可选）
        yield _sse({"type": "start"})
        for ev in agent.stream(req.question, req.history):
            yield _sse(ev)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/advise")
def advise(req: AskRequest) -> StreamingResponse:
    """提交一条引导对话消息，返回 SSE 流式响应（没思路的规划引导）。

    与 /api/ask 的区别：使用 ADVISOR_PROMPT 的引导 Agent，
    通过多轮 history 保持引导上下文，一步步帮用户理清规划。
    """
    advisor = get_advisor()

    def event_stream():
        yield _sse({"type": "start"})
        for ev in advisor.stream(req.question, req.history):
            yield _sse(ev)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/plan")
def plan(req: PlanRequest) -> StreamingResponse:
    """提交结构化目标（国家/学校/专业/人群），返回定制规划方案的 SSE 流。"""
    planner = get_planner()
    info = {
        "country": req.country.strip(),
        "school": req.school.strip(),
        "program": req.program.strip(),
        "audience": req.audience.strip(),
    }

    def event_stream():
        for ev in planner.stream(info):
            yield _sse(ev)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(data: dict) -> str:
    """把一个 dict 编码为 SSE 数据帧。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# 挂载前端静态页面（index.html 在 / 下可直接访问）
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")