"""Agent 主循环（ReAct 风格）：思考 -> 行动 -> 观察 -> 思考...，直到给出答案。

核心机制（答辩必问，吃透）：
1. System prompt 约定模型按固定格式输出：「TOOL:」表示要调用工具，或「ANSWER:」表示最终答案。
2. 循环里解析模型输出：TOOL 就执行对应工具，把观察（检索上下文）作为 user 消息喂回；
   ANSWER 就结束，返回答案。
3. 设置 max_steps 最大步数防止死循环 / 无限追问。
4. 幻觉控制三件套：强制引用来源、检索不到就说没查到、禁止编造。

调用方式：
- run(question, history)：一次性返回最终答案文本（命令行 / 非流式）。
- stream(question, history)：逐事件产出 dict，供后端 SSE 流式转发
  （tool 调用、token 增量、来源列表），同时支持打字机效果。
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from src.agent.tools import Tool
from src.llm.base import BaseLLM, Message

# 工具输出格式约定与系统提示词统一从 prompts.py 导入（集中管理）
from src.agent.prompts import ANSWER_PREFIX, SYSTEM_PROMPT, TOOL_PREFIX


def _parse_output(text: str) -> tuple[str, str] | None:
    """解析模型输出，返回 (kind, payload)；kind ∈ {tool, answer}。

    - `TOOL: search_knowledge("...")` → (tool, 检索词)
    - `ANSWER: ...` → (answer, 答案正文)
    解析失败返回 None（视为还需再思考，要求模型重新输出）。
    """
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith(TOOL_PREFIX):
        payload = stripped[len(TOOL_PREFIX):].strip()
        # 兼容 JSON 字符串 / 带引号的括号参数 / 裸检索词
        return ("tool", _extract_tool_arg(payload))
    if stripped.startswith(ANSWER_PREFIX):
        return ("answer", stripped[len(ANSWER_PREFIX):].strip())
    # 未按格式输出：退化为调用工具（检索），最稳妥
    return ("tool", stripped)


def _extract_tool_arg(payload: str) -> str:
    """从 `search_knowledge("...")` / `search_knowledge("...")` 或裸词中提取检索词。"""
    # 去掉函数名前缀
    name = "search_knowledge"
    if name in payload:
        payload = payload.split(name, 1)[1]
    # 去掉括号
    payload = payload.strip().lstrip("(").rstrip(")")
    # 去掉引号
    for q in ('"', "'", "“", "”"):
        payload = payload.strip(q)
    return payload.strip()


class Agent:
    """检索增强的问答智能体。"""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool],
        max_steps: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        # 缓存最近一次检索到的来源，供 SSE 展示
        self.last_sources: list[str] = []

    def _system_message(self) -> Message:
        return Message(role="system", content=self.system_prompt)

    def _history_messages(self, history: list[dict[str, str]]) -> list[Message]:
        """把多轮历史转成消息（过滤非 user/assistant 的异常项）。"""
        out: list[Message] = []
        for turn in history:
            role = turn.get("role")
            if role in ("user", "assistant"):
                out.append(Message(role=role, content=turn.get("content", "")))
        return out

    def _tool(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"未知工具: {name}，可用: {list(self.tools)}")
        return self.tools[name]

    def _execute(self, tool_name: str, arg: str) -> tuple[str, list[str]]:
        """执行工具，返回 (观察文本, 来源列表)。"""
        tool = self._tool(tool_name)
        observation = tool(arg)
        sources = _extract_sources(observation)
        return observation, sources

    def run(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        """回答一个问题（非流式），返回最终答案文本。"""
        answer = ""
        for _event in self.stream(question, history or []):
            if _event["type"] == "token":
                answer += _event["text"]
        return answer

    def stream(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> Iterator[dict[str, Any]]:
        """ReAct 循环，逐事件产出 dict，供后端 SSE 流式转发。

        事件类型：
        - {"type": "tool", "name": ..., "arg": ...}：开始调用工具
        - {"type": "source", "sources": [...]}：检索到的来源列表
        - {"type": "token", "text": ...}：答案 token 增量（打字机）
        - {"type": "done", "answer": ...}：结束
        """
        history = history or []
        messages: list[Message] = [self._system_message()]
        messages += self._history_messages(history)
        messages.append(Message(role="user", content=question))

        final_answer = ""
        reasoning: list[Message] = []
        used_sources: list[str] = []

        for _step in range(self.max_steps):
            full = self.llm.chat(messages + reasoning)
            parsed = _parse_output(full)
            if parsed is None:
                # 没按格式输出：要求重来一次（再喂一个提示）
                reasoning.append(Message(role="assistant", content=full))
                reasoning.append(Message(
                    role="user",
                    content=f"请严格按格式输出，只能输出 {TOOL_PREFIX} 或 {ANSWER_PREFIX} 开头的一行。",
                ))
                continue

            kind, payload = parsed
            if kind == "tool":
                reasoning.append(Message(role="assistant", content=full))
                yield {"type": "tool", "name": payload.split("(", 1)[0].strip() or "search_knowledge", "arg": payload}
                observation, sources = self._execute("search_knowledge", payload)
                used_sources = sources
                if sources:
                    yield {"type": "source", "sources": sources}
                # 观察作为 user 消息喂回（ReAct 的 observation 环）
                reasoning.append(Message(role="user", content=f"【检索结果】\n{observation}"))
                continue

            # kind == "answer"
            final_answer = payload
            # 流式产出答案（真实打字机）：直接用模型已生成的最终答案逐 token 产出
            yield {"type": "token", "text": final_answer}
            self.last_sources = used_sources
            yield {"type": "done", "answer": final_answer}
            return

        # 达到最大步数仍未结束：给出兜底回答
        final_answer = "抱歉，我经过多次尝试仍未在知识库中找到确定答案，请换个问法或咨询官方渠道。"
        yield {"type": "token", "text": final_answer}
        self.last_sources = used_sources
        yield {"type": "done", "answer": final_answer}


def _extract_sources(context_text: str) -> list[str]:
    """从检索工具返回的上下文中提取来源（[片段N]（来源：...））。"""
    import re

    return re.findall(r"来源：([^）\]]+)", context_text)