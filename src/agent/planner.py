"""规划助手（Planner Agent）：根据用户提供的国家/学校/专业/人群，生成定制留学规划方案。

与问答 Agent（agent.py）的区别：
- 问答 Agent 面向单条具体问题，自由决定检索什么。
- Planner 面向"整份规划"，拿到结构化的 {country, school, program, audience} 后，
  按固定维度（语言要求、截止日期、学费、申请材料与流程、人群特定要求）依次检索，
  再把全部证据交给大模型统一生成一份结构化方案，并标注来源。

这样做的理由（答辩可讲）：
1. 规划覆盖多个事实维度，分维度检索比让模型一次问一条更高效、更不丢信息；
2. 检索顺序由代码编排（deterministic），结果稳定、来源清晰，便于前端分步展示；
3. 生成阶段用单一 system prompt 强约束"只依据检索内容 + 标注来源 + 查不到就说没查到"，
   复用问答 Agent 的幻觉控制三件套思路。

事件流（复用问答 Agent 的 SSE 约定，供后端转发）：
- {"type":"tool",   "name":..., "arg":...}      # 开始检索某维度
- {"type":"source", "sources":[...]}            # 该维度命中的来源
- {"type":"token",  "text":...}                 # 方案 token 增量（打字机）
- {"type":"done",   "answer":...}
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from src.agent.prompts import PLANNER_PROMPT
from src.agent.tools import Tool
from src.llm.base import BaseLLM, Message

# 规划维度：每个维度对应一条检索 query 模板。
# {school}/{program}/{audience} 会被实际值替换；缺省字段留空则该维度跳过。
PLAN_DIMENSIONS = [
    {
        "name": "program_overview",
        "label": "项目概况",
        "query": "{school} {program} 硕士项目介绍 课程 学制 学费",
    },
    {
        "name": "language",
        "label": "语言要求",
        "query": "{school} {program} 语言要求 雅思 托福 英语成绩",
    },
    {
        "name": "deadline",
        "label": "申请截止日期",
        "query": "{school} {program} 申请截止日期 申请时间 2026 2027",
    },
    {
        "name": "tuition",
        "label": "学费与费用",
        "query": "{school} {program} 学费 费用 生活费 资金",
    },
    {
        "name": "materials",
        "label": "申请材料与流程",
        "query": "{school} {program} 申请材料 申请条件 申请流程 要求",
    },
    {
        "name": "audience",
        "label": "人群特定要求",
        "query": "{school} {program} {audience} 申请要求 条件",
    },
]


class Planner:
    """根据结构化用户信息生成定制留学规划方案。"""

    def __init__(self, llm: BaseLLM, tools: list[Tool], max_dimensions: int | None = None) -> None:
        self.llm = llm
        self.retrieve_tool = next(t for t in tools if t.name == "search_knowledge")
        self.max_dimensions = max_dimensions

    def _fill(self, template: str, info: dict[str, str]) -> str:
        """把 {school}/{program}/{audience} 模板填充为用户信息；空字段用通用词。"""
        school = info.get("school", "").strip()
        program = info.get("program", "").strip()
        audience = info.get("audience", "").strip()
        return (
            template
            .replace("{school}", school if school else "目标学校")
            .replace("{program}", program if program else "目标专业")
            .replace("{audience}", audience if audience else "申请人")
        )

    def _collect_evidence(self, info: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
        """按维度依次检索，返回 (证据列表, 全部来源列表)。"""
        evidence: list[dict[str, Any]] = []
        all_sources: list[str] = []
        dims = PLAN_DIMENSIONS
        if self.max_dimensions:
            dims = dims[: self.max_dimensions]

        for dim in dims:
            query = self._fill(dim["query"], info)
            obs, sources = self._execute_retrieve(query)
            evidence.append({
                "dim": dim["name"],
                "label": dim["label"],
                "query": query,
                "context": obs,
            })
            all_sources.extend(sources)
        return evidence, all_sources

    def _execute_retrieve(self, query: str) -> tuple[str, list[str]]:
        """执行一次检索，返回 (观察文本, 来源列表)。"""
        observation = self.retrieve_tool(query)
        return observation, _extract_sources(observation)

    def _build_messages(
        self, info: dict[str, str], evidence: list[dict[str, Any]]
    ) -> list[Message]:
        """构造送给 LLM 的消息：system + 用户信息 + 全部检索证据。"""
        country = info.get("country", "").strip()
        school = info.get("school", "").strip()
        program = info.get("program", "").strip()
        audience = info.get("audience", "").strip()

        user_info_lines = [
            "【用户目标】",
            f"- 国家/地区：{country or '未提供'}",
            f"- 学校：{school or '未提供'}",
            f"- 专业：{program or '未提供'}",
            f"- 人群类型：{audience or '未提供'}",
            "",
            "【检索证据】",
        ]
        for i, ev in enumerate(evidence, 1):
            user_info_lines.append(f"[证据{i} · {ev['label']}]（检索词：{ev['query']}）")
            user_info_lines.append(ev["context"])
            user_info_lines.append("")

        return [
            Message(role="system", content=PLANNER_PROMPT),
            Message(role="user", content="\n".join(user_info_lines)),
        ]

    def stream(self, info: dict[str, str]) -> Iterator[dict[str, Any]]:
        """生成规划方案，逐事件产出（供后端 SSE 转发）。"""
        # 阶段一：分维度检索
        evidence, all_sources = self._collect_evidence(info)
        for ev in evidence:
            yield {"type": "tool", "name": "search_knowledge", "arg": ev["query"]}
            srcs = _extract_sources(ev["context"])
            if srcs:
                yield {"type": "source", "sources": srcs}

        # 阶段二：统一生成方案
        messages = self._build_messages(info, evidence)
        answer = ""
        for token in self.llm.chat_stream(messages):
            yield {"type": "token", "text": token}
            answer += token

        yield {"type": "done", "answer": answer}

    def run(self, info: dict[str, str]) -> str:
        """非流式入口，直接返回方案文本。"""
        answer = ""
        for ev in self.stream(info):
            if ev["type"] == "token":
                answer += ev["text"]
        return answer


def _extract_sources(context_text: str) -> list[str]:
    """从检索工具返回的上下文中提取来源（[片段N]（来源：...））。"""
    return re.findall(r"来源：([^）\]]+)", context_text)
