"""工具定义：Agent 可调用的能力，这里以"检索工具"为主。

- Tool 需要 name（唯一标识）、description（喂给大模型，让它知道何时调用）、func（执行函数）
- 至少提供"检索知识库"工具，衔接 src.rag.retriever

自研要点（答辩可讲）：
- 把 retriever.retrieve + format_context 包成一个纯文本工具，LLM 只能看到文本结果，
  无法直接访问向量库，安全且职责单一。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    """一个工具。"""
    name: str
    description: str
    func: Callable[..., str]

    def __call__(self, *args, **kwargs) -> str:
        return self.func(*args, **kwargs)


def make_retrieve_tool(retriever) -> Tool:
    """创建"检索知识库"工具。

    检索器把用户问题召回相关片段后，再用 format_context 拼成一段
    带来源标注的上下文文本返回。LLM 拿到的就是可读的文本证据。
    """

    def _retrieve(query: str) -> str:
        results = retriever.retrieve(query)
        if not results:
            return "（知识库中未检索到相关片段）"
        return retriever.format_context(results)

    return Tool(
        name="search_knowledge",
        description=(
            "在留学知识库中检索与问题相关的资料片段，返回带来源的上下文。"
            "当问题涉及院校、专业、语言要求、学费、申请、签证、留学等具体事实时，"
            "必须调用本工具获取依据后再作答。"
        ),
        func=_retrieve,
    )