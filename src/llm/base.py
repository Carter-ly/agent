"""LLM 抽象层：定义统一接口，让 RAG 和 Agent 不关心具体用哪家模型。

职责：定义大模型的统一接口，具体实现按 provider 可插拔。
本实现默认走 DeepSeek（OpenAI 兼容接口），见 src/llm/openai_llm.py。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Message:
    """一条对话消息。role: system / user / assistant。"""
    role: str
    content: str


class BaseLLM(ABC):
    """大模型统一接口。"""

    @abstractmethod
    def chat(self, messages: list[Message], temperature: float = 0.0) -> str:
        """给定消息列表，返回模型生成的文本回复。

        Args:
            messages: 按顺序排列的对话消息（含 system 提示、历史、当前问题）。
            temperature: 采样温度，检索问答类任务建议默认 0。
        """
        raise NotImplementedError

    def chat_stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """流式返回模型生成的 token（默认退化为一次性返回）。

        子类可实现为真正的流式输出，用于前后端 SSE 打字机效果。
        """
        yield self.chat(messages, temperature)


def get_llm() -> BaseLLM:
    """工厂函数：根据 config 创建对应的 LLM 实例（默认 DeepSeek 兼容接口）。"""
    from config import config

    from src.llm.openai_llm import OpenAILLM

    return OpenAILLM(
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )
