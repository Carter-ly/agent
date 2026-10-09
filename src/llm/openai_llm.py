"""DeepSeek / OpenAI 兼容接口的大模型实现。

通过 OpenAI 兼容的 chat.completions 接口调用（默认 DeepSeek，可换任意
兼容 OpenAI 接口的服务商）。提供两种方式：
- chat:        一次性返回完整回答（Agent 的 ReAct 循环推理用）
- chat_stream: 流式返回 token（前后端 SSE 流式展示用）
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from src.llm.base import BaseLLM, Message


class OpenAILLM(BaseLLM):
    """基于 OpenAI 兼容接口（默认 DeepSeek）的模型实现。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _to_openai(messages: list[Message]) -> list[dict[str, str]]:
        """把内部 Message 列表转成 OpenAI 的 messages 格式。"""
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: list[Message], temperature: float = 0.0) -> str:
        """给定消息列表，一次性返回模型生成的文本回复。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai(messages),
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """流式返回模型生成的 token（用于 SSE 打字机效果）。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai(messages),
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            token = delta.content
            if token:
                yield token