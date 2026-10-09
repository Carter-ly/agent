"""全局配置：从环境变量 / .env 文件读取，集中管理。

模型 provider 尚未最终确定，所有可选项都先通过环境变量暴露，
后续切换模型只需改 .env，无需改代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 启动时自动加载项目根目录下的 .env
load_dotenv()


@dataclass
class LLMConfig:
    """大模型配置。"""
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str | None = os.getenv("LLM_BASE_URL") or None


@dataclass
class EmbeddingConfig:
    """向量化配置。"""
    provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key: str = os.getenv("EMBEDDING_API_KEY", "")


@dataclass
class RAGConfig:
    """RAG 检索配置。"""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "4"))


@dataclass
class Config:
    """总配置。"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)


# 全局单例，各处 `from config import config` 即可使用
config = Config()
