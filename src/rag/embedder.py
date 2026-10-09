"""文本向量化：把文本转成向量，是"语义相似"检索的基础。

- query 与文档必须使用同一向量空间
- 本实现：本地 sentence-transformers 加载 BAAI/bge-small-zh-v1.5，
  中英双语、512 维、CPU 可跑，不依赖网络/API。
- 国内网络：HuggingFace 直连经常超时，这里强制走 hf-mirror.com 镜像；
  模型已下载到本地缓存后，设置 HF_HUB_OFFLINE=1 可完全离线。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

# 国内镜像：sentence-transformers 默认连 huggingface.co，国内直连超时。
# 优先用用户显式设置的值，否则默认走 hf-mirror.com。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class BaseEmbedder(ABC):
    """向量化统一接口。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """把一组文本转为向量矩阵，形状 (len(texts), dim)。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""
        raise NotImplementedError


class LocalSentenceEmbedder(BaseEmbedder):
    """本地 sentence-transformers 向量化（bge-small-zh-v1.5）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        # 模型已下载到本地缓存后，不联网检查文件（避免 huggingface 超时重试）。
        # 首次下载时需要联网（走上面设的 HF_ENDPOINT 镜像）。
        offline = os.environ.get("HF_HUB_OFFLINE", "1") == "1"
        self._model = SentenceTransformer(model_name, local_files_only=offline)
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,   # 归一化后余弦≈点积，检索更稳
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._dim = int(vecs.shape[1])
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed([""])
        return int(self._dim)


def get_embedder() -> BaseEmbedder:
    """工厂函数：按 config 返回本地向量化实现。"""
    return LocalSentenceEmbedder()