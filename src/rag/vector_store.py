"""向量存储与相似度检索。

- 把 chunk 及其向量存起来
- 用余弦相似度做 top-k 检索
- 数据量小用线性扫描；后续可换近似最近邻索引或加持久化
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.rag.chunker import Chunk


@dataclass
class SearchResult:
    """一条检索结果：片段 + 相似度分数。"""
    chunk: Chunk
    score: float


class VectorStore:
    """向量库：numpy 矩阵存储 + 余弦相似度线性扫描。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """写入 chunk 及其向量。vectors 形状需为 (len(chunks), dim)。"""
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors 行数必须与 chunks 数量一致")
        self._chunks.extend(chunks)
        if self._vectors is None:
            self._vectors = np.asarray(vectors, dtype=np.float32)
        else:
            self._vectors = np.concatenate([self._vectors, np.asarray(vectors, dtype=np.float32)], axis=0)

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        """返回与 query_vec 最相似的 top_k 个片段。"""
        if self._vectors is None or len(self._chunks) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        v_norm = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        # 余弦相似度 = 点积 / (|q| * |v|)；embedder 已归一化，此处再除一次更稳
        scores = (q @ self._vectors.T) / (q_norm * v_norm + 1e-9)
        scores = scores[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._chunks)