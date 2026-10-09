"""检索器：把 RAG 各环节组装成"问题 -> 相关片段"的完整链路，供 Agent 工具调用。

- index_directory: 加载 -> 切块 -> 向量化 -> 写入向量库 + 建 BM25 索引
- retrieve: query 向量化 -> 双路召回（向量 + BM25）-> RRF 融合 -> top_k
- format_context: 把结果拼成给大模型的上下文
"""
from __future__ import annotations

import re

import numpy as np

from src.rag.chunker import Chunk, chunk_documents
from src.rag.embedder import BaseEmbedder, get_embedder
from src.rag.loader import load_directory
from src.rag.vector_store import SearchResult, VectorStore

RRF_K = 60  # RRF 平滑常数

# 学校/专业英文缩写 → 中文全称：query 改写时展开，让 BM25 能精确命中知识库里的中文名
SCHOOL_ALIASES = {
    "UCL": "伦敦大学学院",
    "LSE": "伦敦政治经济学院",
    "CS": "计算机科学",
    "MSc": "硕士",
    "MA": "硕士",
    "斯坦福": "斯坦福大学",
    "墨尔本": "墨尔本大学",
}

# 国家对别名 → 知识库 country 字段值（来自 data 目录层级）
COUNTRY_ALIASES = {
    "澳洲": "澳洲",
    "澳大利亚": "澳洲",
    "美国": "美国",
    "英国": "英国",
    "加拿大": "加拿大",
    "新西兰": "新西兰",
    "新加坡": "新加坡",
    "日本": "日本",
    "韩国": "韩国",
}

# 字段词：问题提到这些词时，优先 title 匹配的小节（解决「概括段抢排名」）。
# 注意：不含「材料/申请材料」——该词太通用，会误伤签证类问题（见 Bad Case）；
# 也不含「签证」——签证是类别主题而非字段。仅保留强字段词。
FIELD_KEYWORDS = ["语言要求", "学费", "截止日期", "打工", "资金", "考试", "要求", "条件"]

# 「认证/清单」类问题：优先返回国家认证院校清单（title 含「清单」），绕过 school 过滤
CERT_KEYWORDS = ["认证", "教育部", "院校", "留学服务中心", "学历"]


def _chinese_tokenize(text: str) -> list[str]:
    """中文分词：jieba 切成词语列表，去掉纯空白词。"""
    try:
        import jieba

        jieba.setLogLevel(60)  # 压掉 jieba 词典加载日志
        return [w.strip() for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return re.findall(r"[\w\u4e00-\u9fff]+", text)


class Retriever:
    """从文档库中检索与问题相关的片段。"""

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.store = VectorStore()
        self._chunks: list[Chunk] = []
        self._bm25: object | None = None  # BM25Okapi

    @classmethod
    def from_config(cls) -> "Retriever":
        """按 config 构造默认检索器。"""
        from config import config

        return cls(
            embedder=get_embedder(),
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            top_k=config.rag.top_k,
        )

    def index_directory(self, data_dir: str) -> None:
        """加载目录下文档，切块、向量化并写入向量库与 BM25 索引。"""
        docs = load_directory(data_dir)
        chunks = chunk_documents(docs, self.chunk_size, self.chunk_overlap)
        self._chunks = chunks
        if chunks:
            vecs = self.embedder.embed([c.content for c in chunks])
            self.store.add(chunks, vecs)
            self._build_bm25(chunks)

    def _build_bm25(self, chunks: list[Chunk]) -> None:
        """用 rank_bm25 构建 BM25 索引（jieba 分词）。"""
        from rank_bm25 import BM25Okapi

        tokenized = [_chinese_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _vectors_for_indices(self, idx: list[int]) -> np.ndarray:
        """取指定 chunk 的向量子集。"""
        return self.store._vectors[idx]  # noqa: SLF001

    def _search_subset(
        self, q_vec: np.ndarray, sub_vecs: np.ndarray, sub_idx: list[int], top_k: int
    ) -> list[SearchResult]:
        """在向量子集内按余弦相似度检索。"""
        q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        v_norm = np.linalg.norm(sub_vecs, axis=1, keepdims=True)
        scores = (q @ sub_vecs.T) / (q_norm * v_norm + 1e-9)
        scores = scores[0]
        top = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(chunk=self._chunks[sub_idx[i]], score=float(scores[i])) for i in top]

    def _bm25_subset(
        self, tokens: list[str], sub_idx: list[int], top_k: int
    ) -> list[SearchResult]:
        """在指定 chunk 子集内做 BM25 检索（仅该校）。"""
        out: list[SearchResult] = []
        if self._bm25 is None:
            return out
        scores = self._bm25.get_scores(tokens)
        sub = sorted(sub_idx, key=lambda i: scores[i], reverse=True)
        sub = [i for i in sub if scores[i] > 0][:top_k]
        out = [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in sub]
        return out

    def retrieve(self, query: str) -> list[SearchResult]:
        """检索与 query 最相关的 top_k 个片段（向量 + BM25 双路 RRF 融合）。

        改进点（对应 Bad Case）：
        1. 查询改写：学校缩写展开（LSE→伦敦政治经济学院），BM25 能精确命中中文名；
        2. 国家过滤：query 含国家词时，该国结果优先（解决跨国家同构污染）；
        3. 概括段降权：文档一级标题段（数据来源/简介）扣分，让字段小节上位。
        """
        country = self._detect_country(query)
        query = self._rewrite_query(query)
        school = self._detect_school(query)
        is_cert = any(kw in query for kw in CERT_KEYWORDS)

        q_vec = self.embedder.embed([query])[0]

        # 学校限定检索：query 明确提到某校时，向量/BM25 都优先在该校子集内召回，
        # 避免全库 30+ 学校的同类字段（如「语言要求」）把目标学校的 chunk 挤出候选。
        # 认证类问题（is_cert）不走此路径——它们要的是国家清单，不是该校详情。
        candidate_size = 60 if len(self.store) > 200 else 20
        if school and not is_cert:
            q_program = self._detect_program(query)
            school_idx = [i for i, c in enumerate(self._chunks)
                          if c.metadata.get("school") == school]
            if q_program:
                school_idx = [i for i in school_idx
                              if self._chunks[i].metadata.get("program") == q_program]
            if school_idx:
                school_vecs = self._vectors_for_indices(school_idx)
                vec_results = self._search_subset(q_vec, school_vecs, school_idx,
                                                  top_k=candidate_size)
                bm25_results = self._bm25_subset(_chinese_tokenize(query), school_idx,
                                                 top_k=candidate_size)
                results = self._rrf_fuse(vec_results, bm25_results, top_k=candidate_size)
                results = self._downweight_summary(results)
                results = self._boost_field_title(query, results)
                if country:
                    results = self._prioritize_country(results, country)
                return results[: self.top_k]

        vec_results = self.store.search(q_vec, top_k=candidate_size)

        bm25_results: list[SearchResult] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_chinese_tokenize(query))
            top_idx = np.argsort(scores)[::-1]
            # 只保留有命中（score>0）的，否则全库都进来
            top_idx = [i for i in top_idx if scores[i] > 0][:candidate_size]
            bm25_results = [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx]

        results = self._rrf_fuse(vec_results, bm25_results, top_k=candidate_size)
        results = self._downweight_summary(results)
        results = self._boost_field_title(query, results)
        if is_cert:
            results = self._prioritize_cert_list(query, results)
        elif school:
            results = self._prioritize_school_program(results, school, query)
        if country:
            results = self._prioritize_country(results, country)
        return results[: self.top_k]

    def _detect_school(self, query: str) -> str | None:
        """从 query 里识别学校（中文全名），返回 school 值；识别不到返回 None。"""
        for ch in self._chunks:
            s = ch.metadata.get("school")
            if s and s in query:
                return s
        return None

    def _detect_program(self, query: str) -> str | None:
        """从 query 里识别专业（中文名，如 计算机科学/管理学），返回 program 值。"""
        for ch in self._chunks:
            p = ch.metadata.get("program")
            if p and p in query:
                return p
        return None

    @staticmethod
    def _detect_country(query: str) -> str | None:
        """从 query 里识别国家（含别名），返回知识库 country 值；识别不到返回 None。"""
        for alias, country in COUNTRY_ALIASES.items():
            if alias in query:
                return country
        return None

    @staticmethod
    def _rewrite_query(query: str) -> str:
        """查询改写：英文缩写展开为中文全称，便于 BM25 精确命中。"""
        for abbr, full in SCHOOL_ALIASES.items():
            # 只替换独立出现的缩写（避免把 UCL 拆进别的词）
            query = re.sub(rf"(?<![A-Za-z]){re.escape(abbr)}(?![A-Za-z])", full, query)
        return query

    @staticmethod
    def _downweight_summary(results: list[SearchResult]) -> list[SearchResult]:
        """概括段降权：文档一级标题段（title_level==1）扣 0.005 分重排。

        但「认证院校清单」这类整篇即表格的文档，h1 就是正文本身，
        不降权，否则会误伤（见 Bad Case：Q5）。
        """
        for r in results:
            if r.chunk.metadata.get("title_level") == 1:
                title = r.chunk.metadata.get("title") or ""
                if "清单" not in title:   # 清单文档不降权
                    r.score -= 0.005
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _boost_field_title(query: str, results: list[SearchResult]) -> list[SearchResult]:
        """字段标题加权：问题含「学费/截止日期/语言要求」等字段词时，
        给 title 命中的小节 +0.006 分，让具体字段排在概括段之前。"""
        hit_kw = [kw for kw in FIELD_KEYWORDS if kw in query]
        if not hit_kw:
            return results
        for r in results:
            title = r.chunk.metadata.get("title") or ""
            if any(kw in title for kw in hit_kw):
                r.score += 0.006
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _prioritize_school_program(
        results: list[SearchResult], school: str, query: str
    ) -> list[SearchResult]:
        """学校+专业过滤：query 明确提到某校时，该校结果前置；若 query 还含
        具体专业词（计算机科学/电气工程/管理学…），该校该专业的 chunk 再优先，
        避免问「斯坦福 CS 截止日期」时被同校其他专业（如电气工程）抢排名。"""
        # 从 query 中识别专业词：遍历所有已知 program 值
        known_programs: set[str] = set()
        for r in results:
            p = r.chunk.metadata.get("program")
            if p:
                known_programs.add(p)
        q_program = None
        for p in known_programs:
            if p and p in query:
                q_program = p
                break

        school_and_prog = [
            r for r in results
            if r.chunk.metadata.get("school") == school
            and q_program and r.chunk.metadata.get("program") == q_program
        ]
        school_only = [
            r for r in results
            if r.chunk.metadata.get("school") == school
            and not (q_program and r.chunk.metadata.get("program") == q_program)
        ]
        miss = [r for r in results if r.chunk.metadata.get("school") != school]
        return school_and_prog + school_only + miss

    @staticmethod
    def _prioritize_cert_list(
        query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """认证清单优先：「认证/教育部/院校」类问题，优先含该校名的国家清单。

        各国清单模板完全一致（同一权威源开头），BM25 无法靠「认证/院校」区分国家，
        因此改为：检测 query 中的学校中文名，把包含该校名的清单 chunk 顶到最前。
        """
        # 找出 query 里出现的学校中文全称（改写后 query 已展开 UCL→伦敦大学学院 等）
        hit: list[SearchResult] = []
        miss: list[SearchResult] = []
        for r in results:
            is_list = "清单" in (r.chunk.metadata.get("title") or "") or "清单" in (r.chunk.metadata.get("source") or "")
            if is_list:
                # 该清单内容里是否包含 query 中的某所学校名
                contains_school = any(
                    name and name in r.chunk.content
                    for name in SCHOOL_ALIASES.values()
                    if len(name) >= 2 and name in query
                )
                if contains_school:
                    hit.append(r)
                else:
                    miss.append(r)
            else:
                miss.append(r)
        return hit + miss

    @staticmethod
    def _prioritize_country(
        results: list[SearchResult], country: str
    ) -> list[SearchResult]:
        """国家过滤：该国结果前置（保持各自内部顺序），他国结果殿后。"""
        hit = [r for r in results if r.chunk.metadata.get("country") == country]
        miss = [r for r in results if r.chunk.metadata.get("country") != country]
        return hit + miss

    @staticmethod
    def _rrf_fuse(
        vec: list[SearchResult], bm25: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion：对两路排名取 1/(k+rank) 求和后排序。"""
        score_map: dict[int, float] = {}  # chunk id -> rrf 分数
        rank_map: dict[int, Chunk] = {}
        for results in (vec, bm25):
            for rank, r in enumerate(results):
                cid = id(r.chunk)
                rank_map[cid] = r.chunk
                score_map[cid] = score_map.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            SearchResult(chunk=rank_map[cid], score=round(rrf, 4))
            for cid, rrf in ordered
        ]

    def metadata_filtered(self, results: list[SearchResult], **filters: object) -> list[SearchResult]:
        """按元数据过滤（如 country='英国'、school='UCL'），保持原顺序。"""
        out: list[SearchResult] = []
        for r in results:
            m = r.chunk.metadata
            if all(m.get(k) == v for k, v in filters.items()):
                out.append(r)
        return out

    def _build_parent_index(self) -> dict[str, list[Chunk]]:
        """按 parent_id 聚合 chunks：parent_id -> 该文档的全部 chunks（按 chunk_index 排序）。"""
        index: dict[str, list[Chunk]] = {}
        for c in self._chunks:
            pid = c.metadata.get("parent_id")
            if pid:
                index.setdefault(pid, []).append(c)
        for chunks in index.values():
            chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        return index

    def parent_expand(self, results: list[SearchResult]) -> list[SearchResult]:
        """Small-to-Big：命中某个 chunk 后，把同一父文档（parent_id）的全部相邻 chunk
        合并为一条完整上下文，弥补切块造成的语义断裂（如清单表格被切块）。

        返回合并后的 SearchResult 列表：每个结果对应一个父文档，
        content 为整篇文档拼接文本，分数保留原命中 chunk 的分数。
        """
        index = self._build_parent_index()
        # 保留首个命中顺序，同父文档只合并一次
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for r in results:
            pid = r.chunk.metadata.get("parent_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            siblings = index.get(pid) or [r.chunk]
            if len(siblings) == 1:
                merged.append(r)
            else:
                full = "\n\n".join(c.content for c in siblings)
                # 复制命中 chunk 的元数据，替换内容为整篇
                meta = dict(r.chunk.metadata)
                title = meta.get("title") or "(整篇)"
                meta["title"] = title
                merged.append(SearchResult(chunk=Chunk(
                    content=full, source=r.chunk.source, index=r.chunk.index, metadata=meta
                ), score=r.score))
        return merged

    def format_context(self, results: list[SearchResult], expand_parent: bool = True) -> str:
        """把检索结果拼成给大模型的上下文文本。

        expand_parent=True 时先做 parent 聚合（Small-to-Big），
        清单/表格类被切块的文档会以整篇形式返回，保证语义完整。
        """
        if expand_parent:
            results = self.parent_expand(results)
        parts = []
        for i, r in enumerate(results, 1):
            m = r.chunk.metadata
            # 来源优先展示真实网页 url（frontmatter 里标注的官方来源）；
            # 没有 url 时退回文件路径（source），保证可追溯。
            src = m.get("url") or m.get("source", r.chunk.source)
            title = m.get("title") or "(无标题)"
            parts.append(f"[片段{i}]（来源：{src} / {title}）\n{r.chunk.content}")
        return "\n\n".join(parts)