"""文本切块：按 Markdown 标题 / 段落边界切分，500 字兜底，生成带元数据的 Chunk。

自研核心（答辩必问，吃透这几点）：
1. 先按 # / ## / ### 标题把文档切成「语义小节」，每小节是一条完整信息单元
   （如「## 语言要求」），优先保证信息不被切断。
2. 小节超过 chunk_size 时，才按段落做字数兜底切分，并带 overlap，防止
   数字 / 单位被切在边界（如「20 小时」）。
3. 每个 chunk 继承父文档元数据（category/country/school/program/audience，
   由 loader 从目录层级 + 文件名推导），并额外记录所属标题、chunk_id、块长等。

用法：
    python -m src.rag.chunker   # 打印分块测试结果（作业一截图②）
"""
from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.loader import Document, load_directory

HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    """一个文本片段及其出处（含元数据，便于检索过滤与来源引用）。"""
    content: str
    source: str
    index: int                        # 在所属文档中的块序号
    metadata: dict = field(default_factory=dict)


def _split_by_headers(text: str) -> list[tuple[int, str, str]]:
    """按 # / ## / ### 切分，返回 [(标题级别, 标题, 正文)]；无标题则整篇一段。

    标题级别 = # 的数量（1/2/3），无标题记 0。级别用于检索时对文档
    一级标题（概括段）降权，避免其抢占具体字段小节的排名。
    """
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [(0, "", text.strip())]
    sections: list[tuple[int, str, str]] = []
    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            sections.append((0, "", pre))
    for i, m in enumerate(matches):
        level = len(m.group(1))          # 标题级别（# 个数）
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((level, title, body))
    return sections


def _split_long_body(body: str, chunk_size: int, overlap: int) -> list[str]:
    """小节超长时按段落贪心聚合成约 chunk_size 的块，相邻块重叠 overlap 字。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= chunk_size:
            cur = cur + "\n" + p
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + p).strip() if tail else p
    if cur.strip():
        chunks.append(cur)

    # 兜底：仍有超长块（单个段落就超限）则硬切
    out: list[str] = []
    for c in chunks:
        while len(c) > chunk_size:
            out.append(c[:chunk_size])
            c = c[chunk_size - overlap:] if overlap else c[chunk_size:]
        if c.strip():
            out.append(c.strip())
    return out


def chunk_document(
    doc: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """把单篇文档切成多个 chunk，每个 chunk 继承文档元数据（含 audience）。"""
    sections = _split_by_headers(doc.content)
    chunks: list[Chunk] = []
    for level, title, body in sections:
        pieces = (
            _split_long_body(body, chunk_size, chunk_overlap)
            if len(body) > chunk_size
            else [body]
        )
        for piece in pieces:
            meta = dict(doc.metadata)   # 继承 category/country/school/program/audience
            meta.update({
                "title": title,
                "title_level": level,
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": len(chunks),
                "chunk_size": len(piece),
            })
            chunks.append(
                Chunk(content=piece, source=doc.source, index=len(chunks), metadata=meta)
            )
    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """对多篇文档统一切块。"""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return out


def _fmt(chunk: Chunk) -> str:
    """把一个 chunk 格式化成可读文本，用于测试打印。"""
    m = chunk.metadata
    title = m.get("title") or "(无标题)"
    tag = " / ".join(
        x for x in (
            m.get("audience"), m.get("category"), m.get("country"),
            m.get("school"), m.get("program"),
        ) if x
    )
    return (
        f"[{tag}] {title}  |  块长 {m['chunk_size']} 字\n"
        f"    来源: {m.get('source')}\n"
        f"    内容: {chunk.content[:90].replace(chr(10), ' ')}..."
    )


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parents[2] / "data"

    docs = load_directory(data_root)
    chunks = chunk_documents(docs)

    sizes = [c.metadata["chunk_size"] for c in chunks]
    print("=" * 56)
    print(f"分块测试结果：{len(docs)} 篇文档 → {len(chunks)} 个 chunk")
    print("=" * 56)
    print(f"块长：平均 {sum(sizes) // len(sizes)} 字 | 最小 {min(sizes)} | 最大 {max(sizes)}")

    cat_counts = Counter(c.metadata["category"] for c in chunks)
    print("\n各主题 chunk 数：")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<6} {n} 个")

    aud_counts = Counter(c.metadata.get("audience") for c in chunks)
    print("\n各人群 chunk 数：")
    for aud, n in sorted(aud_counts.items(), key=lambda x: (x[0] is None, str(x[0]))):
        print(f"  {str(aud):<6} {n} 个")

    print("\n" + "-" * 56)
    print("抽样展示 3 个 chunk：")
    print("-" * 56)
    for i in (0, len(chunks) // 2, len(chunks) - 1):
        print(_fmt(chunks[i]))
        print()
