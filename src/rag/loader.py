"""文档加载：遍历 data/ 读取 md/txt，包装成带元数据的 Document。

目录约定（层级即元数据，最多五级）：
    data/院校项目/国家/学校/专业/人群.md  → category=院校项目, country/school/program, audience=人群
    data/院校项目/国家/学校/文档.md        → program=None（学校级通用，如 _院校清单）
    data/院校项目/国家/通用文档.md          → school/program=None（国家级通用内容）
    data/考试/大类/具体考试/文档.md       → category=考试, exam_category=大类, exam=具体考试
    data/留学生活/国家/文档.md             → category=留学生活, country=国家（签证并入生活）

人群（audience）来自专业目录下的文件名（如 硕士.md → 硕士），不再来自文档内章节。

自研部分（答辩可讲）：
- 从目录层级自动推导 category / country / school / program / audience，无需重复标注
- 解析可选 YAML frontmatter，补充 url / updated_at 等
- 用相对路径的 md5 生成确定性 parent_id

用法：
    python -m src.rag.loader    # 打印知识库目录结构 + 加载统计（作业一截图①）
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# 一级目录三分类（用户 2026-09-04 定）：
# 1) 院校项目 —— 国家/学校/专业，人群 = 专业目录下的文件名
# 2) 考试     —— 大类（语言考试/专业考试/其他考试）→ 具体考试（雅思/托福/GRE…），备考规划 + 题目指导
# 3) 留学生活 —— 按国家分（签证行前 + 落地生活，二者已合并）
CATEGORIES = ["院校项目", "考试", "留学生活"]


@dataclass
class Document:
    """一篇文档及其元信息。"""
    content: str
    source: str                       # 相对 data/ 的路径，用于追溯出处
    metadata: dict = field(default_factory=dict)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析开头 `--- ... ---` 的元数据块，返回 (metadata, 正文)。"""
    meta: dict = {}
    if not text.startswith("---"):
        return meta, text
    end = text.find("\n---", 3)
    if end == -1:
        return meta, text
    block = text[3:end]
    body = text[end + 4:].lstrip("\n")
    for line in block.splitlines():
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def _derive_meta(rel: str) -> dict:
    """从相对路径层级推导 category / country / school / program / audience / exam。

    院校项目：院校项目/国家/学校/专业/人群.md → country/school/program, audience=人群
    考试：考试/大类/具体考试/文档.md         → exam_category=大类, exam=具体考试
    留学生活：留学生活/国家/文档.md          → country
    """
    parts = rel.split("/")
    meta = {"category": parts[0] if parts and parts[0] in CATEGORIES else "其他"}
    dirs = parts[1:-1]   # 类别与文件名之间的目录层级
    top = parts[0]
    if top == "院校项目":
        meta["country"] = dirs[0] if len(dirs) >= 1 else None
        meta["school"] = dirs[1] if len(dirs) >= 2 else None
        meta["program"] = dirs[2] if len(dirs) >= 3 else None
        # 人群 = 专业目录下的文件名（如 硕士.md → 硕士）
        meta["audience"] = Path(parts[-1]).stem if len(dirs) >= 3 else None
    elif top == "考试":
        # 考试/大类/具体考试/文档.md → exam_category=大类, exam=具体考试
        # 兼容旧式 考试/考试名/文档.md → exam=考试名
        meta["exam_category"] = dirs[0] if dirs else None
        meta["exam"] = dirs[1] if len(dirs) >= 2 else (dirs[0] if dirs else None)
    elif top == "留学生活":
        meta["country"] = dirs[0] if dirs else None
    return meta


def load_file(path: str | Path, data_root: str | Path | None = None) -> Document:
    """按扩展名加载单个文件，生成带元数据的 Document。"""
    p = Path(path)
    root = Path(data_root) if data_root else p.parent
    text = p.read_text(encoding="utf-8")

    fm, body = _parse_frontmatter(text)

    rel = p.resolve().relative_to(root.resolve()).as_posix()
    meta = _derive_meta(rel)          # 目录层级 → category/country/school/program/audience
    meta.update(fm)                   # frontmatter 补充/覆盖 url/updated_at 等
    if "updated_at" not in meta:      # 原则①：每条信息需标注更新时间
        print(f"[警告] 缺 updated_at（信息更新时间）：{rel}")
    meta.update({
        "source": rel,
        "doc_title": p.stem,
        "parent_id": hashlib.md5(rel.encode("utf-8")).hexdigest(),
    })
    return Document(content=body.strip(), source=rel, metadata=meta)


def _is_draft(path: Path) -> bool:
    """判断是否为骨架占位文档（frontmatter 含 draft: true）。

    知识库为 90 所学校按人群建了骨架文档（本科/硕士/博士及博士后/访问学者/交换生），
    尚未核实真实内容。它们不应进入检索，后续填了真实数据并移除 draft 标记后自动生效。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return False
    if not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    if end == -1:
        return False
    return "draft: true" in text[3:end]


def load_directory(data_dir: str | Path) -> list[Document]:
    """加载目录下所有 md / txt 文档（跳过 README 与骨架占位文档）。"""
    root = Path(data_dir)
    docs: list[Document] = []
    for suffix in ("*.md", "*.txt"):
        for f in sorted(root.rglob(suffix)):
            if f.name.lower() == "readme.md":   # 跳过数据目录说明文件
                continue
            if _is_draft(f):                    # 跳过待填充骨架文档
                continue
            try:
                docs.append(load_file(f, data_root=root))
            except Exception as e:  # noqa: BLE001
                print(f"[跳过] {f.name}: {e}")
    return docs


def _print_tree(d: Path, prefix: str = "") -> None:
    """递归打印目录树（用于知识库目录截图）。"""
    entries = sorted(
        [e for e in d.iterdir() if not e.name.startswith(".")],
        key=lambda e: (e.is_file(), e.name),
    )
    for i, e in enumerate(entries):
        last = i == len(entries) - 1
        branch = "└── " if last else "├── "
        print(f"{prefix}{branch}{e.name}")
        if e.is_dir():
            _print_tree(e, prefix + ("    " if last else "│   "))


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parents[2] / "data"

    print("=" * 56)
    print("知识库目录结构（data/）")
    print("=" * 56)
    _print_tree(data_root)

    docs = load_directory(data_root)
    cat_counts = Counter(d.metadata["category"] for d in docs)
    country_counts = Counter(d.metadata.get("country", "—") for d in docs)
    aud_counts = Counter(d.metadata.get("audience") for d in docs)
    program_count = sum(1 for d in docs if d.metadata.get("program"))

    print()
    print("=" * 56)
    print(f"共加载 {len(docs)} 篇文档")
    print("按主题：")
    for cat in CATEGORIES + ["其他"]:
        if cat_counts.get(cat):
            print(f"  {cat:<6} {cat_counts[cat]} 篇")
    print("按人群：")
    for aud, n in sorted(aud_counts.items(), key=lambda x: (x[0] is None, str(x[0]))):
        print(f"  {str(aud):<6} {n} 篇")
    print("按国家：")
    for c, n in sorted(country_counts.items()):
        print(f"  {c:<6} {n} 篇")
    print(f"其中挂到具体专业的（院校项目/国家/学校/专业/人群）：{program_count} 篇")
    print("=" * 56)
