"""院校清单爬虫：批量抓取全球院校清单（校名 + 国家），生成知识库目录骨架。

数据源（用户 2026-09-04 定，唯一权威来源，不再用 QS 等第三方排名）：
- 教育部留学服务中心「认证院校查询」（yxcx.cscse.edu.cn，官方权威，学历认证唯一依据）
  - POST /api/xlxwrzz/xlxwrz/getAllCountry           → 146 个国家/地区
  - POST /api/xlxwrzz/xlxwrz/getUniversityListOrPage → 按国家分页拉院校
  - 字段：CHINESE_NAME（中文名）/ ENGLISH_NAME（英文名）/ COUNTRY（国家）
  - 注意：该服务器 TLS 用了 legacy renegotiation，OpenSSL 3 默认禁掉，
    需对 requests 挂一个 ssl.OP_LEGACY_SERVER_CONNECT 的 HTTPS adapter。

产出（写入 crawler_output/）：
    schools.json —— 教育部认证院校名单全量：{zh, en, country}

用法：
    python -m src.crawler fetch      # 抓取教育部名单，写 crawler_output/schools.json
    python -m src.crawler skeleton   # 按清单生成 data/院校项目/{国家}/ 骨架

设计要点（对齐数据规范）：
- 只爬「院校清单」（校名+国家），不爬每校申请详情；详情仍需逐校联网核实，避免造假。
- 以「中文名 + 国家」精确去重，绝不做模糊匹配（避免数据造假）。
- 国家名统一用教育部规范简称（美国/英国/澳大利亚/…），映射表只处理存量目录的
  旧写法（澳大利亚→澳洲 之类），保证与已核实数据一致。
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

MOE_BASE = "https://yxcx.cscse.edu.cn/api/xlxwrzz"

PROJ_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJ_ROOT / "crawler_output"
DATA_DIR = PROJ_ROOT / "data"

# 国家名映射：教育部规范简称 → 存量目录旧写法（保证与已核实数据一致）
COUNTRY_ALIAS = {
    "澳大利亚": "澳洲",
}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"


# --------------------------------------------------------------------------- #
# 教育部名单：需要 legacy renegotiation 的 HTTPS adapter
# --------------------------------------------------------------------------- #

class LegacySSLAdapter(HTTPAdapter):
    """OpenSSL 3 默认禁用 legacy renegotiation，旧服务器 yxcx 需要它。"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT   # 0x4，允许旧式重协商
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _moe_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", LegacySSLAdapter())
    s.headers.update({
        "Content-Type": "application/json;charset=utf-8",
        "User-Agent": UA,
    })
    return s


# --------------------------------------------------------------------------- #
# 抓取：教育部名单
# --------------------------------------------------------------------------- #

def _moe_post(session: requests.Session, path: str, payload: dict) -> dict:
    """POST 到教育部接口，带重试。返回解析后的 JSON。"""
    url = f"{MOE_BASE}/{path}"
    for attempt in range(3):
        try:
            r = session.post(url, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_moe_countries(session: requests.Session) -> list[str]:
    """拉全部国家/地区名（146 个）。"""
    d = _moe_post(session, "xlxwrz/getAllCountry",
                  {"currentPage": 1, "pageSize": 5000, "countryIndex": ""})
    return [x["COUNTRY"] for x in d.get("data", [])]


def fetch_moe_schools(session: requests.Session, country: str) -> list[dict]:
    """拉某国全部院校，自动翻页。返回 [{zh, en, country}]。"""
    out: list[dict] = []
    page, page_size = 1, 500
    while True:
        d = _moe_post(session, "xlxwrz/getUniversityListOrPage", {
            "currentPage": page,
            "pageSize": page_size,
            "country": country,
            "universityIndex": "",
            "universityName": "",
        })
        for x in d.get("data", []):
            zh = (x.get("CHINESE_NAME") or "").strip()
            en = (x.get("ENGLISH_NAME") or "").strip()
            if zh:
                out.append({"zh": zh, "en": en, "country": country})
        total = int(d.get("total", 0))
        if page * page_size >= total or not d.get("data"):
            break
        page += 1
    return out


# --------------------------------------------------------------------------- #
# 去重与落盘
# --------------------------------------------------------------------------- #

def dedup_schools(schools: list[dict]) -> list[dict]:
    """以「中文名 + 国家」精确去重，保持首次出现顺序。"""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for s in schools:
        key = (s["zh"], s["country"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {path}（{len(data)} 条）")


def cmd_fetch() -> None:
    t0 = time.time()
    session = _moe_session()

    print("=" * 56)
    print("抓取教育部留学服务中心「认证院校查询」名单")
    print("=" * 56)
    countries = fetch_moe_countries(session)
    print(f"国家/地区：{len(countries)} 个")
    schools: list[dict] = []
    for i, c in enumerate(countries, 1):
        try:
            got = fetch_moe_schools(session, c)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(countries)}] {c} 抓取失败：{e}")
            continue
        schools.extend(got)
        print(f"  [{i}/{len(countries)}] {c}：{len(got)} 所")
        time.sleep(0.2)

    schools = dedup_schools(schools)
    schools.sort(key=lambda x: (x["country"], x["zh"]))
    print(f"\n教育部认证院校合计：{len(schools)} 所（去重后）")

    save_json(schools, OUT_DIR / "schools.json")
    print(f"\n耗时 {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------- #
# 骨架生成
# --------------------------------------------------------------------------- #

def _safe_name(name: str) -> str:
    """把校名转成安全的目录名（替换文件系统保留字符）。"""
    for ch in '/\\:*?"<>|':
        name = name.replace(ch, "·")
    return name.strip() or "未命名"


def cmd_skeleton() -> None:
    schools = json.loads((OUT_DIR / "schools.json").read_text(encoding="utf-8"))

    # 国家 → 院校列表
    by_country: dict[str, list[dict]] = {}
    for s in schools:
        by_country.setdefault(s["country"], []).append(s)

    total_schools = 0
    for country, country_schools in by_country.items():
        cdir = COUNTRY_ALIAS.get(country, country)
        base = DATA_DIR / "院校项目" / cdir
        base.mkdir(parents=True, exist_ok=True)
        # 每个国家目录放一份 _院校清单.md（真实数据，标 updated_at + 来源）
        write_school_index(base, country, country_schools)
        # 每个学校建空目录；学院/专业在逐校填充数据时再建
        for s in country_schools:
            (base / _safe_name(s["zh"])).mkdir(parents=True, exist_ok=True)
            total_schools += 1

    print(f"已生成 {len(by_country)} 个国家目录、{total_schools} 个学校目录（院校项目/国家/学校）")


def write_school_index(country_dir: Path, country: str, schools: list[dict]) -> None:
    """在某国家目录下写 _院校清单.md（该国院校名单索引）。

    只含校名 + 国家，是教育部官方认证名单；不含任何申请要求，后续逐校填充。
    """
    lines = [
        "---",
        "updated_at: 2026-09-04",
        "source: 教育部留学服务中心「认证院校查询」(yxcx.cscse.edu.cn)",
        "---",
        "",
        f"# {country} 认证院校清单",
        "",
        "> 本清单来自教育部留学服务中心「认证院校查询」官方权威数据，",
        "> 抓取时间 2026-09-04。仅收录校名与国家，申请要求详见各院校单独目录。",
        "",
        "| 序号 | 中文名 | 英文名 |",
        "|------|--------|--------|",
    ]
    for i, s in enumerate(schools, 1):
        en = (s.get("en") or "").replace("|", "/")
        zh = s["zh"].replace("|", "/")
        lines.append(f"| {i} | {zh} | {en} |")
    lines.append("")
    (country_dir / "_院校清单.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="院校清单爬虫 + 目录骨架生成")
    ap.add_argument("cmd", choices=["fetch", "skeleton"],
                    help="fetch=抓取教育部名单；skeleton=按清单生成目录骨架")
    args = ap.parse_args()
    if args.cmd == "fetch":
        cmd_fetch()
    else:
        cmd_skeleton()


if __name__ == "__main__":
    main()
