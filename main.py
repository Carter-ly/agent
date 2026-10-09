"""命令行入口：组装 RAG + Agent，提供交互式问答。

用法（在 proj/ 目录下）：
    python main.py
    >>> UCL 管理学硕士的语言要求是什么？
    >>> 输入 exit 退出

组装链路（对应作业三「生成链路」）：
    Retriever（检索） -> Agent（ReAct 决策） -> LLM（DeepSeek 生成）
"""
from __future__ import annotations

import time
from pathlib import Path

from src.agent.agent import Agent
from src.agent.tools import make_retrieve_tool
from src.llm.base import get_llm
from src.rag.retriever import Retriever


def build_agent() -> Agent:
    """构建检索增强问答智能体。"""
    data_dir = Path(__file__).resolve().parent / "data"

    print("正在加载 Embedding 模型并索引知识库（首次约 1 分钟）...")
    t0 = time.time()
    retriever = Retriever.from_config()
    retriever.index_directory(str(data_dir))
    print(f"知识库索引完成：{len(retriever.store)} 个 chunk，耗时 {time.time() - t0:.1f}s\n")

    llm = get_llm()
    retrieve_tool = make_retrieve_tool(retriever)
    return Agent(llm=llm, tools=[retrieve_tool])


def main() -> None:
    agent = build_agent()
    print("=" * 60)
    print("留学全周期智能助手（RAG + Agent）已就绪，输入问题开始问答。")
    print("输入 exit 或 Ctrl+C 退出。")
    print("=" * 60)

    history: list[dict[str, str]] = []
    while True:
        try:
            question = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        print("\n（思考中...）")
        t0 = time.time()
        answer = agent.run(question, history)
        print(f"\n[回答·{time.time() - t0:.1f}s]")
        print(answer)
        # 展示来源
        if agent.last_sources:
            print("\n[来源]")
            for s in agent.last_sources:
                print(f"  - {s}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()