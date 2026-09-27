"""命令行入口：python main.py ingest / search / ask。"""

import argparse
import sys

from openai import APIConnectionError, APIStatusError

from config import load_settings
from rag import RAG


def show_sources(sources: list[dict]):
    for i, item in enumerate(sources, 1):
        # 余弦距离越小越相似；这个数字不是“回答正确率”。
        print(f"\n[{i}] {item['source']} · 片段 {item['chunk']} "
              f"· 距离 {item['distance']:.4f}")
        print(item["text"])


def main():
    parser = argparse.ArgumentParser(description="最小 RAG：入库、检索、资料问答")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="将 docs 中的资料增量同步到本地索引")
    for name, help_text in [("search", "只检索原文"), ("ask", "解析问题、检索并回答")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("question", help="问题（请用引号包起来）")
    args = parser.parse_args()

    try:
        rag = RAG(load_settings())
        if args.command == "ingest":
            print("正在扫描资料并同步索引……", flush=True)
            result = rag.ingest()
            mode = "全量重建" if result.rebuilt else "增量同步"
            print(
                f"{mode}完成：新增 {result.added_files}，修改 {result.modified_files}，"
                f"删除 {result.deleted_files}，未变化 {result.unchanged_files}；"
                f"本次生成 {result.embedded_chunks} 个向量。\n"
                f"当前索引：{result.total_files} 个文件，{result.total_chunks} 个片段。"
            )
        elif args.command == "search":
            show_sources(rag.search(args.question))
        else:
            print("正在解析问题、检索资料并生成回答……", flush=True)
            result = rag.ask(args.question)
            print(f"\n检索问题：{result['query']}")
            show_sources(result["sources"])
            print(f"\n回答：\n{result['answer']}")
    except APIStatusError as error:
        # 保留服务商的错误说明，便于定位无效模型、参数或额度问题。
        detail = error.body
        if isinstance(detail, dict):
            detail = detail.get("error", detail)
        if isinstance(detail, dict):
            detail = detail.get("message", "")
        print(f"模型接口返回 HTTP {error.status_code}：{detail or '请求失败'}。"
              "请检查 .env 中的 Key、地址、"
              "模型名称和服务额度。", file=sys.stderr)
        return 1
    except APIConnectionError:
        print("模型接口连接失败或超时，请检查网络和 BASE_URL。", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
