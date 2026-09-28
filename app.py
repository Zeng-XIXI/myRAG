"""面向用户的 RAG 网页入口。"""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
import streamlit as st

from config import load_settings
from rag import RAG
from streamlit_citation_viewer import streamlit_citation_viewer

st.set_page_config(
    page_title="资料问答助手",
    page_icon="📚",
    layout="wide",
)

"""计算时间流逝了多少"""
def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 1)

@st.cache_resource
def get_rag() -> RAG:
    """创建并复用 RAG，避免每次页面刷新都重新连接向量库。"""
    return RAG(load_settings())

"""在弹窗中预览 Markdown 或文本文件。"""
@st.dialog("文件预览",width="large",icon=":material/preview:")
def preview_file(path: Path, docs_dir: Path):

    try:
        relative_path = path.relative_to(docs_dir)
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        st.error(f"无法预览文件：{error}")
        return

    st.subheader(str(relative_path))
    st.caption(f"共 {len(content)} 个字符")

    if path.suffix.lower() == ".md":
        st.markdown(content)
    else:
        st.text_area(
            "文件内容",
            value=content,
            height=500,
            disabled=True,
        )

"""为引用预览补充原文件，并标记文件是否仍与索引一致。"""
def sources_with_file_content(sources: list[dict]) -> list[dict]:
    docs_dir = load_settings().docs_dir.resolve()
    prepared = []

    for source in sources:
        item = dict(source)
        source_path = (docs_dir / source["source"]).resolve()
        file_available = source_path.is_relative_to(docs_dir) and source_path.is_file()
        item["file_available"] = file_available
        item["is_markdown"] = source_path.suffix.lower() == ".md"

        if file_available:
            try:
                content = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                item["file_available"] = False
            else:
                item["file_content"] = content
                item["file_current"] = (
                    sha256(content.encode("utf-8")).hexdigest()
                    == source.get("file_hash")
                )

        prepared.append(item)

    return prepared

"""展示一次问答的阶段耗时。"""
def show_timings(timings: dict):
    if not timings:
        return

    labels = [
        ("RAG 准备", "rag_prepare_ms"),
        ("索引检查", "index_check_ms"),
        ("问题改写", "rewrite_ms"),
        ("Query Embedding", "embedding_ms"),
        ("向量检索", "vector_search_ms"),
        ("Context 组装", "context_build_ms"),
        ("答案生成", "answer_generation_ms"),
        ("引用文件准备", "preview_prepare_ms"),
    ]

    total_ms = timings.get("total_ms", 0)

    with st.expander(
        f"耗时详情 · {total_ms / 1000:.2f} 秒",
        icon=":material/timer:",
    ):
        rows = [
            {
                "环节": label,
                "耗时": f"{timings[key] / 1000:.3f} 秒",
                "占总耗时": (
                    f"{timings[key] / total_ms * 100:.1f}%"
                    if total_ms
                    else "—"
                ),
            }
            for label, key in labels
            if key in timings
        ]

        st.table(rows)

if "messages" not in st.session_state:
    st.session_state.messages = []


chat_column, files_column = st.columns(
    [3, 1],
    gap="large",
)


with chat_column:
    st.title("资料问答助手")
    st.write("根据本地知识库回答问题，并展示引用来源。")
    st.info(
        "注意：当前每次问答相互独立，聊天记录仅用于展示，"
        "模型不会参考之前的对话。"
    )

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            sources = message.get("sources", [])
            if message["role"] == "assistant" and sources:
                preview_sources = message.get("preview_sources")

                if preview_sources is None:
                    preview_sources = sources_with_file_content(sources)

                streamlit_citation_viewer(
                    message["content"],
                    preview_sources,
                    key=f"assistant_message_{message_index}",
                )

                show_timings(message.get("timings", {}))
            else:
                st.markdown(message["content"])

    question = st.chat_input(
        "请输入问题",
        submit_mode="disable",
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                request_start = perf_counter()

                rag_prepare_start = perf_counter()
                rag = get_rag()
                rag_prepare_ms = elapsed_ms(rag_prepare_start)

                with st.spinner("正在检索资料并生成回答……"):
                    result = rag.ask(question)

                preview_start = perf_counter()
                preview_sources = sources_with_file_content(result["sources"])
                preview_prepare_ms = elapsed_ms(preview_start)

                timings = result["timings"]
                timings["rag_prepare_ms"] = rag_prepare_ms
                timings["preview_prepare_ms"] = preview_prepare_ms
                timings["total_ms"] = elapsed_ms(request_start)

                st.markdown(result["answer"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "query": result["query"],
                        "sources": result["sources"],
                        "preview_sources": preview_sources,
                        "timings": timings,
                    }
                )

                st.rerun()
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"请求失败：{error}")


with files_column:
    st.header("知识库文件")

    settings = load_settings()

    if st.button(
        "同步知识库",
        icon=":material/sync:",
        width="stretch",
    ):
        try:
            with st.spinner("正在扫描资料并同步向量库……"):
                sync_result = get_rag().ingest()

            mode = "全量重建" if sync_result.rebuilt else "增量同步"

            st.success(
                f"{mode}完成："
                f"新增 {sync_result.added_files}，"
                f"修改 {sync_result.modified_files}，"
                f"删除 {sync_result.deleted_files}，"
                f"未变化 {sync_result.unchanged_files}。"
            )
            st.caption(
                f"本次生成 {sync_result.embedded_chunks} 个向量；"
                f"当前共 {sync_result.total_files} 个文件、"
                f"{sync_result.total_chunks} 个片段。"
            )
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f"同步失败：{error}")

    uploaded_file = st.file_uploader(
        "添加资料",
        type=["md", "txt"],
        help="仅支持 UTF-8 编码的 Markdown 和文本文件。",
        max_upload_size=5,
    )

    if uploaded_file is not None:
        if st.button(
            "添加到知识库",
            type="primary",
            icon=":material/upload_file:",
        ):
            # Path(...).name 会去掉上传文件中可能携带的目录部分。
            filename = Path(uploaded_file.name).name
            suffix = Path(filename).suffix.lower()
            destination = settings.docs_dir / filename

            if suffix not in {".md", ".txt"}:
                st.error("仅支持 `.md` 和 `.txt` 文件。")
            elif destination.exists():
                st.error(f"已存在同名文件：{filename}")
            else:
                try:
                    text = uploaded_file.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    st.error("文件不是有效的 UTF-8 编码，请转换编码后重试。")
                else:
                    if not text.strip():
                        st.warning("文件内容为空，没有保存。")
                    else:
                        destination.write_text(text, encoding="utf-8")
                        st.success(f"已添加：{filename}")
                        st.warning("新文件尚未同步到向量库。")

    files = sorted(
        path
        for path in settings.docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )

    if files:
        st.caption(f"共 {len(files)} 个文件")

        for file_index, path in enumerate(files):
            relative_path = path.relative_to(settings.docs_dir)

            with st.container(border=True):
                st.markdown(f":material/description: `{relative_path}`")

                if st.button(
                        "预览",
                        icon=":material/visibility:",
                        width="stretch",
                        key=f"preview_file_{file_index}",
                ):
                    preview_file(path, settings.docs_dir)
    else:
        st.warning("知识库中还没有 `.md` 或 `.txt` 文件。")
