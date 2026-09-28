import streamlit as st

from streamlit_citation_viewer import streamlit_citation_viewer


st.title("Inline citation viewer")

streamlit_citation_viewer(
    "RAG 在推理时把外部知识放入 Context。[1] 检索和重排同样重要。[2]",
    [
        {
            "source": "intro.md",
            "chunk": 1,
            "distance": 0.12,
            "text": "RAG 不改变模型参数。",
            "start_line": 2,
            "end_line": 3,
            "file_content": "# RAG\nRAG 不改变模型参数。\n它在推理时注入外部知识。",
            "file_available": True,
            "file_current": True,
            "is_markdown": True,
        },
        {
            "source": "retrieval.txt",
            "chunk": 2,
            "distance": 0.23,
            "text": "企业级 RAG 的重点是检索和重排。",
            "start_line": 1,
            "end_line": 1,
            "file_content": "企业级 RAG 的重点是检索和重排。",
            "file_available": True,
            "file_current": True,
            "is_markdown": False,
        },
    ],
    key="citation-example",
)
