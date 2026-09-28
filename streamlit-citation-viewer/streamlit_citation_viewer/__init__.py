"""Streamlit 2.0 component for inline RAG citations."""

from typing import Any

import streamlit as st


_citation_viewer = st.components.v2.component(
    "streamlit-citation-viewer.streamlit_citation_viewer",
    js="index-*.js",
    css="index-*.css",
    html='<div class="react-root"></div>',
)


def streamlit_citation_viewer(
    answer: str,
    sources: list[dict[str, Any]],
    *,
    key: str | None = None,
):
    """Render Markdown with clickable, inline RAG citation previews."""
    return _citation_viewer(
        key=key,
        data={"answer": answer, "sources": sources},
    )
