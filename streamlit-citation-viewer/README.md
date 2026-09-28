# streamlit-citation-viewer

Streamlit Custom Component v2 for inline RAG citations.

It renders Markdown answers, turns citations such as `[1]` into inline buttons,
and opens a resizable source preview anchored to the citation. The preview can
highlight indexed source lines or render the complete Markdown file.

## Development

Build the frontend:

```bash
cd streamlit_citation_viewer/frontend
npm install
npm run build
```

Install the Python package from this directory:

```bash
uv pip install -e . --no-deps
```

Run the no-model example:

```bash
streamlit run example.py
```

## Usage

```python
from streamlit_citation_viewer import streamlit_citation_viewer

streamlit_citation_viewer(
    "RAG 在推理时注入外部知识。[1]",
    [
        {
            "source": "intro.md",
            "chunk": 1,
            "distance": 0.12,
            "text": "RAG 在推理时注入外部知识。",
            "start_line": 2,
            "end_line": 2,
            "file_content": "# RAG\nRAG 在推理时注入外部知识。",
            "file_available": True,
            "file_current": True,
            "is_markdown": True,
        }
    ],
    key="answer-1",
)
```
