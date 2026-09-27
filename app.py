"""面向用户的 RAG 网页入口。"""

import streamlit as st


st.set_page_config(
    page_title="资料问答助手",
    page_icon="📚",
)

st.title("📚 资料问答助手")
st.write("根据本地知识库回答问题，并展示引用来源。")

question = st.text_input(
    "请输入问题",
    placeholder="例如：误删笔记后怎么恢复？",
)

if st.button("发送", type="primary"):
    if question.strip():
        st.success(f"收到问题：{question}")
    else:
        st.warning("请先输入问题。")