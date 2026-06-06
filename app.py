import streamlit as st
from utils.embedder import load_vectorstore
from utils.chain import build_rag_chain, ask

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Intelligent Document Q&A")
st.caption("Ask questions about your uploaded PDF documents")

@st.cache_resource
def load_chain():
    vs = load_vectorstore()
    return build_rag_chain(vs)

try:
    chain = load_chain()
    st.success("FAISS index loaded. Ready to answer questions!")
except FileNotFoundError:
    st.error("No index found. Please run `python main.py` first.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(f"📌 {src}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = ask(chain, prompt, st.session_state.messages)
        st.write(result["answer"])
        if result["sources"]:
            with st.expander("Sources used"):
                for src in result["sources"]:
                    st.caption(f"📌 {src}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })