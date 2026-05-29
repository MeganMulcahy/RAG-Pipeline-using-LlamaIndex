"""
Streamlit UI for the RAG pipeline.

Run:
  streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from persist import PersistStore
from rag_pipeline import dedupe_source_nodes, ingest_file, load_persistent_index, rewrite_query

st.set_page_config(page_title="Document RAG", layout="wide")
st.title("Document RAG")

if "store" not in st.session_state:
    st.session_state.store = PersistStore()
if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []

store: PersistStore = st.session_state.store

with st.sidebar:
    st.subheader("Documents")
    st.caption(f"{store.chunk_count()} chunk(s) in Chroma")

    uploaded = st.file_uploader(
        "Upload a file",
        type=None,
        help="PDF, Word, Excel, slides, HTML, txt, images, etc.",
    )

    if st.button("Load saved index", use_container_width=True):
        try:
            _, engine = load_persistent_index(store)
            st.session_state.engine = engine
            st.success(f"Loaded {store.chunk_count()} chunk(s) from Chroma.")
        except FileNotFoundError as e:
            st.error(str(e))

    if uploaded and st.button("Ingest upload", use_container_width=True):
        suffix = Path(uploaded.name).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            with st.spinner(f"Ingesting {uploaded.name}…"):
                _, engine, file_hash = ingest_file(tmp_path, store)
            st.session_state.engine = engine
            st.success(f"Ready — hash {file_hash[:16]}…")
        except Exception as e:
            st.error(f"Ingest failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.engine is None:
        reply = "Upload and ingest a file first, or load the saved Chroma index."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                search_query = rewrite_query(prompt)
                response = st.session_state.engine.query(search_query)
                answer = str(response).strip()
                sources = []
                for node in dedupe_source_nodes(response.source_nodes):
                    m = node.metadata
                    sources.append(
                        f"p.{m.get('page_start')}–{m.get('page_end')} · "
                        f"{m.get('file_name')} v{m.get('version_num')} · {m.get('doc_id')}"
                    )
                if sources:
                    answer += "\n\n**Sources:** " + " | ".join(sources)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
