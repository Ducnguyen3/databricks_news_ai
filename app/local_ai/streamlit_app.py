from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings
from app.local_ai.pipeline import create_embedding_model, create_rag_service, create_vector_store, index_articles
from app.utils.logging import configure_logging


def main() -> None:
    load_dotenv()
    configure_logging("INFO")
    settings = load_settings()

    st.set_page_config(
        page_title="Databricks News AI Chatbot",
        layout="wide",
    )

    st.title("Databricks News AI Chatbot")
    st.caption("Load cleaned articles from Databricks, index them into local ChromaDB, then ask questions via Ollama.")

    embedding_model = create_embedding_model(settings.local_ai)
    vector_store = create_vector_store(settings.local_ai)
    rag_service = create_rag_service(settings.local_ai, embedding_model, vector_store)

    with st.sidebar:
        st.header("Configuration")
        _render_env_status(settings.local_ai.databricks_articles_table)
        st.divider()

        limit = st.number_input("Article limit", min_value=1, value=100, step=10)
        top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=settings.local_ai.rag_top_k)
        chunk_size = st.number_input("Chunk size", min_value=200, value=settings.local_ai.chunk_size, step=50)
        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=max(0, int(chunk_size) - 1),
            value=min(settings.local_ai.chunk_overlap, max(0, int(chunk_size) - 1)),
            step=25,
        )
        reset_index = st.checkbox("Reset index before indexing", value=False)

        index_col, reset_col = st.columns(2)
        with index_col:
            if st.button("Build Index", use_container_width=True):
                _handle_indexing(
                    settings=settings.local_ai,
                    embedding_model=embedding_model,
                    vector_store=vector_store,
                    limit=int(limit),
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    reset_index=reset_index,
                )
        with reset_col:
            if st.button("Reset Only", use_container_width=True):
                vector_store.reset_collection()
                st.session_state["last_index_result"] = None
                st.success("Chroma collection has been reset.")

        st.metric("Indexed chunks", vector_store.count())

    _render_last_index_result()

    question = st.text_area(
        "Question",
        value=st.session_state.get("question_input", "Tin AI moi nhat co gi dang chu y?"),
        height=120,
        placeholder="Nhap cau hoi bang tieng Viet...",
    )
    st.session_state["question_input"] = question

    ask_col, sample_col = st.columns([1, 3])
    with ask_col:
        ask = st.button("Ask", type="primary", use_container_width=True)
    with sample_col:
        st.caption("Can index at least once before asking. Existing Chroma data is reused between runs.")

    if ask:
        if vector_store.count() == 0:
            st.error("Chroma index is empty. Build the index first.")
        elif not question.strip():
            st.error("Question cannot be empty.")
        else:
            with st.spinner("Generating answer..."):
                result = rag_service.answer(question.strip(), top_k=top_k)
            _render_answer(result)


def _handle_indexing(
    settings,
    embedding_model,
    vector_store,
    limit: int,
    chunk_size: int,
    chunk_overlap: int,
    reset_index: bool,
) -> None:
    try:
        with st.spinner("Loading articles from Databricks and building Chroma index..."):
            result = index_articles(
                settings=settings,
                embedding_model=embedding_model,
                vector_store=vector_store,
                limit=limit,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                reset_index=reset_index,
            )
    except Exception as exc:
        st.error(str(exc))
        return

    st.session_state["last_index_result"] = result
    st.success("Indexing completed.")


def _render_env_status(articles_table: str) -> None:
    st.subheader("Environment")
    st.write(f"Articles table: `{articles_table}`")
    required_keys = [
        "DATABRICKS_SERVER_HOSTNAME",
        "DATABRICKS_HTTP_PATH",
        "DATABRICKS_TOKEN",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    ]
    for key in required_keys:
        is_set = bool(os.getenv(key))
        st.write(f"{'OK' if is_set else 'Missing'} `{key}`")


def _render_last_index_result() -> None:
    result = st.session_state.get("last_index_result")
    if result is None:
        return

    st.subheader("Latest Index Run")
    first, second, third, fourth = st.columns(4)
    first.metric("Articles", result.articles_loaded)
    second.metric("Chunks Created", result.chunks_created)
    third.metric("Chunks Upserted", result.chunks_upserted)
    fourth.metric("Index Size", result.index_size)


def _render_answer(result: dict[str, object]) -> None:
    st.subheader("Answer")
    st.write(str(result.get("answer") or ""))

    sources = result.get("sources", [])
    st.subheader("Sources")
    if not isinstance(sources, list) or not sources:
        st.info("No sources returned.")
        return

    for source in sources:
        if not isinstance(source, dict):
            continue
        title = str(source.get("title") or "Untitled")
        source_name = str(source.get("source") or "unknown")
        url = str(source.get("url") or "")
        category = str(source.get("category") or "")
        published_at = str(source.get("published_at") or "")
        chunk_id = str(source.get("chunk_id") or "")
        with st.expander(f"{title} | {source_name}", expanded=False):
            st.write(f"URL: {url}")
            st.write(f"Category: {category}")
            st.write(f"Published at: {published_at}")
            st.write(f"Chunk ID: {chunk_id}")


if __name__ == "__main__":
    main()
