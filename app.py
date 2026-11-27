
"""Streamlit RAG Chatbot powered by LangChain and OpenRouter."""

from __future__ import annotations

import os
import tempfile
import certifi
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Use certifi's trusted CA certificate bundle for HTTPS connections.
os.environ["SSL_CERT_FILE"] = certifi.where()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_EXTENSIONS = ("pdf", "txt", "md", "docx", "csv")

DEFAULT_CHAT_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "Custom...",
]

DEFAULT_EMBEDDING_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "baai/bge-m3",
    "Custom...",
]


def is_valid_api_key(api_key: str | None) -> bool:
    """Return True if an OpenRouter API key looks configured."""
    if not api_key:
        return False
    cleaned = api_key.strip()
    return bool(cleaned) and cleaned != "your_openrouter_api_key_here"


def build_chat_model(
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> ChatOpenRouter:
    """Initialize a LangChain ChatOpenRouter LLM client."""
    return ChatOpenRouter(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        app_title="LangChain OpenRouter RAG Chatbot",
    )


def build_embeddings(api_key: str, model_name: str) -> OpenAIEmbeddings:
    """Initialize OpenRouter-backed embeddings."""
    return OpenAIEmbeddings(
        model=model_name,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        check_embedding_ctx_length=False,
    )


def load_single_file(file_path: Path, original_name: str) -> list[Document]:
    """Load a single uploaded file."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix == ".csv":
        loader = CSVLoader(str(file_path), encoding="utf-8")
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = original_name

    return docs


def ingest_uploaded_files(
    uploaded_files: Iterable,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Document], list[str]]:
    """Read uploaded files and split their text into chunks."""
    all_docs: list[Document] = []
    skipped_files: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)

        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            dest_path = tmp_root / safe_name
            dest_path.write_bytes(uploaded_file.getvalue())

            file_docs = load_single_file(
                dest_path,
                original_name=safe_name,
            )

            non_empty_docs = [
                d for d in file_docs
                if d.page_content and d.page_content.strip()
            ]

            if not non_empty_docs:
                skipped_files.append(safe_name)
                continue

            all_docs.extend(non_empty_docs)

    if not all_docs:
        return [], skipped_files

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(all_docs)

    return chunks, skipped_files


def format_docs_for_prompt(
    docs_with_scores: list[tuple[Document, float]],
) -> str:
    """Format retrieved chunks for the LLM prompt."""
    formatted_parts: list[str] = []

    for idx, (doc, _score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        location = (
            f"{source}, page {page + 1}"
            if isinstance(page, int)
            else source
        )

        formatted_parts.append(
            f"[{idx}] ({location})\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted_parts)


def to_langchain_messages(messages: list[dict]) -> list[BaseMessage]:
    """Convert chat history into LangChain messages."""
    lc_messages: list[BaseMessage] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return lc_messages


def contextualize_question(
    llm: ChatOpenRouter,
    chat_history: list[BaseMessage],
    question: str,
) -> str:
    """Rewrite a follow-up question for document retrieval."""
    if not chat_history:
        return question

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, "
                "rewrite the question so it is completely standalone "
                "and suitable for semantic document retrieval. "
                "Do NOT answer the question; only return the "
                "standalone search query.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = contextualize_prompt | llm | StrOutputParser()

    rewritten = chain.invoke(
        {
            "chat_history": chat_history,
            "input": question,
        }
    ).strip()

    return rewritten or question


def build_rag_answer_stream(
    llm: ChatOpenRouter,
    context_text: str,
    chat_history: list[BaseMessage],
    question: str,
):
    """Stream a grounded RAG answer."""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Retrieval-Augmented Generation "
                "(RAG) assistant. Answer the user's question using "
                "ONLY the retrieved document context below. "
                "Cite the source file name (and page number if "
                "available) when relevant. If the answer is not "
                "contained in the provided context, clearly state "
                "that you could not find the answer in the uploaded "
                "documents.\n\nRetrieved Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = qa_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "context": context_text,
            "chat_history": chat_history,
            "input": question,
        }
    )


def render_sources(sources: list[dict]) -> None:
    """Display retrieved document sources."""
    if not sources:
        return

    with st.expander(
        f"Retrieved Sources ({len(sources)})",
        expanded=False,
    ):
        for idx, item in enumerate(sources, start=1):
            label = item["source"]

            if item.get("page") is not None:
                label += f" (Page {item['page'] + 1})"

            st.markdown(
                f"**{idx}. {label}** — "
                f"Similarity: `{item['score']:.3f}`"
            )

            st.caption(item["snippet"])


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "messages": [],
        "vector_store": None,
        "indexed_files": [],
        "chunk_count": 0,
        "embedding_model_used": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def select_model_widget(
    label: str,
    options: list[str],
    env_var: str,
) -> str:
    """Render a model selection widget."""
    env_default = os.getenv(env_var, "").strip()
    initial_options = list(options)

    if env_default and env_default not in initial_options:
        initial_options.insert(0, env_default)

    selected = st.selectbox(label, initial_options, index=0)

    if selected == "Custom...":
        return st.text_input(
            f"Enter custom {label.lower()}",
            value=env_default,
        ).strip()

    return selected


def main() -> None:
    st.set_page_config(
        page_title="RAG Chatbot • LangChain & OpenRouter",
        page_icon="📚",
        layout="wide",
    )

    init_session_state()

    st.title("📚 Document RAG Chatbot")
    st.caption(
        "Powered by LangChain (`langchain-openrouter`) and OpenRouter"
    )

    with st.sidebar:
        st.header("🔑 OpenRouter Settings")

        env_key = os.getenv("OPENROUTER_API_KEY", "")

        api_key = st.text_input(
            "OpenRouter API Key",
            value=env_key if is_valid_api_key(env_key) else "",
            type="password",
            placeholder="sk-or-v1-...",
            help=(
                "Loaded from .env if set, or paste your "
                "OpenRouter API key here."
            ),
        ).strip()

        chat_model = select_model_widget(
            "Chat Model",
            DEFAULT_CHAT_MODELS,
            "OPENROUTER_CHAT_MODEL",
        )

        embedding_model = select_model_widget(
            "Embedding Model",
            DEFAULT_EMBEDDING_MODELS,
            "OPENROUTER_EMBEDDING_MODEL",
        )

        with st.expander("⚙️ RAG Parameters", expanded=False):
            temperature = st.slider(
                "Temperature", 0.0, 1.0, 0.2, 0.05
            )

            max_tokens = st.number_input(
                "Max Tokens",
                min_value=128,
                max_value=8192,
                value=1024,
                step=128,
            )

            top_k = st.slider(
                "Top-K Retrieved Chunks", 1, 10, 4
            )

            chunk_size = st.number_input(
                "Chunk Size",
                min_value=200,
                max_value=4000,
                value=1000,
                step=100,
            )

            max_overlap = max(0, int(chunk_size) - 50)

            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=max_overlap,
                value=min(200, max_overlap),
                step=50,
            )

        st.divider()
        st.header("📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, MD, or CSV files",
            type=list(SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )

        col_index, col_clear = st.columns(2)

        process_clicked = col_index.button(
            "Index Files",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        clear_clicked = col_clear.button(
            "Reset Chat & Index",
            use_container_width=True,
        )

        if clear_clicked:
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.indexed_files = []
            st.session_state.chunk_count = 0
            st.session_state.embedding_model_used = None
            st.rerun()

        if process_clicked:
            if not is_valid_api_key(api_key):
                st.error(
                    "Please enter a valid OpenRouter API key before indexing."
                )

            elif not embedding_model:
                st.error(
                    "Please specify an OpenRouter embedding model."
                )

            else:
                with st.spinner(
                    "Reading, chunking, and embedding documents..."
                ):
                    try:
                        chunks, skipped = ingest_uploaded_files(
                            uploaded_files=uploaded_files,
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                        )

                        if not chunks:
                            st.warning(
                                "No readable text was extracted "
                                "from the uploaded files."
                            )

                        else:
                            embeddings = build_embeddings(
                                api_key=api_key,
                                model_name=embedding_model,
                            )

                            vector_store = InMemoryVectorStore(embeddings)

                            vector_store.add_documents(
                                documents=chunks
                            )

                            st.session_state.vector_store = vector_store

                            st.session_state.indexed_files = [
                                f.name
                                for f in uploaded_files
                                if f.name not in skipped
                            ]

                            st.session_state.chunk_count = len(chunks)

                            st.session_state.embedding_model_used = (
                                embedding_model
                            )

                            st.success(
                                f"Indexed {len(chunks)} chunks from "
                                f"{len(st.session_state.indexed_files)} "
                                f"file(s)."
                            )

                            if skipped:
                                st.warning(
                                    "Skipped empty/unreadable file(s): "
                                    + ", ".join(skipped)
                                )

                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")

        if st.session_state.indexed_files:
            st.info(
                f"**Active Knowledge Base**\n\n"
                f"- Files: {', '.join(st.session_state.indexed_files)}\n"
                f"- Chunks: {st.session_state.chunk_count}\n"
                f"- Embeddings: "
                f"`{st.session_state.embedding_model_used}`"
            )

    if not is_valid_api_key(api_key):
        st.info(
            "Add your `OPENROUTER_API_KEY` in `.env` "
            "or the sidebar to get started."
        )

    if st.session_state.vector_store is None:
        st.markdown(
            "### How to use\n"
            "1. Enter your **OpenRouter API Key** in `.env` "
            "or the sidebar.\n"
            "2. Upload one or more **PDF / DOCX / TXT / MD / CSV** "
            "files in the sidebar.\n"
            "3. Click **Index Files** to create embeddings "
            "and populate the vector store.\n"
            "4. Ask questions in the chat box below."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    if prompt_text := st.chat_input(
        "Ask a question about your uploaded documents..."
    ):
        if not is_valid_api_key(api_key):
            st.error(
                "Please provide a valid OpenRouter API key first."
            )
            st.stop()

        if st.session_state.vector_store is None:
            st.warning(
                "Please upload and index at least one document first."
            )
            st.stop()

        if not chat_model:
            st.error(
                "Please select or enter a valid OpenRouter chat model."
            )
            st.stop()

        if (
            st.session_state.embedding_model_used
            and st.session_state.embedding_model_used != embedding_model
        ):
            st.warning(
                f"Documents were indexed with "
                f"`{st.session_state.embedding_model_used}`. "
                f"Re-index your files if you want to query "
                f"using `{embedding_model}`."
            )

        st.session_state.messages.append(
            {"role": "user", "content": prompt_text}
        )

        with st.chat_message("user"):
            st.markdown(prompt_text)

        history_messages = to_langchain_messages(
            st.session_state.messages[:-1]
        )

        with st.chat_message("assistant"):
            try:
                llm = build_chat_model(
                    api_key=api_key,
                    model_name=chat_model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )

                st.write("Step 1: Starting document search...")

                search_query = prompt_text

                st.write("Step 2: Creating query embedding...")

                docs_with_scores = (
                    st.session_state.vector_store.similarity_search_with_score(
                        query=search_query,
                        k=int(top_k),
                    )
                )

                st.write("Step 3: Documents retrieved!")

                context_text = format_docs_for_prompt(
                    docs_with_scores
                )

                st.write("Step 4: Context prepared!")

                stream = build_rag_answer_stream(
                    llm=llm,
                    context_text=context_text,
                    chat_history=history_messages,
                    question=prompt_text,
                )

                st.write("Step 5: Generating answer...")

                answer = st.write_stream(stream)

                st.write("Step 6: Answer generated!")

                source_items = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": (
                            doc.metadata.get("page")
                            if isinstance(doc.metadata.get("page"), int)
                            else None
                        ),
                        "score": float(score),
                        "snippet": (
                            doc.page_content[:300] + "..."
                            if len(doc.page_content) > 300
                            else doc.page_content
                        ),
                    }
                    for doc, score in docs_with_scores
                ]

                render_sources(source_items)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_items,
                    }
                )

            except Exception as exc:
                st.error(f"Error generating response: {exc}")


if __name__ == "__main__":
    main()
