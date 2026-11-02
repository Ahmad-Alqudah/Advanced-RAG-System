# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 12 — STREAMLIT FRONTEND
# ============================================================

import os
import requests
import streamlit as st


# ============================================================
# 1 — APPLICATION CONFIGURATION
# ============================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# 2 — PAGE HEADER
# ============================================================

st.title("Advanced RAG Assistant")

st.write(
    "Ask questions about company documents using "
    "Hybrid Retrieval, Cross-Encoder Reranking, "
    "Evidence Filtering and Local Qwen."
)

st.divider()


# ============================================================
# 3 — SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("Settings")

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.divider()

    st.subheader("Backend Status")

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        health_response.raise_for_status()

        health_data = health_response.json()

        st.success("FastAPI Connected")

        st.metric(
            "Indexed Chunks",
            health_data.get("indexed_chunks", 0),
        )

    except requests.RequestException:

        st.error("FastAPI is not available.")

        st.caption(
            "Start the FastAPI server on port 8001."
        )


# ============================================================
# 4 — USER QUESTION
# ============================================================

st.subheader("Ask a Question")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What does SEC-204 mean?"
    ),
    height=110,
)

ask_button = st.button(
    "Ask Advanced RAG",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 5 — SEND QUESTION TO FASTAPI
# ============================================================

if ask_button:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question.strip(),
                        "top_k": top_k,
                    },
                    timeout=240,
                )

                response.raise_for_status()

                result = response.json()

            except requests.RequestException as error:

                st.error(
                    "Failed to communicate with FastAPI."
                )

                st.exception(error)

                st.stop()

        # ====================================================
        # 6 — ANSWER
        # ====================================================

        st.divider()

        st.subheader("Generated Answer")

        st.info(
            result.get(
                "answer",
                "No answer returned.",
            )
        )

        # ====================================================
        # 7 — RETRIEVAL METRICS
        # ====================================================

        retrieval = result.get(
            "retrieval",
            {},
        )

        st.subheader("Retrieval Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Candidates",
                retrieval.get(
                    "candidate_count",
                    0,
                ),
            )

        with col2:

            st.metric(
                "Before Filtering",
                retrieval.get(
                    "original_chunk_count",
                    0,
                ),
            )

        with col3:

            st.metric(
                "After Filtering",
                retrieval.get(
                    "filtered_chunk_count",
                    0,
                ),
            )

        # ====================================================
        # 8 — SOURCES
        # ====================================================

        st.subheader("Retrieved Sources")

        sources = result.get(
            "sources",
            [],
        )

        if not sources:

            st.info(
                "No supporting sources were returned."
            )

        else:

            for index, source in enumerate(
                sources,
                start=1,
            ):

                source_name = source.get(
                    "source",
                    "Unknown source",
                )

                with st.expander(
                    f"Source {index}: {source_name}"
                ):

                    st.write(
                        "**File:**",
                        source_name,
                    )

                    st.write(
                        "**Chunk ID:**",
                        source.get(
                            "chunk_id",
                            "N/A",
                        ),
                    )

                    score = source.get(
                        "rerank_score",
                        0.0,
                    )

                    st.write(
                        "**Rerank Score:**",
                        round(score, 4),
                    )

                    st.write(
                        "**Matched Queries:**"
                    )

                    st.write(
                        source.get(
                            "matched_queries",
                            [],
                        )
                    )

        # ====================================================
        # 9 — RAW RESPONSE
        # ====================================================

        with st.expander(
            "View Raw API Response"
        ):

            st.json(result)


# ============================================================
# 10 — FOOTER
# ============================================================

st.divider()

st.caption(
    "Advanced RAG System | "
    "FastAPI + Streamlit + FAISS + BM25 + "
    "Cross-Encoder + Ollama"
)