# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 11 — FASTAPI INTEGRATION
# ============================================================

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.advanced_rag import AdvancedRAG


# ============================================================
# 1 — APPLICATION STATE
# ============================================================

rag_system = None


# ============================================================
# 2 — STARTUP AND SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_system

    print("Loading Advanced RAG System...")

    rag_system = AdvancedRAG(
        model_name="qwen2.5:0.5b",
        device="cpu",
    )

    print("Advanced RAG System ready!")

    yield

    rag_system = None

    print("Advanced RAG System stopped.")


# ============================================================
# 3 — CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Advanced RAG System API",
    description=(
        "Hybrid Retrieval, Cross-Encoder Reranking, "
        "Evidence Filtering and Local Qwen Generation"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 4 — REQUEST AND RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class AskResponse(BaseModel):

    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieval: dict[str, Any]


# ============================================================
# 5 — ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Advanced RAG System",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ============================================================
# 6 — HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    return {
        "status": "healthy",
        "rag_loaded": True,
        "indexed_chunks": rag_system.chunk_count,
    }


# ============================================================
# 7 — ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):

    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    try:

        result = rag_system.ask(
            question=request.question,
            top_k=request.top_k,
        )

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            retrieval={
                "candidate_count": result["retrieval"][
                    "candidate_count"
                ],
                "original_chunk_count": result["retrieval"][
                    "original_chunk_count"
                ],
                "filtered_chunk_count": result["retrieval"][
                    "filtered_chunk_count"
                ],
            },
        )

    except Exception as error:

        print("RAG error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from error