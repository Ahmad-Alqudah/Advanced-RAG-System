# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 05 — CROSS-ENCODER RERANKING
# FILE: app/retrieval/reranker.py
# ============================================================

from pathlib import Path

from sentence_transformers import CrossEncoder

from app.ingestion.document_loader import load_documents
from app.ingestion.chunking import create_chunks
from app.retrieval.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not candidates:
            return []

        # Build question-document pairs
        pairs = [
            (
                query,
                candidate["title"] + "\n" + candidate["text"],
            )
            for candidate in candidates
        ]

        # Score every pair using the Cross-Encoder
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        # Preserve original metadata and add reranking scores
        reranked = []

        for candidate, score in zip(candidates, scores):

            result = candidate.copy()
            result["rerank_score"] = float(score)

            reranked.append(result)

        # Sort by Cross-Encoder relevance
        reranked.sort(
            key=lambda item: (
                -item["rerank_score"],
                item["chunk_id"],
            )
        )

        return reranked[:top_k]


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    documents_path = PROJECT_ROOT / "data" / "documents"

    documents = load_documents(
        str(documents_path)
    )

    chunks = create_chunks(documents)

    hybrid = HybridRetriever(
        chunks=chunks,
        rrf_k=60,
        device="cpu",
    )

    reranker = CrossEncoderReranker(
        device="cpu",
    )

    questions = [
        "What does SEC-204 mean?",
        "Who manages Project Atlas?",
        "What does P1 indicate?",
        "Can employees work from home?",
    ]

    print(f"Indexed chunks: {len(chunks)}")

    for question in questions:

        print("\n" + "=" * 65)
        print("QUESTION:", question)

        # Retrieve a larger candidate pool before reranking
        candidates = hybrid.search(
            query=question,
            top_k=8,
            candidate_k=8,
        )

        results = reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=3,
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Rerank Score: {result['rerank_score']:.4f}")
            print(f"RRF Score: {result['rrf_score']:.6f}")
            print(f"Source: {result['source']}")
            print(f"Original ranks: {result['ranks']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text']}")