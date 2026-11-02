
# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 04 — HYBRID SEARCH + RRF
# FILE: app/retrieval/hybrid_retriever.py
# ============================================================

from pathlib import Path

from app.ingestion.document_loader import load_documents
from app.ingestion.chunking import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.faiss_retriever import FAISSRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        chunks: list[dict],
        rrf_k: int = 60,
        device: str = "cpu",
    ):

        if not chunks:
            raise ValueError("Chunks list cannot be empty.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        self.chunks = chunks
        self.rrf_k = rrf_k

        # Build both retrievers using identical chunks
        self.bm25 = BM25Retriever(chunks)

        self.faiss = FAISSRetriever(
            chunks=chunks,
            device=device,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 6,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError(
                "top_k and candidate_k must be positive."
            )

        # Retrieve candidates from both search methods
        bm25_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
        )

        faiss_results = self.faiss.search(
            query=query,
            top_k=candidate_k,
        )

        # Store fused results by stable chunk ID
        fused = {}

        for method, results in [
            ("bm25", bm25_results),
            ("faiss", faiss_results),
        ]:

            for rank, result in enumerate(
                results,
                start=1,
            ):

                chunk_id = result["chunk_id"]

                if chunk_id not in fused:

                    fused[chunk_id] = {
                        "chunk_id": chunk_id,
                        "document_id": result["document_id"],
                        "title": result["title"],
                        "source": result["source"],
                        "text": result["text"],
                        "rrf_score": 0.0,
                        "retrieved_by": [],
                        "ranks": {},
                    }

                # Add RRF contribution for this retriever
                fused[chunk_id]["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                fused[chunk_id]["retrieved_by"].append(
                    method
                )

                fused[chunk_id]["ranks"][method] = rank

        # Rank by combined RRF score
        ranked_results = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["chunk_id"],
            ),
        )

        return ranked_results[:top_k]


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    documents_path = PROJECT_ROOT / "data" / "documents"

    documents = load_documents(
        str(documents_path)
    )

    chunks = create_chunks(documents)

    retriever = HybridRetriever(
        chunks=chunks,
        rrf_k=60,
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

        results = retriever.search(
            query=question,
            top_k=3,
            candidate_k=6,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(f"\nRank: {rank}")
            print(f"RRF Score: {result['rrf_score']:.6f}")
            print(f"Source: {result['source']}")
            print(f"Retrieved by: {result['retrieved_by']}")
            print(f"Original ranks: {result['ranks']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text']}")
