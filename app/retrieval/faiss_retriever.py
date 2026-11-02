
# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# FILE: app/retrieval/faiss_retriever.py
# PURPOSE: FAISS Semantic Retrieval
# ============================================================

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.document_loader import load_documents
from app.ingestion.chunking import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# FAISS RETRIEVER
# ============================================================

class FAISSRetriever:

    def __init__(
        self,
        chunks: list[dict],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):

        if not chunks:
            raise ValueError("Chunks list cannot be empty.")

        self.chunks = chunks

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        texts = [
            chunk["title"] + "\n" + chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        query_embedding = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32
        )

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            chunk = self.chunks[int(index)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score)
            })

        return results


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    documents_path = PROJECT_ROOT / "data" / "documents"

    documents = load_documents(
        str(documents_path)
    )

    chunks = create_chunks(documents)

    retriever = FAISSRetriever(
        chunks=chunks,
        device="cpu"
    )

    questions = [
        "What does SEC-204 mean?",
        "Who manages Project Atlas?",
        "What does P1 indicate?",
        "Can employees work from home?"
    ]

    print(f"Indexed chunks: {retriever.index.ntotal}")
    print(f"Embedding dimensions: {retriever.index.d}")

    for question in questions:

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        results = retriever.search(
            query=question,
            top_k=3
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(f"\nRank: {rank}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text']}")
