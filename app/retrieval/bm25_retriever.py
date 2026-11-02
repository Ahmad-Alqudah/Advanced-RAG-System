
# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# FILE: app/retrieval/bm25_retriever.py
# PURPOSE: BM25 Keyword Retrieval
# ============================================================

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.document_loader import load_documents
from app.ingestion.chunking import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into searchable tokens.
    Preserve identifiers such as SEC-204 and ATL-908.
    """

    return re.findall(
        r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*",
        text.lower()
    )


# ============================================================
# BM25 RETRIEVER
# ============================================================

class BM25Retriever:

    def __init__(self, chunks: list[dict]):

        if not chunks:
            raise ValueError("Chunks list cannot be empty.")

        self.chunks = chunks

        # Prepare searchable tokens from titles and content
        self.tokenized_corpus = [
            tokenize(chunk["title"] + " " + chunk["text"])
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

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

        # Tokenize user query
        query_tokens = tokenize(query)

        # Calculate BM25 scores for all chunks
        scores = self.bm25.get_scores(
            query_tokens
        )

        # ====================================================
        # IMPROVEMENT:
        # Exclude chunks with zero or negative BM25 scores
        # before ranking and returning candidates.
        # ====================================================

        ranked_indices = sorted(
            (
                i for i in range(len(scores))
                if scores[i] > 0
            ),
            key=lambda i: (-scores[i], i)
        )

        # Prepare search results
        results = []

        for index in ranked_indices[:top_k]:

            chunk = self.chunks[index]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(scores[index])
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

    retriever = BM25Retriever(chunks)

    questions = [
        "What does SEC-204 mean?",
        "Who manages Project Atlas?",
        "What does P1 indicate?",
        "Can employees work from home?"
    ]

    print(f"Indexed chunks: {len(chunks)}")

    for question in questions:

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        results = retriever.search(
            query=question,
            top_k=3
        )

        if not results:
            print("No positive BM25 matches found.")
            continue

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(f"\nRank: {rank}")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text']}")
