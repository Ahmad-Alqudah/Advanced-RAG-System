# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# ADVANCED RAG + EVIDENCE FILTER + ABSTENTION HANDLING
# FILE: app/rag/advanced_rag.py
# ============================================================

from pathlib import Path
import os
import requests

from app.ingestion.document_loader import load_documents
from app.ingestion.chunking import create_chunks
from app.retrieval.query_transformer import QueryTransformer
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.rag.evidence_filter import EvidenceFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ABSTENTION = (
    "I cannot find the answer in the provided documents."
)


class AdvancedRAG:

    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434"),
        device: str = "cpu",
    ):

        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")

        documents_path = PROJECT_ROOT / "data" / "documents"

        documents = load_documents(str(documents_path))
        chunks = create_chunks(documents)

        if not chunks:
            raise ValueError("No document chunks available.")

        self.transformer = QueryTransformer()

        self.hybrid = HybridRetriever(
            chunks=chunks,
            rrf_k=60,
            device=device,
        )

        self.reranker = CrossEncoderReranker(
            device=device,
        )

        self.evidence_filter = EvidenceFilter(
            max_score_gap=4.0,
            min_chunks=1,
            max_chunks=3,
        )

        self.chunk_count = len(chunks)

    # ========================================================
    # MULTI-QUERY RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        candidate_k: int = 8,
    ) -> dict:

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError(
                "top_k and candidate_k must be positive."
            )

        transformed = self.transformer.transform(question)

        unique_candidates = {}

        for query in transformed["queries"]:

            results = self.hybrid.search(
                query=query,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )

            for result in results:

                chunk_id = result["chunk_id"]

                if chunk_id not in unique_candidates:

                    candidate = result.copy()
                    candidate["matched_queries"] = [query]

                    unique_candidates[chunk_id] = candidate

                else:

                    unique_candidates[chunk_id][
                        "matched_queries"
                    ].append(query)

        candidates = list(unique_candidates.values())

        reranked = self.reranker.rerank(
            query=transformed["original_query"],
            candidates=candidates,
            top_k=top_k,
        )

        return {
            "query_transformation": transformed,
            "candidate_count": len(candidates),
            "chunks": reranked,
        }

    # ========================================================
    # EVIDENCE CONTEXT
    # ========================================================

    def build_context(
        self,
        chunks: list[dict],
    ) -> str:

        context_parts = []

        for index, chunk in enumerate(chunks, start=1):

            context_parts.append(
                f"[Source {index}]\n"
                f"File: {chunk['source']}\n"
                f"Content:\n{chunk['text']}"
            )

        return "\n\n".join(context_parts)

    # ========================================================
    # ABSTENTION NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_abstention(answer: str) -> str:

        cleaned = answer.strip()

        if cleaned.lower().startswith(
            ABSTENTION.lower()
        ):
            return ABSTENTION

        return cleaned

    # ========================================================
    # LOCAL LLM GENERATION
    # ========================================================

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        prompt = f"""You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.
Do not invent facts or use outside knowledge.

If the context does not contain the answer, respond exactly:
{ABSTENTION}

Do not add explanations after the abstention sentence.

If the answer is available:
- Provide the complete relevant facts.
- Keep the answer concise.
- Cite supporting evidence using [Source 1], [Source 2], etc.
- Do not cite unrelated sources.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                },
            },
            timeout=180,
        )

        response.raise_for_status()

        raw_answer = response.json()["response"]

        return self.normalize_abstention(raw_answer)

    # ========================================================
    # COMPLETE RAG PIPELINE
    # ========================================================

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict:

        retrieval = self.retrieve(
            question=question,
            top_k=top_k,
        )

        original_chunks = retrieval["chunks"]

        filtered_chunks = self.evidence_filter.filter(
            original_chunks
        )

        retrieval["filtered_chunks"] = filtered_chunks
        retrieval["original_chunk_count"] = len(original_chunks)
        retrieval["filtered_chunk_count"] = len(filtered_chunks)

        if not filtered_chunks:

            return {
                "question": question,
                "answer": ABSTENTION,
                "sources": [],
                "retrieval": retrieval,
            }

        context = self.build_context(
            filtered_chunks
        )

        answer = self.generate(
            question=question,
            context=context,
        )

        # Never attach sources to an abstention.
        if answer == ABSTENTION:

            sources = []

        else:

            sources = [
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "rerank_score": chunk["rerank_score"],
                    "matched_queries": chunk["matched_queries"],
                }
                for chunk in filtered_chunks
            ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval": retrieval,
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    rag = AdvancedRAG(
        model_name="qwen2.5:0.5b",
        device="cpu",
    )

    print(f"Indexed chunks: {rag.chunk_count}")

    questions = [
        "What does SEC-204 mean?",
        "Who manages Project Atlas?",
        "What does P1 indicate?",
        "Can employees work from home?",
        "Who is the CEO of NovaTech?",
        "What is NovaTech's stock price?",
    ]

    for question in questions:

        print("\n" + "=" * 65)
        print("QUESTION:", question)

        try:

            result = rag.ask(
                question=question,
                top_k=3,
            )

            print("\nANSWER:")
            print(result["answer"])

            print("\nEVIDENCE FILTERING:")
            print(
                "Before:",
                result["retrieval"]["original_chunk_count"],
            )
            print(
                "After:",
                result["retrieval"]["filtered_chunk_count"],
            )

            print("\nSOURCES:")

            if not result["sources"]:
                print("No supporting sources returned.")

            for source in result["sources"]:
                print(
                    f"- {source['source']} "
                    f"(score={source['rerank_score']:.4f})"
                )

        except requests.RequestException as error:

            print(
                "Ollama connection/generation error:",
                error,
            )
            break