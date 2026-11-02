# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 08 — RETRIEVAL EVALUATION
# FILE: evaluation/retrieval_evaluation.py
# ============================================================

from pathlib import Path

from app.ingestion.document_loader import load_documents
from app.ingestion.chunking import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.faiss_retriever import FAISSRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# EVALUATION DATASET
# ============================================================

TEST_CASES = [
    {
        "question": "What does SEC-204 mean?",
        "source": "security_policy.txt",
        "expected_text": "SEC-204 identifies unauthorized account access",
    },
    {
        "question": "Who manages Project Atlas?",
        "source": "project_atlas.txt",
        "expected_text": "The project manager is David Wilson",
    },
    {
        "question": "What does P1 indicate?",
        "source": "technical_support.txt",
        "expected_text": "P1 indicates a production outage",
    },
    {
        "question": "Can employees work from home?",
        "source": "employee_policy.txt",
        "expected_text": "Remote work is permitted up to two days per week",
    },
    {
        "question": "What does P2 indicate?",
        "source": "technical_support.txt",
        "expected_text": "P2 indicates major functionality degradation",
    },
    {
        "question": "Who manages Project Orion?",
        "source": "project_orion.txt",
        "expected_text": "The project manager is Sarah Johnson",
    },
    {
        "question": "What is the code for Project Atlas?",
        "source": "project_atlas.txt",
        "expected_text": "ATL-908 belongs to Project Atlas",
    },
    {
        "question": "What technology is used for deployments?",
        "source": "deployment_guide.txt",
        "expected_text": "deployed using Docker containers",
    },
    {
        "question": "How many days of annual leave do employees receive?",
        "source": "employee_policy.txt",
        "expected_text": "21 days of paid annual leave",
    },
    {
        "question": "When is Project Orion planned to finish?",
        "source": "project_orion.txt",
        "expected_text": "February 20, 2027",
    },
]


# ============================================================
# RELEVANCE CHECK
# ============================================================

def is_relevant(result: dict, test_case: dict) -> bool:

    same_source = (
        result["source"] == test_case["source"]
    )

    contains_answer = (
        test_case["expected_text"].lower()
        in result["text"].lower()
    )

    return same_source and contains_answer


# ============================================================
# METRIC CALCULATION
# ============================================================

def evaluate_results(
    results: list[dict],
    test_case: dict,
) -> dict:

    first_relevant_rank = None

    for rank, result in enumerate(results, start=1):

        if is_relevant(result, test_case):
            first_relevant_rank = rank
            break

    hit_at_1 = int(first_relevant_rank == 1)

    hit_at_3 = int(
        first_relevant_rank is not None
        and first_relevant_rank <= 3
    )

    reciprocal_rank = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    return {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": first_relevant_rank,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    documents = load_documents(
        str(PROJECT_ROOT / "data" / "documents")
    )

    chunks = create_chunks(documents)

    bm25 = BM25Retriever(chunks)

    faiss = FAISSRetriever(
        chunks=chunks,
        device="cpu",
    )

    hybrid = HybridRetriever(
        chunks=chunks,
        rrf_k=60,
        device="cpu",
    )

    reranker = CrossEncoderReranker(
        device="cpu",
    )

    methods = [
        "BM25",
        "FAISS",
        "HYBRID",
        "HYBRID_RERANK",
    ]

    totals = {
        method: {
            "hit_at_1": 0,
            "hit_at_3": 0,
            "reciprocal_rank": 0.0,
        }
        for method in methods
    }

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]

        bm25_results = bm25.search(
            query=question,
            top_k=3,
        )

        faiss_results = faiss.search(
            query=question,
            top_k=3,
        )

        hybrid_results = hybrid.search(
            query=question,
            top_k=3,
            candidate_k=8,
        )

        # Use a larger pool before reranking.
        rerank_candidates = hybrid.search(
            query=question,
            top_k=8,
            candidate_k=8,
        )

        reranked_results = reranker.rerank(
            query=question,
            candidates=rerank_candidates,
            top_k=3,
        )

        all_results = {
            "BM25": bm25_results,
            "FAISS": faiss_results,
            "HYBRID": hybrid_results,
            "HYBRID_RERANK": reranked_results,
        }

        print("\n" + "=" * 65)
        print(f"TEST {index}: {question}")

        for method, results in all_results.items():

            metrics = evaluate_results(
                results=results,
                test_case=test_case,
            )

            for key in totals[method]:
                totals[method][key] += metrics[key]

            rank = metrics["first_relevant_rank"]

            print(
                f"{method:15} "
                f"Rank={str(rank):5} "
                f"Hit@1={metrics['hit_at_1']} "
                f"Hit@3={metrics['hit_at_3']} "
                f"RR={metrics['reciprocal_rank']:.3f}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    count = len(TEST_CASES)

    print("\n" + "=" * 75)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 75)

    print(
        f"{'METHOD':18}"
        f"{'HIT@1':>12}"
        f"{'HIT@3':>12}"
        f"{'MRR@3':>12}"
    )

    print("-" * 75)

    for method in methods:

        scores = totals[method]

        hit1 = scores["hit_at_1"] / count
        hit3 = scores["hit_at_3"] / count
        mrr = scores["reciprocal_rank"] / count

        print(
            f"{method:18}"
            f"{hit1:>12.2%}"
            f"{hit3:>12.2%}"
            f"{mrr:>12.4f}"
        )

    print("\nTotal questions:", count)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()