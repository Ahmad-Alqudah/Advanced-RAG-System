# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 09 — ANSWER QUALITY EVALUATION
# FILE: evaluation/answer_evaluation.py
# ============================================================

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.rag.advanced_rag import AdvancedRAG


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "answer_results.json"
)

ABSTENTION = (
    "I cannot find the answer in the provided documents."
)


# ============================================================
# SMALL REFERENCE DATASET
# ============================================================

TEST_CASES = [
    {
        "id": "Q01",
        "question": "What does SEC-204 mean?",
        "answerable": True,
        "expected_facts": [
            ["SEC-204"],
            ["unauthorized", "account", "access"],
        ],
        "expected_source": "security_policy.txt",
    },
    {
        "id": "Q02",
        "question": "Who manages Project Atlas?",
        "answerable": True,
        "expected_facts": [
            ["David Wilson"],
        ],
        "expected_source": "project_atlas.txt",
    },
    {
        "id": "Q03",
        "question": "What does P1 indicate?",
        "answerable": True,
        "expected_facts": [
            ["P1"],
            ["production", "outage"],
        ],
        "expected_source": "technical_support.txt",
    },
    {
        "id": "Q04",
        "question": "Can employees work from home?",
        "answerable": True,
        "expected_facts": [
            ["remote work", "work from home"],
            ["two days", "2 days"],
            ["manager approval", "manager's approval"],
        ],
        "expected_source": "employee_policy.txt",
    },
    {
        "id": "Q05",
        "question": "Who is the CEO of NovaTech?",
        "answerable": False,
        "expected_facts": [],
        "expected_source": None,
    },
    {
        "id": "Q06",
        "question": "What is NovaTech's stock price?",
        "answerable": False,
        "expected_facts": [],
        "expected_source": None,
    },
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text: str) -> str:

    return " ".join(
        re.findall(r"[a-z0-9]+", text.lower())
    )


# ============================================================
# REFERENCE FACT COVERAGE
# ============================================================

def fact_coverage(
    answer: str,
    expected_facts: list[list[str]],
) -> float:

    if not expected_facts:
        return 0.0

    normalized_answer = normalize(answer)

    matched = 0

    for alternatives in expected_facts:

        if any(
            normalize(phrase) in normalized_answer
            for phrase in alternatives
        ):
            matched += 1

    return matched / len(expected_facts)


# ============================================================
# ANSWER EVALUATION
# ============================================================

def evaluate_answer(
    test_case: dict,
    rag_result: dict,
) -> dict:

    answer = rag_result["answer"]
    sources = rag_result["sources"]

    is_abstention = normalize(answer) == normalize(
        ABSTENTION
    )

    if test_case["answerable"]:

        coverage = fact_coverage(
            answer,
            test_case["expected_facts"],
        )

        correct_behavior = (
            coverage == 1.0
            and not is_abstention
        )

        source_found = any(
            source["source"] == test_case["expected_source"]
            for source in sources
        )

    else:

        coverage = None
        correct_behavior = is_abstention
        source_found = None

    # Inspect the retrieved context for human grounding review.
    retrieved_chunks = rag_result["retrieval"]["chunks"]

    evidence = [
        {
            "source": chunk["source"],
            "text": chunk["text"],
            "rerank_score": chunk["rerank_score"],
        }
        for chunk in retrieved_chunks
    ]

    return {
        "id": test_case["id"],
        "question": test_case["question"],
        "answerable": test_case["answerable"],
        "answer": answer,
        "expected_facts": test_case["expected_facts"],
        "fact_coverage": coverage,
        "correct_behavior": correct_behavior,
        "expected_source_found": source_found,
        "abstained": is_abstention,
        "evidence": evidence,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    rag = AdvancedRAG(
        model_name="qwen2.5:0.5b",
        device="cpu",
    )

    results = []

    print(f"Indexed chunks: {rag.chunk_count}")

    for test_case in TEST_CASES:

        print("\n" + "=" * 70)
        print("QUESTION:", test_case["question"])

        rag_result = rag.ask(
            question=test_case["question"],
            top_k=3,
        )

        evaluation = evaluate_answer(
            test_case,
            rag_result,
        )

        results.append(evaluation)

        print("ANSWER:", evaluation["answer"])

        print(
            "EXPECTED:",
            "Answer available"
            if test_case["answerable"]
            else "Must abstain",
        )

        print(
            "FACT COVERAGE:",
            evaluation["fact_coverage"],
        )

        print(
            "CORRECT BEHAVIOR:",
            evaluation["correct_behavior"],
        )

        print(
            "EXPECTED SOURCE FOUND:",
            evaluation["expected_source_found"],
        )

        print("RETRIEVED EVIDENCE:")

        for item in evaluation["evidence"]:

            print(
                f"  {item['source']} "
                f"(score={item['rerank_score']:.4f})"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    answerable = [
        result for result in results
        if result["answerable"]
    ]

    unanswerable = [
        result for result in results
        if not result["answerable"]
    ]

    answered_correctly = sum(
        result["correct_behavior"]
        for result in answerable
    )

    abstained_correctly = sum(
        result["correct_behavior"]
        for result in unanswerable
    )

    total_correct = sum(
        result["correct_behavior"]
        for result in results
    )

    summary = {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "fact_check_pass_rate": (
            answered_correctly / len(answerable)
            if answerable else 0.0
        ),
        "abstention_accuracy": (
            abstained_correctly / len(unanswerable)
            if unanswerable else 0.0
        ),
        "overall_behavior_pass_rate": (
            total_correct / len(results)
            if results else 0.0
        ),
    }

    print("\n" + "=" * 70)
    print("FINAL ANSWER QUALITY EVALUATION")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nReport saved to:", REPORT_PATH)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()