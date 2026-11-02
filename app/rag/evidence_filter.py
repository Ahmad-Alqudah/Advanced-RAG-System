# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 10 — EVIDENCE FILTERING
# FILE: app/rag/evidence_filter.py
# ============================================================


class EvidenceFilter:

    def __init__(
        self,
        max_score_gap: float = 4.0,
        min_chunks: int = 1,
        max_chunks: int = 3,
    ):

        if max_score_gap < 0:
            raise ValueError("max_score_gap cannot be negative.")

        if min_chunks < 0:
            raise ValueError("min_chunks cannot be negative.")

        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive.")

        if min_chunks > max_chunks:
            raise ValueError(
                "min_chunks cannot exceed max_chunks."
            )

        self.max_score_gap = max_score_gap
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks

    def filter(
        self,
        chunks: list[dict],
    ) -> list[dict]:

        if not chunks:
            return []

        ranked = sorted(
            chunks,
            key=lambda item: -item["rerank_score"],
        )

        best_score = ranked[0]["rerank_score"]

        selected = []

        for chunk in ranked:

            score_gap = best_score - chunk["rerank_score"]

            if score_gap <= self.max_score_gap:
                selected.append(chunk)

        # Keep a minimum number of chunks for context.
        if len(selected) < self.min_chunks:
            selected = ranked[:self.min_chunks]

        return selected[:self.max_chunks]


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    evidence_filter = EvidenceFilter(
        max_score_gap=4.0,
        min_chunks=1,
        max_chunks=3,
    )

    example_chunks = [
        {
            "source": "security_policy.txt",
            "rerank_score": 5.3669,
        },
        {
            "source": "technical_support.txt",
            "rerank_score": -10.3601,
        },
        {
            "source": "project_atlas.txt",
            "rerank_score": -10.9696,
        },
    ]

    filtered = evidence_filter.filter(example_chunks)

    print("Original chunks:", len(example_chunks))
    print("Filtered chunks:", len(filtered))

    for chunk in filtered:
        print(
            chunk["source"],
            chunk["rerank_score"],
        )