# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 06 — QUERY TRANSFORMATION
# FILE: app/retrieval/query_transformer.py
# ============================================================

import re


class QueryTransformer:

    def __init__(self):

        # Domain-specific phrase alternatives.
        # These are examples, not a universal synonym dictionary.
        self.expansions = {
            "work from home": "remote work",
            "working from home": "remote work",
            "login problem": "account access issue",
            "sign in problem": "account access issue",
            "deploy application": "production deployment",
            "critical ticket": "high priority incident",
        }

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize(self, query: str) -> str:

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        # Remove extra whitespace without changing identifiers.
        normalized = " ".join(query.strip().split())

        if not normalized:
            raise ValueError("Query cannot be empty.")

        return normalized

    # ========================================================
    # QUERY EXPANSION
    # ========================================================

    def expand(self, query: str) -> str:

        normalized = self.normalize(query)

        expanded_terms = []

        for phrase, alternative in self.expansions.items():

            if phrase in normalized.lower():
                expanded_terms.append(alternative)

        if not expanded_terms:
            return normalized

        # Keep original query intact and append alternatives.
        return normalized + " " + " ".join(expanded_terms)

    # ========================================================
    # IDENTIFIER EXTRACTION
    # ========================================================

    def extract_identifiers(self, query: str) -> list[str]:

        normalized = self.normalize(query)

        # Detect codes such as SEC-204, ATL-908, P1, P2.
        pattern = r"\b(?:[A-Za-z]+-\d+|P\d+)\b"

        identifiers = re.findall(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        return list(dict.fromkeys(
            identifier.upper()
            for identifier in identifiers
        ))

    # ========================================================
    # MULTI-QUERY GENERATION
    # ========================================================

    def transform(self, query: str) -> dict:

        original = self.normalize(query)

        expanded = self.expand(original)

        identifiers = self.extract_identifiers(original)

        # Keep unique query variants in their original order.
        queries = list(dict.fromkeys([
            original,
            expanded,
        ]))

        return {
            "original_query": original,
            "expanded_query": expanded,
            "identifiers": identifiers,
            "queries": queries,
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    transformer = QueryTransformer()

    questions = [
        "Can employees work from home?",
        "What does SEC-204 mean?",
        "What does P1 indicate?",
        "Who manages Project Atlas?",
        "How do we deploy application?",
        "How can I report a login problem?",
    ]

    for question in questions:

        result = transformer.transform(question)

        print("\n" + "=" * 65)
        print("Original:", result["original_query"])
        print("Expanded:", result["expanded_query"])
        print("Identifiers:", result["identifiers"])
        print("Search Queries:", result["queries"])