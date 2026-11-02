# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 13 — FASTAPI AUTOMATED TESTING
# ============================================================

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api.main as api_module


ABSTENTION = (
    "I cannot find the answer in the provided documents."
)


# ============================================================
# 1 — MOCK RAG SYSTEM
# ============================================================

@pytest.fixture
def mock_rag(monkeypatch):

    rag = MagicMock()

    rag.chunk_count = 12

    rag.ask.return_value = {
        "question": "What does SEC-204 mean?",
        "answer": (
            "SEC-204 identifies unauthorized account access."
        ),
        "sources": [
            {
                "source": "security_policy.txt",
                "chunk_id": "test-chunk-001",
                "rerank_score": 5.3669,
                "matched_queries": [
                    "What does SEC-204 mean?"
                ],
            }
        ],
        "retrieval": {
            "candidate_count": 8,
            "original_chunk_count": 3,
            "filtered_chunk_count": 1,
        },
    }

    monkeypatch.setattr(
        api_module,
        "rag_system",
        rag,
    )

    return rag


# ============================================================
# 2 — TEST CLIENT
# ============================================================

@pytest.fixture
def client(mock_rag):

    with TestClient(
        api_module.app,
        raise_server_exceptions=False,
    ) as test_client:

        # Lifespan initializes the real RAG system.
        # Restore the mock after startup.
        api_module.rag_system = mock_rag

        yield test_client


# ============================================================
# TEST 01 — ROOT ENDPOINT
# ============================================================

def test_root_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Advanced RAG System"
    assert data["version"] == "1.0.0"


# ============================================================
# TEST 02 — HEALTH ENDPOINT
# ============================================================

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["rag_loaded"] is True
    assert data["indexed_chunks"] == 12


# ============================================================
# TEST 03 — ASK ENDPOINT
# ============================================================

def test_ask_endpoint(client, mock_rag):

    response = client.post(
        "/ask",
        json={
            "question": "What does SEC-204 mean?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "SEC-204" in data["answer"]

    assert len(data["sources"]) == 1

    assert data["sources"][0]["source"] == (
        "security_policy.txt"
    )

    mock_rag.ask.assert_called_once_with(
        question="What does SEC-204 mean?",
        top_k=3,
    )


# ============================================================
# TEST 04 — EMPTY QUESTION
# ============================================================

def test_empty_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


# ============================================================
# TEST 05 — INVALID TOP-K
# ============================================================

def test_invalid_top_k(client):

    response = client.post(
        "/ask",
        json={
            "question": "Test question",
            "top_k": -1,
        },
    )

    assert response.status_code == 422


# ============================================================
# TEST 06 — MISSING QUESTION
# ============================================================

def test_missing_question(client):

    response = client.post(
        "/ask",
        json={
            "top_k": 3,
        },
    )

    assert response.status_code == 422


# ============================================================
# TEST 07 — RAG NOT READY
# ============================================================

def test_rag_not_ready(client, monkeypatch):

    monkeypatch.setattr(
        api_module,
        "rag_system",
        None,
    )

    response = client.get("/health")

    assert response.status_code == 503


# ============================================================
# TEST 08 — ASK WHEN RAG NOT READY
# ============================================================

def test_ask_when_rag_not_ready(
    client,
    monkeypatch,
):

    monkeypatch.setattr(
        api_module,
        "rag_system",
        None,
    )

    response = client.post(
        "/ask",
        json={
            "question": "What does SEC-204 mean?",
        },
    )

    assert response.status_code == 503


# ============================================================
# TEST 09 — INTERNAL RAG ERROR
# ============================================================

def test_internal_rag_error(client, mock_rag):

    mock_rag.ask.side_effect = RuntimeError(
        "Simulated retrieval failure"
    )

    response = client.post(
        "/ask",
        json={
            "question": "Test question",
        },
    )

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "Failed to process the question."
    )


# ============================================================
# TEST 10 — ABSTENTION RESPONSE
# ============================================================

def test_abstention_response(client, mock_rag):

    mock_rag.ask.return_value = {
        "question": "Who is the CEO of NovaTech?",
        "answer": ABSTENTION,
        "sources": [],
        "retrieval": {
            "candidate_count": 8,
            "original_chunk_count": 3,
            "filtered_chunk_count": 3,
        },
    }

    response = client.post(
        "/ask",
        json={
            "question": "Who is the CEO of NovaTech?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == ABSTENTION
    assert data["sources"] == []