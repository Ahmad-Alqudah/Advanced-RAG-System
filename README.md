# Advanced RAG System

An end-to-end Retrieval-Augmented Generation (RAG) application featuring hybrid search, cross-encoder reranking, evidence filtering, and local LLM inference.

## Features

- TXT document ingestion and chunking
- BM25 keyword retrieval
- FAISS semantic vector search
- Hybrid retrieval with Reciprocal Rank Fusion
- Cross-Encoder reranking
- Rule-based query transformation
- Relative-score evidence filtering
- Context-grounded answer generation
- Abstention for unsupported questions
- FastAPI backend
- Streamlit frontend
- Docker Compose deployment

## Architecture

User Question
    |
    v
Query Transformation
    |
    v
BM25 + FAISS
    |
    v
Hybrid Retrieval (RRF)
    |
    v
Cross-Encoder Reranking
    |
    v
Evidence Filtering
    |
    v
Context Construction
    |
    v
Ollama / Qwen2.5-0.5B
    |
    v
Answer + Retrieved Sources

## Technology Stack

- Python 3.11
- FastAPI
- Streamlit
- FAISS
- Rank-BM25
- Sentence Transformers
- Cross-Encoder
- Ollama / Qwen2.5-0.5B
- Pytest
- Docker / Docker Compose

## Evaluation Results

### Retrieval Evaluation

| Method | Hit@1 | Hit@3 | MRR@3 |
|---|---:|---:|---:|
| BM25 | 80% | 90% | 0.85 |
| FAISS | 90% | 90% | 0.90 |
| Hybrid | 90% | 90% | 0.90 |
| Hybrid + Reranking | 100% | 100% | 1.00 |

Evaluation dataset: 10 internal questions.

### Answer Evaluation

- 6/6 overall checks passed
- 4/4 answerable-question fact coverage checks passed
- 2/2 unanswerable-question abstention checks passed

Evaluation uses small internal test cases and lexical checks.
Results do not establish performance on unseen datasets.

### Automated API Testing

10/10 FastAPI tests passed using mocked RAG responses.

### Docker Verification

- FastAPI health check passed
- Streamlit connected to FastAPI
- Grounded SEC-204 question answered correctly
- Unsupported CEO question returned abstention
- Local Ollama inference worked through Docker

## Running with Docker

### Requirements

- Docker Desktop
- Ollama running on the host
- Ollama model: qwen2.5:0.5b

Pull the model:

    ollama pull qwen2.5:0.5b

Start the application:

    docker compose up --build -d

Open:

- Streamlit: http://localhost:8502
- FastAPI Swagger: http://localhost:8002/docs
- Health Check: http://localhost:8002/health

Stop:

    docker compose down

The Docker configuration connects to host Ollama using
host.docker.internal:11434.

## Known Limitations

- Evaluated on a small synthetic document collection
- Supports TXT ingestion in this project version
- Local Qwen2.5-0.5B may produce inaccurate answers
- Evidence filtering uses relative reranking scores
- Retrieved sources are not claim-level citation verification
- No authentication or rate limiting
- Docker deployment verified locally, not in cloud production

## Project Status

Completed locally with FastAPI, Streamlit, Docker,
retrieval evaluation, answer evaluation, and automated API tests.