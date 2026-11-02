# Project 02 — Advanced RAG System
## Hybrid Search, Reranking & Retrieval Evaluation

**Project Type:** AI Engineering / LLM Engineering
**Status:** In Development
**Language:** Arabic & English
**Deployment:** Local LLM + FastAPI + Streamlit + Docker (Planned)

---

# القسم الأول — الشرح بالعربي

## 1. مقدمة المشروع

يهدف هذا المشروع إلى بناء نظام متقدم لاسترجاع المعلومات وتوليد الإجابات باستخدام تقنية Retrieval-Augmented Generation (RAG).

يسمح النظام للمستخدم بطرح أسئلة على مستندات الشركات، ويقوم بالبحث عن المعلومات الأكثر ارتباطًا بالسؤال، ثم يستخدم نموذج لغة محليًا لتوليد إجابة مستندة إلى المصادر.

يركز المشروع على تحسين جودة الاسترجاع باستخدام عدة تقنيات بدل الاعتماد على البحث الدلالي وحده.

## 2. المشكلة التي يحلها المشروع

تعتمد أنظمة RAG التقليدية أحيانًا على البحث الدلالي فقط، وهذا قد يسبب صعوبة في استرجاع المعلومات التي تحتوي على:

- رموز دقيقة مثل SEC-204.
- أسماء المشاريع والأشخاص.
- أرقام التذاكر والمنتجات.
- مصطلحات تقنية محددة.
- أسئلة تستخدم صياغة مختلفة عن نص المستند.

لذلك سنجمع بين البحث بالكلمات والبحث الدلالي، ثم نعيد ترتيب النتائج لتحسين اختيار المعلومات.

## 3. أهداف المشروع

1. إنشاء نظام Document Ingestion وChunking.
2. تنفيذ BM25 Keyword Search.
3. تنفيذ FAISS Semantic Search.
4. دمج نتائج البحث باستخدام Reciprocal Rank Fusion (RRF).
5. إعادة ترتيب النتائج باستخدام Cross-Encoder Reranking.
6. تجربة Query Transformation لتحسين الاسترجاع.
7. ربط نظام الاسترجاع بنموذج لغة محلي عبر Ollama.
8. تقييم جودة الاسترجاع باستخدام Recall@K وMRR وnDCG.
9. بناء REST API باستخدام FastAPI.
10. بناء واجهة مستخدم باستخدام Streamlit.
11. تجهيز النظام للتشغيل باستخدام Docker.

## 4. كيف يعمل النظام؟

```text
User Question
     |
     v
Query Transformation
     |
     +-----------------------+
     |                       |
     v                       v
BM25 Keyword Search     FAISS Semantic Search
     |                       |
     +-----------+-----------+
                 |
                 v
       Reciprocal Rank Fusion
                 |
                 v
       Cross-Encoder Reranking
                 |
                 v
          Relevant Context
                 |
                 v
          Local LLM (Ollama)
                 |
                 v
       Final Answer + Sources
```

## 5. التقنيات المستخدمة والمخططة

| التقنية | الوظيفة |
|---|---|
| Python | لغة البرمجة الأساسية |
| LangChain Text Splitters | تقسيم المستندات |
| BM25 | البحث بالكلمات المفتاحية |
| FAISS | البحث الدلالي باستخدام Embeddings |
| Sentence Transformers | إنشاء Embeddings وإعادة ترتيب النتائج |
| RRF | دمج ترتيب نتائج البحث |
| Cross-Encoder | تقييم علاقة السؤال بالمقاطع |
| Ollama / Qwen | توليد الإجابات محليًا |
| FastAPI | بناء Backend API |
| Streamlit | بناء واجهة المستخدم |
| Docker | تشغيل خدمات التطبيق |
| Retrieval Evaluation | قياس جودة البحث |

## 6. الفرق عن المشروع الأول

**Project 01 — AI Document Intelligence**

ركز على بناء نظام RAG متكامل، يشمل رفع المستندات والبحث الدلالي وتوليد الإجابات والواجهة وAPI وDocker.

**Project 02 — Advanced RAG System**

مشروع مستقل يركز على تحسين جودة الاسترجاع من خلال Hybrid Search وReranking وQuery Transformation، مع مقارنة طرق البحث باستخدام مقاييس تقييم واضحة.

## 7. البيانات التجريبية

نستخدم حاليًا ستة مستندات لشركة افتراضية NovaTech، تشمل:

- Employee Policy
- Security Policy
- Project Atlas
- Project Orion
- Technical Support
- Deployment Guide

هذه البيانات مخصصة للتطوير الأولي، وليست كافية وحدها لإثبات تفوق طريقة استرجاع على أخرى.

## 8. النتيجة النهائية المتوقعة

تطبيق يستطيع قراءة مستندات الشركات والبحث داخلها والإجابة عن الأسئلة مع إظهار المصادر.

سيتيح المشروع مقارنة طرق الاسترجاع وقياس نتائجها، ثم عرضها في واجهة عملية قابلة للتشغيل محليًا.

---

# Part Two — English Project Overview

## 1. Project Introduction

Advanced RAG System is an independent AI engineering project focused on improving document retrieval quality in Retrieval-Augmented Generation applications.

The system retrieves relevant information from enterprise documents and uses a locally hosted language model to generate grounded answers with source references.

Unlike a basic semantic-search RAG pipeline, this project combines keyword retrieval, dense retrieval, result fusion, and reranking.

## 2. Problem Statement

Semantic retrieval alone may struggle with exact identifiers, technical codes, project names, and other keyword-sensitive information.

Keyword retrieval can identify exact terms but may fail when user questions express the same meaning using different vocabulary.

This project investigates whether combining both approaches improves retrieval effectiveness.

## 3. Project Objectives

1. Implement document ingestion and chunking.
2. Build a BM25 keyword retriever.
3. Build a FAISS semantic retriever.
4. Combine ranked results using Reciprocal Rank Fusion.
5. Apply Cross-Encoder reranking.
6. Explore query rewriting and expansion.
7. Integrate a local LLM through Ollama.
8. Evaluate retrieval with Recall@K, MRR, and nDCG.
9. Expose the system through FastAPI.
10. Build a Streamlit user interface.
11. Containerize the application with Docker.

## 4. System Architecture

```text
User Query
   |
Query Transformation
   |
   +--> BM25 Retriever
   |
   +--> FAISS Retriever
   |
   v
Reciprocal Rank Fusion
   |
Cross-Encoder Reranker
   |
Relevant Document Chunks
   |
Local LLM via Ollama
   |
Grounded Answer + Source References
```

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| LangChain Text Splitters | Document chunking |
| BM25 | Sparse keyword retrieval |
| FAISS | Dense vector retrieval |
| Sentence Transformers | Embeddings and reranking |
| Reciprocal Rank Fusion | Retrieval result fusion |
| Cross-Encoder | Relevance reranking |
| Ollama / Qwen | Local answer generation |
| FastAPI | REST API |
| Streamlit | Web interface |
| Docker | Containerized deployment |
| Recall@K / MRR / nDCG | Retrieval evaluation |

## 6. Relationship to Project 01

Project 01 — AI Document Intelligence established a complete document-question-answering application using semantic retrieval.

Project 02 is a separate application that extends the engineering concepts by introducing hybrid retrieval, reranking, and comparative evaluation.

The projects are independently organized and can be documented and presented separately in a professional portfolio.

## 7. Current Progress

- [x] Project directory structure
- [x] Initial six-document dataset
- [x] TXT document loader
- [x] Document chunking with stable chunk IDs
- [ ] BM25 keyword retrieval
- [ ] FAISS semantic retrieval
- [ ] Hybrid retrieval and RRF
- [ ] Cross-Encoder reranking
- [ ] Query transformation
- [ ] Advanced RAG pipeline
- [ ] Retrieval evaluation
- [ ] FastAPI and Streamlit
- [ ] Docker deployment
- [ ] Final documentation and review

## 8. Expected Deliverables

- Independent Advanced RAG application.
- Reusable retrieval pipeline components.
- Source-grounded question answering.
- Comparative retrieval evaluation.
- FastAPI backend and Streamlit frontend.
- Docker deployment configuration.
- Portfolio-ready documentation.

## 9. Evaluation Note

Retrieval improvements must be demonstrated experimentally rather than assumed.

The initial NovaTech documents are a small synthetic development dataset. More representative documents and labeled queries will be required for meaningful evaluation.

---

**Project Status:** In Development
**Project Category:** Advanced RAG / LLM Engineering
