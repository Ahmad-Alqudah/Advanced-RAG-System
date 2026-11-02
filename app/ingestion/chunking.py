
# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# FILE: app/ingestion/chunking.py
# PURPOSE: Create meaningful chunks with stable IDs
# ============================================================

from pathlib import Path
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.document_loader import load_documents


def create_chunks(
    documents: list[dict],
    chunk_size: int = 200,
    chunk_overlap: int = 40,
) -> list[dict]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be between 0 and chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:

        lines = document["text"].splitlines()

        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()

        if not content:
            continue

        text_chunks = splitter.split_text(content)

        for index, text in enumerate(text_chunks):

            raw_id = (
                f"{document['document_id']}:{index}:{text}"
            )

            chunk_id = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document["document_id"],
                "source": document["source"],
                "title": title,
                "text": text,
            })

    return chunks


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    documents_path = project_root / "data" / "documents"

    documents = load_documents(
        str(documents_path)
    )

    chunks = create_chunks(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks:

        print("\n" + "=" * 50)
        print("Chunk ID:", chunk["chunk_id"])
        print("Title:", chunk["title"])
        print("Source:", chunk["source"])
        print("Text:", chunk["text"])
