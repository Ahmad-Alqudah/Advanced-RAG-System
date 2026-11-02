
# ============================================================
# PROJECT 02 — ADVANCED RAG SYSTEM
# PART 01 — DOCUMENT INGESTION
# FILE: app/ingestion/document_loader.py
#
# PURPOSE:
# Load TXT documents and preserve source metadata.
# ============================================================

from pathlib import Path


def load_documents(folder_path: str) -> list[dict]:
    """
    Load all TXT documents from a folder.
    """

    folder = Path(folder_path)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Document folder not found: {folder}"
        )

    documents = []

    for file_path in sorted(folder.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not text:
            continue

        documents.append({
            "document_id": file_path.stem,
            "text": text,
            "source": file_path.name,
        })

    if not documents:
        raise ValueError(
            "No readable TXT documents found."
        )

    return documents


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    documents_path = project_root / "data" / "documents"

    loaded_documents = load_documents(
        str(documents_path)
    )

    print(f"Loaded documents: {len(loaded_documents)}")

    for document in loaded_documents:
        print(
            document["document_id"],
            "|",
            document["source"],
            "|",
            len(document["text"]),
            "characters",
        )
