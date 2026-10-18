
from pathlib import Path

from chunking.chunker import create_chunks
from embeddings.embedder import embed_text, embed_image
from retrieval.vector_store import (
    get_qdrant_client,
    create_collection,
    upsert_chunks,
    COLLECTION_NAME,
)


def _table_to_text(data) -> str:
    """Convert extracted table data into searchable text."""
    if isinstance(data, str):
        return data

    if isinstance(data, (list, tuple)):
        return "\n".join(
            " | ".join(str(cell) for cell in row)
            if isinstance(row, (list, tuple))
            else str(row)
            for row in data
        )

    if isinstance(data, dict):
        return "\n".join(
            f"{key}: {value}" for key, value in data.items()
        )

    return str(data)


def index_document(document: dict) -> dict:
    """
    Chunk an ingested document, embed its children,
    and upsert them into Qdrant Cloud.
    """
    document_id = document["metadata"]["document_id"]
    chunked = create_chunks(document)

    qdrant_chunks = []

    for child in chunked["children"]:
        content = child["content"]
        content_type = content["type"]

        text_for_search = None

        if content_type == "text":
            text_for_search = content["text"]
            dense_vector = embed_text(
                text_for_search,
                input_type="document",
            )

        elif content_type == "table":
            text_for_search = _table_to_text(content["data"])
            dense_vector = embed_text(
                text_for_search,
                input_type="document",
            )

        elif content_type == "image":
            image_path = Path(content["image_path"])

            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            dense_vector = embed_image(str(image_path))

        else:
            raise ValueError(
                f"Unsupported child content type: {content_type}"
            )

        payload = {
            "chunk_id": child["chunk_id"],
            "parent_id": child["parent_id"],
            "document_id": child["document_id"],
            "location": child["location"],
            "content": content,
            "metadata": child["metadata"],
        }

        qdrant_chunk = {
            "id": child["chunk_id"],
            "dense_vector": dense_vector,
            "payload": payload,
        }

        # Qdrant's BM25 inference needs text, not image bytes.
        if text_for_search and text_for_search.strip():
            qdrant_chunk["text"] = text_for_search

        qdrant_chunks.append(qdrant_chunk)

    client = get_qdrant_client()
    create_collection(client, COLLECTION_NAME)
    upsert_chunks(client, qdrant_chunks, COLLECTION_NAME)

    return {
        "document_id": document_id,
        "parent_count": len(chunked["parents"]),
        "child_count": len(chunked["children"]),
        "indexed_count": len(qdrant_chunks),
    }
