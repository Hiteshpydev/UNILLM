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
    """Convert table data into searchable text."""

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
            f"{key}: {value}"
            for key, value in data.items()
        )

    return str(data)


def index_document(document: dict) -> dict:
    """
    Chunk, embed, and index a document in Qdrant.

    Every indexed chunk receives the document's session_id
    so that session-specific cleanup can be implemented.
    """

    metadata = document["metadata"]

    document_id = metadata["document_id"]
    session_id = metadata.get("session_id")

    if not session_id:
        raise ValueError(
            "Missing session_id in document metadata."
        )

    # Create parent and child chunks.
    chunked = create_chunks(document)

    qdrant_chunks = []

    for child in chunked["children"]:

        content = child["content"]
        content_type = content["type"]

        text_for_search = None

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        if content_type == "text":

            text_for_search = content["text"]

            dense_vector = embed_text(
                text_for_search,
                input_type="document",
            )

        elif content_type == "table":

            text_for_search = _table_to_text(
                content["data"]
            )

            dense_vector = embed_text(
                text_for_search,
                input_type="document",
            )

        elif content_type == "image":

            image_path = Path(
                content["image_path"]
            )

            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            dense_vector = embed_image(
                str(image_path)
            )

        else:

            raise ValueError(
                f"Unsupported chunk type: {content_type}"
            )

        # --------------------------------------------------
        # Prepare Qdrant payload
        # --------------------------------------------------

        payload = {
            "chunk_id": child["chunk_id"],
            "parent_id": child["parent_id"],
            "document_id": child["document_id"],
            "session_id": session_id,
            "location": child["location"],
            "content": content,
            "metadata": {
                **child["metadata"],
                "session_id": session_id,
            },
        }

        qdrant_chunk = {
            "id": f"{session_id}:{child['chunk_id']}",
            "dense_vector": dense_vector,
            "payload": payload,
        }

        # BM25 text is stored for text and table chunks.
        if (
            text_for_search
            and text_for_search.strip()
        ):
            qdrant_chunk["text"] = text_for_search

        qdrant_chunks.append(qdrant_chunk)

    # --------------------------------------------------
    # Store chunks in Qdrant Cloud
    # --------------------------------------------------

    client = get_qdrant_client()

    create_collection(
        client,
        COLLECTION_NAME,
    )

    upsert_chunks(
        client,
        qdrant_chunks,
        COLLECTION_NAME,
    )

    return {
        "document_id": document_id,
        "session_id": session_id,
        "parent_count": len(chunked["parents"]),
        "child_count": len(chunked["children"]),
        "indexed_count": len(qdrant_chunks),
    }