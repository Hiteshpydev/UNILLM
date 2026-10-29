import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "unillm_chunks"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

VECTOR_SIZE = 1024
BM25_MODEL = "qdrant/bm25"


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud."""

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "Set QDRANT_URL and QDRANT_API_KEY "
            "in the project .env file."
        )

    return QdrantClient(
        url=url,
        api_key=api_key,
        cloud_inference=True,
    )


# ============================================================
# COLLECTION MANAGEMENT
# ============================================================

def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
):
    """Create the Qdrant collection if it does not exist."""

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


# ============================================================
# SESSION PAYLOAD INDEX
# ============================================================

def ensure_session_id_index(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
):
    """Ensure current and legacy session fields are indexed."""

    if not client.collection_exists(collection_name):
        create_collection(client, collection_name)

    for field_name in ("session_id", "metadata.session_id"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


# ============================================================
# POINT ID GENERATION
# ============================================================

def _point_id(identifier: str | int) -> str:
    """Generate a deterministic UUID from an identifier."""

    return str(uuid5(NAMESPACE_URL, str(identifier)))


# ============================================================
# UPSERT CHUNKS
# ============================================================

def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    collection_name: str = COLLECTION_NAME,
):
    """Insert or update chunks in Qdrant."""

    if not client.collection_exists(collection_name):
        create_collection(client, collection_name)

    points = []

    for chunk in chunks:

        identifier = chunk["id"]
        dense_vector = chunk["dense_vector"]
        payload = chunk.get("payload", {})
        text = chunk.get("text")

        if len(dense_vector) != VECTOR_SIZE:
            raise ValueError(
                f"Expected dense vector size {VECTOR_SIZE}, "
                f"got {len(dense_vector)}."
            )

        if not isinstance(payload, dict):
            raise TypeError(
                "Chunk payload must be a dictionary."
            )

        vectors = {
            DENSE_VECTOR_NAME: dense_vector,
        }

        # Generate the BM25 sparse vector from searchable text.
        if isinstance(text, str) and text.strip():
            vectors[SPARSE_VECTOR_NAME] = models.Document(
                text=text,
                model=BM25_MODEL,
            )

        points.append(
            models.PointStruct(
                id=_point_id(identifier),
                vector=vectors,
                payload=payload,
            )
        )

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    return len(points)


# ============================================================
# DELETE SESSION CHUNKS
# ============================================================

def delete_session_chunks(
    client: QdrantClient,
    session_id: str,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Delete only points belonging to the specified study session.

    Returns the number of matching points counted before deletion.
    """

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(
            "A non-empty session_id is required."
        )

    if not client.collection_exists(collection_name):
        return 0

    # Ensure Qdrant can filter by session_id.
    ensure_session_id_index(
        client,
        collection_name,
    )

    session_filter = models.Filter(
        should=[
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            ),
            models.FieldCondition(
                key="metadata.session_id",
                match=models.MatchValue(value=session_id),
            ),
        ],
    )

    # Count matching points before deletion.
    count_result = client.count(
        collection_name=collection_name,
        count_filter=session_filter,
        exact=True,
    )

    matching_count = count_result.count

    if matching_count == 0:
        return 0

    # Resolve matching IDs first, then delete those exact points.
    point_ids = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=session_filter,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        point_ids.extend(record.id for record in records)

        if offset is None:
            break

    if not point_ids:
        raise RuntimeError(
            f"Qdrant count found {matching_count} points for session "
            f"{session_id}, but they could not be read for deletion."
        )

    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(
            points=point_ids,
        ),
        wait=True,
        ordering=models.WriteOrdering.STRONG,
    )

    remaining_count = client.count(
        collection_name=collection_name,
        count_filter=session_filter,
        exact=True,
    ).count

    if remaining_count:
        raise RuntimeError(
            "Qdrant cleanup did not remove all points for session "
            f"{session_id}. Remaining points: {remaining_count}."
        )

    return matching_count