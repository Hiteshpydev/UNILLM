
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "unillm_chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
VECTOR_SIZE = 1024
BM25_MODEL = "qdrant/bm25"


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "Set QDRANT_URL and QDRANT_API_KEY in the project .env file."
        )

    return QdrantClient(
        url=url,
        api_key=api_key,
        cloud_inference=True,
    )


def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Create a collection with named dense and BM25 vectors."""
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


def _point_id(identifier: str | int) -> str:
    """Generate a stable UUID from a chunk identifier."""
    return str(uuid5(NAMESPACE_URL, str(identifier)))


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Store text and image chunks.

    Required fields per chunk:
        id: stable unique chunk identifier
        dense_vector: 1024-dimensional Jina CLIP v2 embedding
        payload: metadata dictionary

    Text chunks additionally require:
        text: text used for Qdrant BM25 inference

    Image chunks can omit text and store only the dense vector.
    """
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
                f"Chunk {identifier!r}: expected {VECTOR_SIZE} "
                f"dense dimensions, got {len(dense_vector)}."
            )

        if not isinstance(payload, dict):
            raise TypeError("Chunk payload must be a dictionary.")

        vectors = {
            DENSE_VECTOR_NAME: dense_vector,
        }

        # Qdrant generates the sparse BM25 vector server-side.
        # Image-only chunks do not receive a sparse vector.
        if text is not None and text.strip():
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
