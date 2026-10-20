from qdrant_client import models

from embeddings.embedder import embed_text
from retrieval.reranker import rerank_chunks
from retrieval.vector_store import (
    get_qdrant_client,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    BM25_MODEL,
)


def retrieve_chunks(
    question: str,
    limit: int = 5,
    dense_limit: int = 20,
    sparse_limit: int = 20,
    candidate_limit: int = 10,
) -> list[dict]:
    """
    Retrieve candidates using Qdrant hybrid search and rerank them with Jina.

    Pipeline:
        Dense retrieval:       20 candidates
        Sparse BM25 retrieval: 20 candidates
        RRF fusion:            10 candidates
        Jina reranking:        final 5 candidates
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    limits = {
        "limit": limit,
        "dense_limit": dense_limit,
        "sparse_limit": sparse_limit,
        "candidate_limit": candidate_limit,
    }

    for name, value in limits.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if limit > candidate_limit:
        raise ValueError(
            "limit cannot exceed candidate_limit."
        )

    question = question.strip()

    # 1. Create the dense query embedding.
    query_vector = embed_text(
        question,
        input_type="query",
    )

    # 2. Connect to Qdrant Cloud.
    client = get_qdrant_client()

    # 3. Retrieve dense + sparse candidates and fuse using RRF.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                limit=dense_limit,
            ),
            models.Prefetch(
                query=models.Document(
                    text=question,
                    model=BM25_MODEL,
                ),
                using=SPARSE_VECTOR_NAME,
                limit=sparse_limit,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=candidate_limit,
        with_payload=True,
        with_vectors=False,
    )

    # 4. Convert Qdrant points into the existing chunk format.
    candidates = []

    for point in response.points:
        payload = point.payload or {}

        candidates.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "parent_id": payload.get("parent_id"),
                "document_id": payload.get("document_id"),
                "location": payload.get("location"),
                "content": payload.get("content"),
                "metadata": payload.get("metadata"),
                "score": point.score,
            }
        )

    # 5. Rerank the candidates using Jina Reranker M0.
    if not candidates:
        return []

    results = rerank_chunks(
        question=question,
        chunks=candidates,
        top_n=min(limit, len(candidates)),
    )

    # 6. Return the final ranked chunks.
    return results