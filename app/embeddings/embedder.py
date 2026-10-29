import base64
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import requests
from dotenv import load_dotenv


# Load the existing project .env file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-clip-v2"
EMBEDDING_DIMENSIONS = 1024
REQUEST_TIMEOUT = 60
MAX_EMBEDDING_TEXT_CHARS = 12000
EMBEDDING_CACHE_PATH = PROJECT_ROOT / "data" / "embedding_cache.sqlite3"


def _cache_key(content: bytes, input_type: str) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return f"{JINA_MODEL}:{input_type}:{EMBEDDING_DIMENSIONS}:{digest}"


def _get_cached_embedding(cache_key: str) -> list[float] | None:
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(EMBEDDING_CACHE_PATH, timeout=30)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT vector_json FROM embeddings WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

    if row is None:
        return None

    vector = json.loads(row[0])
    if (
        not isinstance(vector, list)
        or len(vector) != EMBEDDING_DIMENSIONS
        or not all(isinstance(value, (int, float)) for value in vector)
    ):
        raise ValueError("Embedding cache contains an invalid vector.")

    return vector


def _cache_embedding(cache_key: str, embedding: list[float]) -> None:
    with closing(sqlite3.connect(EMBEDDING_CACHE_PATH, timeout=30)) as connection:
        with connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embeddings (cache_key, vector_json)
                VALUES (?, ?)
                """,
                (cache_key, json.dumps(embedding)),
            )


def embed_text(
    text: str,
    input_type: str = "document",
) -> list[float]:
    """
    Generate a normalized Jina CLIP v2 text embedding.

    input_type:
        document -> retrieval.query (indexing)
        query    -> retrieval.query (searching)
    """

    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        raise ValueError(
            "JINA_API_KEY is missing from the project .env file."
        )

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text to embed must be a non-empty string.")

    text = text.strip()[:MAX_EMBEDDING_TEXT_CHARS]

    task_mapping = {
        "document": "retrieval.query",
        "query": "retrieval.query",
    }

    if input_type not in task_mapping:
        raise ValueError(
            "input_type must be 'document' or 'query'."
        )

    cache_key = _cache_key(text.encode("utf-8"), input_type)
    cached_embedding = _get_cached_embedding(cache_key)
    if cached_embedding is not None:
        return cached_embedding

    payload = {
        "model": JINA_MODEL,
        "input": [{"text": text}],
        "dimensions": EMBEDDING_DIMENSIONS,
        "normalized": True,
        "embedding_type": "float",
        "task": task_mapping[input_type],
    }

    response = requests.post(
        JINA_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"Jina API error ({response.status_code}): "
            f"{response.text}"
        )

    result = response.json()
    embedding = result["data"][0]["embedding"]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"received {len(embedding)}."
        )

    _cache_embedding(cache_key, embedding)
    return embedding



def embed_image(image_path: str) -> list[float]:
    """Generate a normalized Jina CLIP v2 image embedding."""

    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        raise ValueError(
            "JINA_API_KEY is missing from the project .env file."
        )

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    cache_key = _cache_key(image_bytes, "image")
    cached_embedding = _get_cached_embedding(cache_key)
    if cached_embedding is not None:
        return cached_embedding

    payload = {
        "model": JINA_MODEL,
        "input": [{"image": image_base64}],
        "dimensions": EMBEDDING_DIMENSIONS,
        "normalized": True,
        "embedding_type": "float",
    }

    response = requests.post(
        JINA_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"Jina API error ({response.status_code}): "
            f"{response.text}"
        )

    result = response.json()
    embedding = result["data"][0]["embedding"]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"received {len(embedding)}."
        )

    _cache_embedding(cache_key, embedding)
    return embedding
