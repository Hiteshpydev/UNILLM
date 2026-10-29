import base64
import os
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

    return embedding
