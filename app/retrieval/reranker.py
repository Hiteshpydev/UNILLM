import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-m0"


def _image_to_data_url(image_path: str) -> str:
    """Convert a local image into a Base64 data URL."""
    path = Path(image_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_types.get(suffix)

    if mime_type is None:
        raise ValueError(f"Unsupported image format: {suffix}")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _chunk_to_document(chunk: dict) -> dict | None:
    """Convert a retrieved chunk into a Jina multimodal document."""
    content = chunk.get("content") or {}
    content_type = content.get("type")

    if content_type == "text":
        text = content.get("text", "").strip()
        if text:
            return {"text": text}

    elif content_type == "table":
        data = content.get("data")
        if data is not None:
            return {"text": _table_to_text(data)}

    elif content_type == "image":
        image_path = content.get("image_path")
        if image_path:
            return {"image": _image_to_data_url(image_path)}

    return None


def _table_to_text(data) -> str:
    """Convert table data into readable text."""
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
        return "\n".join(f"{key}: {value}" for key, value in data.items())

    return str(data)


def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Rerank retrieved text, table, and image chunks using Jina Reranker M0.

    Returns the original chunk dictionaries, ordered by reranker relevance.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    if not chunks:
        return []

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY is missing from the project .env file.")

    documents = []
    original_indices = []

    for index, chunk in enumerate(chunks):
        document = _chunk_to_document(chunk)
        if document is not None:
            documents.append(document)
            original_indices.append(index)

    if not documents:
        return []

    payload = {
        "model": JINA_RERANK_MODEL,
        "query": question.strip(),
        "documents": documents,
        "top_n": min(top_n, len(documents)),
        "return_documents": False,
    }

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" API response: {exc.response.text[:1000]}"
        raise RuntimeError(f"Jina reranking request failed.{detail}") from exc

    try:
        ranked_items = response.json()["results"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Jina returned an unexpected reranking response.") from exc

    ranked_chunks = []

    for item in ranked_items:
        document_index = item.get("index")

        if not isinstance(document_index, int):
            continue

        if not 0 <= document_index < len(original_indices):
            continue

        original_chunk_index = original_indices[document_index]
        chunk = dict(chunks[original_chunk_index])
        chunk["rerank_score"] = item.get("relevance_score")
        ranked_chunks.append(chunk)

    return ranked_chunks