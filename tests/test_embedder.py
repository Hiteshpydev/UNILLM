import base64
from io import BytesIO
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch

from app.embeddings.embedder import (
    IMAGE_EMBEDDING_SIZE,
    embed_image,
    embed_text,
)
from PIL import Image


class TestEmbedImage(unittest.TestCase):
    def test_sends_a_512_by_512_image(self):
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.jpg"
            Image.new("RGB", (1200, 600), color="red").save(source_path)

            captured_payload = {}

            class Response:
                ok = True

                def json(self):
                    return {"data": [{"embedding": [0.0] * 1024}]}

            def fake_post(*args, **kwargs):
                captured_payload.update(kwargs["json"])
                return Response()

            with patch.dict("os.environ", {"JINA_API_KEY": "test-key"}):
                with patch(
                    "app.embeddings.embedder.EMBEDDING_CACHE_PATH",
                    Path(temporary_directory) / "cache.sqlite3",
                ):
                    with patch(
                        "app.embeddings.embedder.requests.post",
                        side_effect=fake_post,
                    ):
                        embed_image(str(source_path))

            image_bytes = base64.b64decode(
                captured_payload["input"][0]["image"]
            )

            with Image.open(BytesIO(image_bytes)) as embedded_image:
                self.assertEqual(
                    embedded_image.size,
                    IMAGE_EMBEDDING_SIZE,
                )

    def test_reuses_cached_image_embedding(self):
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.jpg"
            Image.new("RGB", (1200, 600), color="red").save(source_path)
            request_count = 0

            class Response:
                ok = True

                def json(self):
                    return {"data": [{"embedding": [0.0] * 1024}]}

            def fake_post(*args, **kwargs):
                nonlocal request_count
                request_count += 1
                return Response()

            with patch.dict("os.environ", {"JINA_API_KEY": "test-key"}):
                with patch(
                    "app.embeddings.embedder.EMBEDDING_CACHE_PATH",
                    Path(temporary_directory) / "cache.sqlite3",
                ):
                    with patch(
                        "app.embeddings.embedder.requests.post",
                        side_effect=fake_post,
                    ):
                        first_embedding = embed_image(str(source_path))
                        second_embedding = embed_image(str(source_path))

            self.assertEqual(request_count, 1)
            self.assertEqual(first_embedding, second_embedding)

    def test_reuses_cached_text_embedding(self):
        with TemporaryDirectory() as temporary_directory:
            request_count = 0

            class Response:
                ok = True

                def json(self):
                    return {"data": [{"embedding": [0.0] * 1024}]}

            def fake_post(*args, **kwargs):
                nonlocal request_count
                request_count += 1
                return Response()

            with patch.dict("os.environ", {"JINA_API_KEY": "test-key"}):
                with patch(
                    "app.embeddings.embedder.EMBEDDING_CACHE_PATH",
                    Path(temporary_directory) / "cache.sqlite3",
                ):
                    with patch(
                        "app.embeddings.embedder.requests.post",
                        side_effect=fake_post,
                    ):
                        first_embedding = embed_text("same document chunk")
                        second_embedding = embed_text("same document chunk")

            self.assertEqual(request_count, 1)
            self.assertEqual(first_embedding, second_embedding)
