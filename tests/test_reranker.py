import base64
from io import BytesIO
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.retrieval.reranker import rerank_chunks


class TestRerankerImages(unittest.TestCase):
    def test_resizes_images_before_sending_them(self):
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "source.jpg"
            Image.new("RGB", (1200, 600), color="red").save(image_path)
            captured_payload = {}

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"results": [{"index": 0, "relevance_score": 1.0}]}

            def fake_post(*args, **kwargs):
                captured_payload.update(kwargs["json"])
                return Response()

            with patch.dict("os.environ", {"JINA_API_KEY": "test-key"}):
                with patch(
                    "app.retrieval.reranker.requests.post",
                    side_effect=fake_post,
                ):
                    rerank_chunks(
                        "What is shown?",
                        [
                            {
                                "content": {
                                    "type": "image",
                                    "image_path": str(image_path),
                                }
                            }
                        ],
                    )

            encoded_image = captured_payload["documents"][0]["image"]
            with Image.open(
                BytesIO(base64.b64decode(encoded_image.split(",", 1)[1]))
            ) as reranked_image:
                self.assertEqual(reranked_image.size, (512, 512))
