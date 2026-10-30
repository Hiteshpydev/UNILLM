import unittest
from unittest.mock import Mock, patch

from qdrant_client import models

from app.retrieval.vector_store import delete_session_chunks


class TestDeleteSessionChunks(unittest.TestCase):
    def test_deletes_with_a_qdrant_filter_and_verifies_completion(self):
        client = Mock()
        client.collection_exists.return_value = True
        client.count.side_effect = [
            Mock(count=3),
            Mock(count=0),
        ]

        with patch(
            "app.retrieval.vector_store.ensure_session_id_index"
        ):
            deleted_count = delete_session_chunks(
                client=client,
                session_id="session-1",
            )

        self.assertEqual(deleted_count, 3)
        client.delete.assert_called_once()
        selector = client.delete.call_args.kwargs["points_selector"]
        self.assertIsInstance(selector, models.FilterSelector)
        self.assertEqual(
            selector.filter.should[0].match.value,
            "session-1",
        )
        client.scroll.assert_not_called()
