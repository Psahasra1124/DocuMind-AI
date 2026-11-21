import unittest
from unittest.mock import Mock, patch

from app.services.embeddings import sentence_transformer
from app.services.embeddings.sentence_transformer import (
    MiniLMEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


class MiniLMEmbeddingProviderTests(unittest.TestCase):
    def tearDown(self):
        sentence_transformer._get_model.cache_clear()

    def test_initialization_does_not_load_model(self):
        with patch(
            "app.services.embeddings.sentence_transformer._get_model"
        ) as get_model:
            MiniLMEmbeddingProvider()

        get_model.assert_not_called()

    def test_embed_returns_list_of_float_values(self):
        model = Mock(return_value=[[0.1, 0.2, 0.3]])

        with patch(
            "app.services.embeddings.sentence_transformer._get_model",
            return_value=model,
        ) as get_model:
            result = MiniLMEmbeddingProvider().embed("hello")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        get_model.assert_called_once_with("all-MiniLM-L6-v2")
        model.assert_called_once_with(["hello"])

    def test_embed_many_preserves_input_order(self):
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        model = Mock(return_value=embeddings)

        with patch(
            "app.services.embeddings.sentence_transformer._get_model",
            return_value=model,
        ):
            result = MiniLMEmbeddingProvider().embed_many(
                ["first", "second"],
            )

        self.assertEqual(result, embeddings)
        model.assert_called_once_with(["first", "second"])

    def test_empty_batch_does_not_load_model(self):
        with patch(
            "app.services.embeddings.sentence_transformer._get_model"
        ) as get_model:
            result = MiniLMEmbeddingProvider().embed_many([])

        self.assertEqual(result, [])
        get_model.assert_not_called()

    def test_provider_rejects_unsupported_model(self):
        with self.assertRaisesRegex(ValueError, "Only all-MiniLM-L6-v2"):
            MiniLMEmbeddingProvider("another-model")

    def test_legacy_provider_name_remains_available(self):
        self.assertIs(
            SentenceTransformerEmbeddingProvider,
            MiniLMEmbeddingProvider,
        )

    def test_onnx_model_is_constructed_once(self):
        constructor_calls = []

        class FakeONNXMiniLM:
            def __init__(self, preferred_providers):
                constructor_calls.append(preferred_providers)

        with patch(
            "chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2",
            FakeONNXMiniLM,
        ):
            sentence_transformer._get_model.cache_clear()
            first = sentence_transformer._get_model("all-MiniLM-L6-v2")
            second = sentence_transformer._get_model("all-MiniLM-L6-v2")

        self.assertIs(first, second)
        self.assertEqual(
            constructor_calls,
            [["CPUExecutionProvider"]],
        )
        values = [1.0, 2.0]
        self.assertIs(first._normalize(values), values)


if __name__ == "__main__":
    unittest.main()
