import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.document_chunking import chunk_text
from app.services.document_indexing import (
    DocumentChunkLimitExceeded,
    DocumentIndexer,
)


class FakeEmbeddingProvider:
    def __init__(self):
        self.calls = []

    def embed_many(self, texts):
        self.calls.append(list(texts))

        return [
            [float(index), float(len(text))]
            for index, text in enumerate(texts)
        ]


class FakeVectorStore:
    def __init__(self):
        self.calls = []
        self.vectors = {}

    def add(
        self,
        ids,
        embeddings,
        texts,
        metadatas=None,
    ):
        self.calls.append(
            {
                "ids": list(ids),
                "embeddings": [list(embedding) for embedding in embeddings],
                "texts": list(texts),
                "metadatas": list(metadatas or []),
            }
        )
        for index, vector_id in enumerate(ids):
            self.vectors[vector_id] = {
                "text": texts[index],
                "metadata": (metadatas or [])[index],
            }

    def count(self):
        return len(self.vectors)

    def count_by_document_id(self, document_id):
        return sum(
            vector["metadata"]["document_id"] == document_id
            for vector in self.vectors.values()
        )

    def delete_by_document_id(self, document_id):
        self.vectors = {
            vector_id: vector
            for vector_id, vector in self.vectors.items()
            if vector["metadata"]["document_id"] != document_id
        }


class DocumentIndexingTests(unittest.TestCase):
    def test_indexes_document_using_parser_chunker_embedding_and_vector_store(
        self,
    ):
        document_id = "document-123"
        original_text = (
            "abcdefghij"
        )
        expected_chunks = chunk_text(
            original_text,
            chunk_size=5,
            chunk_overlap=0,
        )

        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore()
        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "document.txt"
            file_path.write_text(
                original_text,
                encoding="utf-8",
            )

            with patch(
                "app.services.document_indexing.iter_text_chunks",
                side_effect=lambda text: chunk_text(
                    text,
                    chunk_size=5,
                    chunk_overlap=0,
                ),
            ) as chunker:
                indexed_count = indexer.index_document(
                    document_id=document_id,
                    file_path=file_path,
                    owner_id="user-123",
                )

        self.assertEqual(
            indexed_count,
            len(expected_chunks),
        )
        self.assertEqual(
            chunker.call_args_list,
            [((original_text,), {}), ((original_text,), {})],
        )
        self.assertEqual(
            embedding_provider.calls,
            [expected_chunks],
        )
        self.assertEqual(
            vector_store.calls,
            [
                {
                    "ids": [
                        "document-123:0",
                        "document-123:1",
                    ],
                    "embeddings": [
                        [0.0, 5.0],
                        [1.0, 5.0],
                    ],
                    "texts": expected_chunks,
                    "metadatas": [
                        {
                            "owner_id": "user-123",
                            "document_id": "document-123",
                            "chunk_index": "0",
                        },
                        {
                            "owner_id": "user-123",
                            "document_id": "document-123",
                            "chunk_index": "1",
                        },
                    ],
                }
            ],
        )
        self.assertEqual(vector_store.count(), len(expected_chunks))

    def test_empty_document_returns_zero_without_embedding_or_vector_store(
        self,
    ):
        embedding_provider = Mock()
        vector_store = Mock()
        vector_store.count.return_value = 0
        vector_store.count_by_document_id.return_value = 0
        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "empty.txt"
            file_path.write_text(
                "",
                encoding="utf-8",
            )

            indexed_count = indexer.index_document(
                document_id="empty-document",
                file_path=file_path,
                owner_id="user-123",
            )

        self.assertEqual(indexed_count, 0)
        embedding_provider.embed_many.assert_not_called()
        vector_store.add.assert_not_called()

    def test_parser_failure_propagates(self):
        embedding_provider = Mock()
        vector_store = Mock()
        vector_store.count.return_value = 0
        vector_store.count_by_document_id.return_value = 0
        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        parser = Mock()
        parser.parse.side_effect = RuntimeError(
            "parser failure"
        )

        with patch(
            "app.services.document_indexing.get_document_parser",
            return_value=parser,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "parser failure",
            ):
                indexer.index_document(
                    document_id="document-123",
                    file_path=Path("document.txt"),
                    owner_id="user-123",
                )

        embedding_provider.embed_many.assert_not_called()
        vector_store.add.assert_not_called()

    def test_embedding_failure_propagates(self):
        embedding_provider = Mock()
        vector_store = Mock()
        vector_store.count.return_value = 0
        vector_store.count_by_document_id.return_value = 0
        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        embedding_provider.embed_many.side_effect = RuntimeError(
            "embedding failure"
        )

        parser = Mock()
        parser.parse.return_value = "abcdefghij"

        with patch(
            "app.services.document_indexing.get_document_parser",
            return_value=parser,
        ):
            with patch(
                "app.services.document_indexing.iter_text_chunks",
                return_value=["abc", "def"],
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "embedding failure",
                ):
                    indexer.index_document(
                        document_id="document-123",
                        file_path=Path("document.txt"),
                        owner_id="user-123",
                    )

        vector_store.add.assert_not_called()

    def test_vector_store_failure_propagates(self):
        embedding_provider = FakeEmbeddingProvider()
        vector_store = Mock()
        vector_store.count.return_value = 0
        vector_store.count_by_document_id.return_value = 0
        vector_store.add.side_effect = RuntimeError(
            "vector store failure"
        )

        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        parser = Mock()
        parser.parse.return_value = "abcdefghij"

        with patch(
            "app.services.document_indexing.get_document_parser",
            return_value=parser,
        ):
            with patch(
                "app.services.document_indexing.iter_text_chunks",
                side_effect=lambda text: ["abc", "def"],
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "vector store failure",
                ):
                    indexer.index_document(
                        document_id="document-123",
                        file_path=Path("document.txt"),
                        owner_id="user-123",
                    )

        self.assertEqual(
            embedding_provider.calls,
            [["abc", "def"]],
        )

    def test_repeated_indexing_does_not_duplicate_vectors(self):
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore()
        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        parser = Mock()
        parser.parse.return_value = "abcdefghij"

        with patch(
            "app.services.document_indexing.get_document_parser",
            return_value=parser,
        ):
            with patch(
                "app.services.document_indexing.iter_text_chunks",
                side_effect=lambda text: ["abc", "def"],
            ):
                first_count = indexer.index_document(
                    document_id="document-123",
                    file_path=Path("document.txt"),
                    owner_id="user-123",
                )
                second_count = indexer.index_document(
                    document_id="document-123",
                    file_path=Path("document.txt"),
                    owner_id="user-123",
                )

        self.assertEqual(first_count, 2)
        self.assertEqual(second_count, 2)

        self.assertEqual(
            vector_store.calls[0]["ids"],
            [
                "document-123:0",
                "document-123:1",
            ],
        )
        self.assertEqual(len(vector_store.calls), 1)
        self.assertEqual(vector_store.count(), 2)

    def test_rejects_empty_owner_id(self):
        indexer = DocumentIndexer(
            embedding_provider=Mock(),
            vector_store=FakeVectorStore(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "owner_id must not be empty",
        ):
            indexer.index_document(
                document_id="document-123",
                file_path=Path("document.txt"),
                owner_id="",
            )

    def test_rejects_empty_document_id(self):
        indexer = DocumentIndexer(
            embedding_provider=Mock(),
            vector_store=FakeVectorStore(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "document_id must not be empty",
        ):
            indexer.index_document(
                document_id="",
                file_path=Path("document.txt"),
                owner_id="user-123",
            )

    def test_indexes_large_document_in_bounded_batches(self):
        chunks = [f"chunk-{index}" for index in range(5)]
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore()
        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        parser = Mock()
        parser.parse.return_value = "unused"

        with patch(
            "app.services.document_indexing.get_document_parser",
            return_value=parser,
        ), patch(
            "app.services.document_indexing.iter_text_chunks",
            side_effect=lambda text: iter(chunks),
        ), patch(
            "app.services.document_indexing.INDEX_BATCH_SIZE",
            2,
        ):
            indexed_count = indexer.index_document(
                document_id="document-123",
                file_path=Path("document.txt"),
                owner_id="user-123",
            )

        self.assertEqual(indexed_count, len(chunks))
        self.assertEqual(
            embedding_provider.calls,
            [chunks[:2], chunks[2:4], chunks[4:]],
        )
        self.assertEqual(
            [call["ids"] for call in vector_store.calls],
            [
                ["document-123:0", "document-123:1"],
                ["document-123:2", "document-123:3"],
                ["document-123:4"],
            ],
        )

    def test_pdf_page_numbers_are_preserved_across_batches(self):
        from app.services.document_parsers.parsers import PDFDocumentParser

        parser = Mock(spec=PDFDocumentParser)
        parser.iter_pages.side_effect = lambda file_path: iter(
            [(1, "first page"), (2, "second page")]
        )
        vector_store = FakeVectorStore()
        indexer = DocumentIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=vector_store,
        )

        with patch(
            "app.services.document_indexing.get_document_parser",
            return_value=parser,
        ), patch(
            "app.services.document_indexing.iter_text_chunks",
            side_effect=lambda text: [text],
        ):
            count = indexer.index_document(
                document_id="document-pdf",
                file_path=Path("document.pdf"),
                owner_id="user-123",
            )

        self.assertEqual(count, 2)
        self.assertEqual(
            [
                metadata["page_number"]
                for call in vector_store.calls
                for metadata in call["metadatas"]
            ],
            ["1", "2"],
        )

    def test_chunk_limit_rejects_before_embedding_or_partial_writes(self):
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore()
        vector_store.add(
            ids=["existing:0", "existing:1", "existing:2"],
            embeddings=[[0.0], [0.0], [0.0]],
            texts=["one", "two", "three"],
            metadatas=[
                {"document_id": "existing"},
                {"document_id": "existing"},
                {"document_id": "existing"},
            ],
        )
        indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        parser = Mock()
        parser.parse.return_value = "new document"

        with patch(
            "app.services.document_indexing.get_document_parser",
            return_value=parser,
        ), patch(
            "app.services.document_indexing.iter_text_chunks",
            side_effect=lambda text: ["new-0", "new-1"],
        ), patch(
            "app.services.document_indexing.settings.MAX_TOTAL_INDEXED_CHUNKS",
            4,
        ):
            with self.assertRaises(DocumentChunkLimitExceeded):
                indexer.index_document(
                    document_id="new",
                    file_path=Path("new.txt"),
                    owner_id="user-123",
                )

        self.assertEqual(embedding_provider.calls, [])
        self.assertEqual(vector_store.count(), 3)
        self.assertEqual(vector_store.count_by_document_id("new"), 0)


if __name__ == "__main__":
    unittest.main()
