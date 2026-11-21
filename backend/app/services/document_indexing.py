from pathlib import Path
from threading import Lock

from app.core.config import settings
from app.services.document_chunking import iter_page_chunks, iter_text_chunks
from app.services.document_parsers.parsers import PDFDocumentParser
from app.services.document_parsers.selector import get_document_parser
from app.services.embeddings.base import EmbeddingProvider
from app.services.vector_store.base import VectorStore


INDEX_BATCH_SIZE = 16
_INDEX_LOCK = Lock()


class DocumentChunkLimitExceeded(Exception):
    """Raised when indexing would exceed the configured vector-store limit."""


class DocumentIndexer:
    """Orchestrate document parsing, chunking, embedding, and vector storage."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index_document(
        self,
        document_id: str,
        file_path: Path,
        owner_id: str,
    ) -> int:
        with _INDEX_LOCK:
            if not document_id:
                raise ValueError("document_id must not be empty")
            if not owner_id:
                raise ValueError("owner_id must not be empty")

            existing_count = self.vector_store.count_by_document_id(document_id)
            expected_count = self.count_document_chunks(file_path)
            total_count = self.vector_store.count()
            total_without_document = total_count - existing_count

            if (
                total_without_document + expected_count
                > settings.MAX_TOTAL_INDEXED_CHUNKS
            ):
                raise DocumentChunkLimitExceeded(
                    "The indexed-chunk limit of "
                    f"{settings.MAX_TOTAL_INDEXED_CHUNKS} has been reached. "
                    "Delete an indexed document before uploading another."
                )

            if existing_count == expected_count:
                return expected_count
            if existing_count:
                self.vector_store.delete_by_document_id(document_id)

            return self._index_document(document_id, file_path, owner_id)

    @staticmethod
    def count_document_chunks(file_path: Path) -> int:
        parser = get_document_parser(file_path)

        if isinstance(parser, PDFDocumentParser):
            chunks = iter_page_chunks(parser.iter_pages(file_path))
        else:
            chunks = iter_text_chunks(parser.parse(file_path))

        return sum(1 for _ in chunks)

    def _index_document(
        self,
        document_id: str,
        file_path: Path,
        owner_id: str,
    ) -> int:
        parser = get_document_parser(file_path)

        if isinstance(parser, PDFDocumentParser):
            chunks = iter_page_chunks(parser.iter_pages(file_path))
        else:
            chunks = (
                (None, chunk)
                for chunk in iter_text_chunks(parser.parse(file_path))
            )

        indexed_count = 0
        batch: list[tuple[int | None, str]] = []

        for page_number, text in chunks:
            batch.append((page_number, text))
            if len(batch) == INDEX_BATCH_SIZE:
                self._index_batch(
                    batch,
                    start_index=indexed_count,
                    document_id=document_id,
                    owner_id=owner_id,
                )
                indexed_count += len(batch)
                batch.clear()

        if batch:
            self._index_batch(
                batch,
                start_index=indexed_count,
                document_id=document_id,
                owner_id=owner_id,
            )
            indexed_count += len(batch)

        return indexed_count

    def _index_batch(
        self,
        batch: list[tuple[int | None, str]],
        start_index: int,
        document_id: str,
        owner_id: str,
    ) -> None:
        texts = [text for _, text in batch]
        embeddings = self.embedding_provider.embed_many(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedding provider must return one vector per text"
            )

        ids = [
            f"{document_id}:{start_index + offset}"
            for offset in range(len(batch))
        ]
        metadatas = []
        for offset, (page_number, _) in enumerate(batch):
            metadata = {
                "owner_id": owner_id,
                "document_id": document_id,
                "chunk_index": str(start_index + offset),
            }
            if page_number is not None:
                metadata["page_number"] = str(page_number)
            metadatas.append(metadata)

        self.vector_store.add(
            ids=ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
        )
