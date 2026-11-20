from pathlib import Path

from app.services.document_chunking import chunk_pages, chunk_text
from app.services.document_parsers.parsers import PDFDocumentParser
from app.services.document_parsers.selector import get_document_parser
from app.services.embeddings.base import EmbeddingProvider
from app.services.vector_store.base import VectorStore


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
        if not document_id:
            raise ValueError("document_id must not be empty")
        if not owner_id:
            raise ValueError("owner_id must not be empty")

        parser = get_document_parser(file_path)

        # Preserve existing TXT/DOCX behavior; PDFs retain their page numbers.
        if isinstance(parser, PDFDocumentParser):
            page_chunks = chunk_pages(parser.parse_pages(file_path))
            chunks = [text for _, text in page_chunks]
            page_numbers: list[int | None] = [
                page_number for page_number, _ in page_chunks
            ]
        else:
            chunks = chunk_text(parser.parse(file_path))
            page_numbers = [None] * len(chunks)

        if not chunks:
            return 0

        embeddings = self.embedding_provider.embed_many(chunks)
        vector_ids = [
            f"{document_id}:{chunk_index}"
            for chunk_index in range(len(chunks))
        ]

        metadatas = []
        for chunk_index, page_number in enumerate(page_numbers):
            metadata = {
                "owner_id": owner_id,
                "document_id": document_id,
                "chunk_index": str(chunk_index),
            }
            if page_number is not None:
                metadata["page_number"] = str(page_number)
            metadatas.append(metadata)

        self.vector_store.add(
            ids=vector_ids,
            embeddings=embeddings,
            texts=chunks,
            metadatas=metadatas,
        )
        return len(chunks)
