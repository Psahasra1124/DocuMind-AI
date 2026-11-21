from functools import lru_cache

from app.services.document_indexing import DocumentIndexer
from app.services.embeddings.factory import get_embedding_provider
from app.services.vector_store_factory import get_vector_store


@lru_cache(maxsize=1)
def get_document_indexer() -> DocumentIndexer:
    return DocumentIndexer(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
    )
