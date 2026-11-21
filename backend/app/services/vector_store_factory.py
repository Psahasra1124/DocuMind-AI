from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.services.vector_store.base import VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    from app.services.vector_store.chroma import ChromaVectorStore

    return ChromaVectorStore(
        persist_directory=settings.VECTOR_STORE_DIR,
        collection_name=settings.VECTOR_COLLECTION_NAME,
    )
