from typing import TYPE_CHECKING

from app.services.vector_store.base import (
    VectorQueryResult,
    VectorStore,
)

if TYPE_CHECKING:
    from app.services.vector_store.chroma import ChromaVectorStore

__all__ = [
    "ChromaVectorStore",
    "VectorQueryResult",
    "VectorStore",
]


def __getattr__(name: str):
    if name == "ChromaVectorStore":
        from app.services.vector_store.chroma import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
