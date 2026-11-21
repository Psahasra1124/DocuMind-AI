from functools import lru_cache

from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.sentence_transformer import MiniLMEmbeddingProvider


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return MiniLMEmbeddingProvider(settings.EMBEDDING_MODEL)
