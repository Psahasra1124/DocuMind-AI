from functools import lru_cache
from threading import Lock

from app.services.embeddings.base import EmbeddingProvider


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_LOCK = Lock()


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    if model_name != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Only {DEFAULT_EMBEDDING_MODEL} is supported by the "
            "lightweight ONNX embedding provider"
        )

    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _L2CompatibleMiniLM(ONNXMiniLM_L6_V2):
        def _normalize(self, values):
            # The previous SentenceTransformer output was unnormalized; Chroma uses L2 distance.
            return values

    return _L2CompatibleMiniLM(
        preferred_providers=["CPUExecutionProvider"],
    )


class MiniLMEmbeddingProvider(EmbeddingProvider):
    """Lazily embed with Chroma's CPU ONNX implementation of MiniLM."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        if model_name != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(
                f"Only {DEFAULT_EMBEDDING_MODEL} is supported by the "
                "lightweight ONNX embedding provider"
            )
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        model = _get_model(self.model_name)
        with _EMBEDDING_LOCK:
            embeddings = model(texts)
        return [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]


# Preserve imports used by older integrations.
SentenceTransformerEmbeddingProvider = MiniLMEmbeddingProvider
