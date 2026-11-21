from app.core.config import settings
from app.services.llm.base import LLMProvider
from app.services.llm.cloudflare import CloudflareWorkersAIProvider
from app.services.llm.ollama import OllamaLLMProvider
from app.services.rag import RagService
from app.services.retrieval import Retriever


def get_llm_provider() -> LLMProvider:
    if settings.LLM_PROVIDER == "cloudflare":
        return CloudflareWorkersAIProvider(
            account_id=settings.CLOUDFLARE_ACCOUNT_ID,
            api_token=settings.CLOUDFLARE_API_TOKEN,
            model_name=settings.CLOUDFLARE_MODEL,
            timeout=settings.CLOUDFLARE_TIMEOUT_SECONDS,
        )

    if settings.LLM_PROVIDER == "ollama":
        return OllamaLLMProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")


def get_rag_service(
    retriever: Retriever,
) -> RagService:
    return RagService(
        retriever=retriever,
        llm_provider=get_llm_provider(),
    )
