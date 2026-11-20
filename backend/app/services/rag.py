from dataclasses import dataclass
from typing import Any

from app.services.llm.base import LLMProvider
from app.services.retrieval import RetrievalResult, Retriever


@dataclass(frozen=True)
class RagContext:
    query: str
    context: str
    sources: list[RetrievalResult]


@dataclass(frozen=True)
class RagResponse:
    query: str
    answer: str
    context: RagContext
    sources: list[RetrievalResult]


def build_rag_prompt(query: str, context: Any) -> str:
    context_text = context.context if context and context.context else ""
    return f"Question:\n{query}\n\nContext:\n{context_text}"


class RagService:
    """Build owner-scoped context and generate a document-grounded answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_provider = llm_provider

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        owner_id: str = "",
    ) -> RagContext:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not owner_id:
            raise ValueError("owner_id must not be empty")

        sources = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            owner_id=owner_id,
        )

        context_parts = []
        for index, source in enumerate(sources, start=1):
            page_number = source.metadata.get("page_number")
            label = f"Source {index}"
            if page_number not in (None, ""):
                label = f"{label} — Page {page_number}"
            context_parts.append(f"[{label}]\n{source.text}")

        return RagContext(
            query=query,
            context="\n\n".join(context_parts),
            sources=sources,
        )

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        owner_id: str = "",
    ) -> RagResponse:
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        if self.llm_provider is None:
            raise ValueError("LLM provider is required")

        context = self.build_context(
            query=query,
            top_k=top_k,
            owner_id=owner_id,
        )
        prompt = build_rag_prompt(query, context)
        answer = self.llm_provider.generate(prompt)

        return RagResponse(
            query=query,
            answer=answer,
            context=context,
            sources=context.sources,
        )