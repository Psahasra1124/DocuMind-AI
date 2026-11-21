from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies.auth import get_current_user
from app.db.models import User
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.chat import ChatService
from app.services.chat_factory import get_chat_service
from app.services.llm import (
    CloudflareAPIError,
    CloudflareRateLimitError,
    CloudflareTimeoutError,
    GeminiAPIError,
    GeminiTimeoutError,
)


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        response = await chat_service.chat(
            user_id=current_user.id,
            query=request.query,
            top_k=request.top_k,
            conversation_id=request.conversation_id,
        )
    except GeminiTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except GeminiAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except CloudflareTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except CloudflareRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "30"},
        ) from exc
    except CloudflareAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        query=response.query,
        answer=response.answer,
        sources=[
            ChatSource(
                text=source.text,
                document_id=source.document_id,
                chunk_index=source.chunk_index,
                distance=source.distance,
                metadata=dict(source.metadata),
            )
            for source in response.sources
        ],
    )
