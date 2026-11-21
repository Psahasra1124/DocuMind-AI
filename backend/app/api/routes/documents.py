import asyncio
import logging
from typing import Annotated
from uuid import UUID

from starlette.concurrency import run_in_threadpool

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.core.config import settings
from app.db.database import get_db
from app.db.models import Document, User
from app.schemas.documents import DocumentListResponse, DocumentResponse
from app.services.document_deletion import DocumentDeletionService
from app.services.document_indexing import DocumentChunkLimitExceeded
from app.services.document_indexing_factory import get_document_indexer
from app.services.document_storage import (
    delete_document,
    get_storage_root,
    store_document,
)
from app.services.document_validation import validate_document_upload
from app.services.vector_store_factory import get_vector_store


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)
_UPLOAD_LOCK = asyncio.Lock()
logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=DocumentListResponse,
)
async def list_documents(
    search: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    filters = [
        Document.owner_id == current_user.id,
    ]

    if search:
        filters.append(Document.original_filename.ilike(f"%{search}%"))

    total_result = await db.execute(
        select(func.count(Document.id)).where(*filters)
    )
    total_count = total_result.scalar_one()

    result = await db.execute(
        select(Document)
        .where(*filters)
        .order_by(
            Document.created_at.desc(),
            Document.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    return DocumentListResponse(
        items=list(result.scalars().all()),
        total_count=total_count,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    validate_document_upload(file)

    async with _UPLOAD_LOCK:
        document_count_result = await db.execute(
            select(func.count(Document.id)).where(
                Document.owner_id == current_user.id
            )
        )
        document_count = document_count_result.scalar_one()
        if document_count >= settings.MAX_DOCUMENT_COUNT:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Your workspace has reached the "
                    f"{settings.MAX_DOCUMENT_COUNT}-document limit. "
                    "Delete a document before uploading another."
                ),
            )

        return await _store_and_index_document(file, current_user, db)


async def _store_and_index_document(
    file: UploadFile,
    current_user: User,
    db: AsyncSession,
):
    storage_path = await store_document(
        file,
        current_user.id,
    )

    document = Document(
        owner_id=current_user.id,
        original_filename=file.filename,
        mime_type=file.content_type,
        storage_path=storage_path,
    )

    try:
        db.add(document)
        await db.commit()
        await db.refresh(document)
    except Exception:
        await db.rollback()
        delete_document(storage_path)
        raise

    indexer = get_document_indexer()
    vector_store = get_vector_store()

    try:
        await run_in_threadpool(
            indexer.index_document,
            document_id=str(document.id),
            file_path=get_storage_root() / document.storage_path,
            owner_id=str(current_user.id),
        )
    except DocumentChunkLimitExceeded as exc:
        await db.delete(document)
        await db.commit()
        delete_document(storage_path)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception:
        try:
            vector_store.delete_by_document_id(
                str(document.id)
            )
        except Exception:
            logger.exception(
                "Failed to clean up vectors after document indexing failed",
                extra={"document_id": str(document.id)},
            )
        raise

    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document_route(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        document_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    result = await db.execute(
        select(Document).where(
            Document.id == document_uuid,
            Document.owner_id == current_user.id,
        )
    )
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    deletion_service = DocumentDeletionService(
        db=db,
        vector_store=get_vector_store(),
    )

    await deletion_service.delete_owned_document(document)
