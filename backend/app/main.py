import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.database import AsyncSessionLocal

from app.core.config import settings

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.documents import router as documents_router
from app.api.routes.retrieval import router as retrieval_router


logger = logging.getLogger(__name__)

INTERNAL_ERROR_DETAIL = "Internal server error"


app = FastAPI(
    title="DocuMind AI API",
    version="1.0.0",
    description="Document-grounded question answering API with tenant-scoped retrieval and source evidence",
)


@app.exception_handler(Exception)
async def internal_server_error_handler(
    request: Request,
    exc: Exception,
):
    logger.exception("Unhandled application exception")
    return JSONResponse(
        status_code=500,
        content={"detail": INTERNAL_ERROR_DETAIL},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(documents_router)
app.include_router(retrieval_router)


@app.get("/health", tags=["operations"])
async def health_check():
    """Liveness probe: the API process is running."""
    return {"status": "healthy"}


@app.get("/ready", tags=["operations"])
async def readiness_check():
    """Readiness probe: the API can reach its primary database."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Readiness check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "dependencies": {"database": "unavailable"}},
        )
    return {"status": "ready", "dependencies": {"database": "available"}}
