import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, Depends, FastAPI

from autobot_stt import __version__
from autobot_stt.config import get_settings
from autobot_stt.dependencies.auth import verify_api_key
from autobot_stt.routes import sessions_router, stream_router
from autobot_stt.services.whisper_service import WhisperService
from autobot_stt.stores.memory import InMemorySessionStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.session_store = InMemorySessionStore()
    app.state.whisper_lock = asyncio.Lock()

    settings = get_settings()
    service = WhisperService(settings)
    logger.info(
        "Loading Whisper model model=%s device=%s",
        settings.whisper_model,
        settings.whisper_device,
    )
    service.load()
    app.state.whisper_service = service
    try:
        yield
    finally:
        service.close()
        logger.info("Whisper model released")


app = FastAPI(
    title="autobot-stt",
    version=__version__,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness."},
        {"name": "sessions", "description": "Session lifecycle: create, finalize, delete."},
        {
            "name": "streaming",
            "description": "WebSocket live audio streaming and partial transcripts.",
        },
    ],
)


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Returns service availability. No authentication required.",
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


# REST routes keep Bearer auth at the router scope; the WebSocket stream route
# performs its own auth (query param + close code 4401) inside the handler.
v1_router = APIRouter(prefix="/v1")
v1_authed_router = APIRouter(dependencies=[Depends(verify_api_key)])
v1_authed_router.include_router(sessions_router)
v1_router.include_router(v1_authed_router)
v1_router.include_router(stream_router)
app.include_router(v1_router)


def run() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
