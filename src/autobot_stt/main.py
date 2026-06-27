import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, Depends, FastAPI

from autobot_stt import __version__
from autobot_stt.config import get_settings
from autobot_stt.dependencies.auth import verify_api_key
from autobot_stt.routes.sessions import router as sessions_router
from autobot_stt.services.whisper_service import WhisperService
from autobot_stt.stores.memory import InMemorySessionStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.session_store = InMemorySessionStore()

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


app = FastAPI(title="autobot-stt", version=__version__, lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


v1_router = APIRouter(prefix="/v1", dependencies=[Depends(verify_api_key)])
v1_router.include_router(sessions_router)
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
