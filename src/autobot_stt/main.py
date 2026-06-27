from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, Depends, FastAPI

from autobot_stt import __version__
from autobot_stt.dependencies.auth import verify_api_key
from autobot_stt.routes.sessions import router as sessions_router
from autobot_stt.stores.memory import InMemorySessionStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.session_store = InMemorySessionStore()
    yield


app = FastAPI(title="autobot-stt", version=__version__, lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


v1_router = APIRouter(prefix="/v1", dependencies=[Depends(verify_api_key)])
v1_router.include_router(sessions_router)
app.include_router(v1_router)


def run() -> None:
    uvicorn.run(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
