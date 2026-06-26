import uvicorn
from fastapi import FastAPI

from autobot_stt import __version__

app = FastAPI(title="autobot-stt", version=__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
