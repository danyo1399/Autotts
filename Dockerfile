# CPU / development image for autobot-stt.
#
# Builds the FastAPI app with the Whisper ``base`` model on CPU. The model
# weights are downloaded on first start of the container (cached under
# ``/root/.cache/huggingface``); for a pre-baked CPU image, replicate the
# RUN step from ``Dockerfile.gpu``.
FROM python:3.12-slim

# Pin uv for reproducible builds; copy the binary from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/

# System dependencies:
#   - ffmpeg: required by audio_decoder.decode_webm_opus_to_pcm
#   - curl:   compose healthcheck and debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_NO_DEV=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=base \
    WHISPER_DEVICE=cpu

WORKDIR /app

# Layer 1: dependencies only (invalidates when lock/pyproject change).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Layer 2: application source.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "autobot_stt.main:app", "--host", "0.0.0.0", "--port", "8000"]
