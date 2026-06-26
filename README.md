# autobot-stt

Speech-to-text (STT) API for Autobot voice input. Built with FastAPI and
designed to stream microphone audio to text via REST and WebSocket endpoints
(coming in later subtasks).

This repository currently contains the **project scaffold** only: a working
`/health` endpoint, dependency management with [uv](https://docs.astral.sh/uv/),
linting with [ruff](https://docs.astral.sh/ruff/), and tests with
[pytest](https://docs.pytest.org/). Speech-to-text, LLM post-processing,
WebSocket streaming, authentication, and Docker are out of scope for this
subtask.

## Prerequisites

- **Python** 3.12 or newer
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **ffmpeg** — required at runtime once audio decoding lands (subtask 4+);
  not needed for the health check in this subtask

## Setup

```bash
uv sync
cp .env.example .env   # optional for this subtask
```

`uv sync` installs runtime and dev dependencies and generates `uv.lock`.

## Run

Start the API in development mode:

```bash
uv run uvicorn autobot_stt.main:app --reload
```

Or use the installed entry point:

```bash
uv run autobot-stt
```

The server listens on `http://localhost:8000`.

## Test

```bash
uv run pytest
```

## Lint

```bash
uv run ruff check .
```

## Health check

With the server running:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Configuration

All configuration is via environment variables (or a `.env` file at the repo
root). See `.env.example` for the full list. None are required for this
subtask.
