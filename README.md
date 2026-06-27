# autobot-stt

Speech-to-text (STT) API for Autobot voice input. Built with FastAPI and
designed to stream microphone audio to text via REST and WebSocket endpoints
(coming in later subtasks).

## Prerequisites

- **Python** 3.12 or newer
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **ffmpeg** — required for audio decoding (WebM/Opus → PCM). Install:
  - Debian/Ubuntu: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Verify: `ffmpeg -version`

## Setup

```bash
uv sync
cp .env.example .env   # optional — defaults apply
```

`uv sync` installs runtime and dev dependencies and generates `uv.lock`.

## Run

Start the API in development mode:

```bash
uv run uvicorn autobot_stt.main:app --reload
```

Or use the installed entry point (equivalent to above):

```bash
uv run autobot-stt
```

The server listens on `http://localhost:8000`.

## Project structure

```
src/autobot_stt/
├── __init__.py          # package version (single source of truth)
├── main.py              # FastAPI app factory, /health endpoint, uvicorn entry
├── config.py            # pydantic-settings config, cached singleton
├── services/
│   ├── __init__.py
│   └── audio_decoder.py # WebM/Opus -> 16 kHz mono PCM (ffmpeg)
├── models/
│   └── session.py       # ChatMessage, Comment, Session, request/response schemas
├── stores/
│   ├── base.py          # SessionStore Protocol
│   └── memory.py        # InMemorySessionStore (asyncio.Lock-guarded dict)
├── dependencies/
│   ├── auth.py          # verify_api_key (Bearer auth on /v1/*)
│   └── store.py         # get_session_store (app.state singleton)
└── routes/
    └── sessions.py      # POST /v1/sessions, DELETE /v1/sessions/{id}, POST /v1/sessions/{id}/finalize
tests/
├── fixtures/
│   └── sample.webm      # tiny WebM/Opus clip for decoder tests
├── conftest.py          # shared fixtures: client, store override, auth_headers
├── test_health.py       # async health endpoint tests (httpx.AsyncClient)
├── test_config.py       # settings defaults, env loading, LogLevel validation
├── test_app.py          # OpenAPI schema, metadata, run() entry point tests
├── test_audio_decoder.py # audio decoder success, sample rate, error paths
├── test_sessions.py     # POST/DELETE session REST contract
└── test_auth.py         # Bearer auth enforcement on /v1/*
```

## Test

```bash
uv run pytest
```

Tests use `pytest-asyncio` (async tests marked with `@pytest.mark.asyncio`)
and `httpx.AsyncClient` with `ASGITransport` for the FastAPI app.

| Test file | What it covers |
|-----------|----------------|
| `test_health.py` | Health endpoint returns 200, JSON content-type, app metadata |
| `test_config.py` | Settings defaults, env overrides, `get_settings()` caching, `LogLevel` validation |
| `test_app.py` | OpenAPI schema shape, `run()` delegates to uvicorn with correct args |
| `test_audio_decoder.py` | WebM/Opus decode to mono float32 PCM, sample-rate handling, error paths; skips when ffmpeg is absent |
| `test_sessions.py` | Session create/delete REST contract, persistence, defaults, 422 on bad input |
| `test_finalize.py` | LLM finalize endpoint: success, 400/404/502/503, session deletion (incl. preserved on OpenAI error), empty-text response, OpenAI payload, `cleanup_transcript` unit tests (whitespace strip, empty choices, None content, error propagation, api_key + timeout passthrough, empty-context placeholders) |
| `test_auth.py` | Bearer auth enforced on `/v1/*` when `STT_API_KEY` set; skipped when empty |

## Lint

```bash
uv run ruff check .
```

## CI

GitHub Actions runs lint and tests on every push and pull request to
`setup-autobot-tts-at1` and `main`:

1. `uv sync --frozen`
2. `uv run ruff check .`
3. `uv run pytest -v` (with `WHISPER_MODEL=base`, `WHISPER_DEVICE=cpu`)

Workflow definition: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Audio decoding

`autobot_stt.services.audio_decoder.decode_webm_opus_to_pcm` converts the
WebM/Opus byte chunks produced by a browser `MediaRecorder` into a 1-D
`numpy.float32` array normalized to `[-1, 1]` and resampled to 16 kHz mono.
The array is ready to hand to the Whisper service (subtask 5).

The function shells out to `ffmpeg` over stdin/stdout — no temp files. A
missing or misconfigured input raises `AudioDecodeError`; if ffmpeg itself is
not installed, `FileNotFoundError` propagates to the caller.

```python
from autobot_stt.services.audio_decoder import decode_webm_opus_to_pcm

pcm = decode_webm_opus_to_pcm(webm_bytes)  # -> np.ndarray, 16 kHz mono float32
```

## Health check

With the server running:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Sessions API

All session endpoints are mounted under `/v1` and require a Bearer token
when `STT_API_KEY` is set. With an empty/unset `STT_API_KEY`, auth is
skipped (local dev mode only).

### Create a session

```bash
curl -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $STT_API_KEY" \
  -d '{"draft_text":"hello","chat_history":[],"comments":[]}'
# HTTP/1.1 201 Created
# {"session_id":"<uuid4>"}
```

Request body fields (all optional, default to empty):

| Field | Type | Notes |
|-------|------|-------|
| `draft_text` | `string` | Default `""` |
| `chat_history` | `list[{role, content}]` | `role` is `"user"` or `"assistant"` |
| `comments` | `list[{author, body}]` | |

### Delete a session

```bash
curl -X DELETE http://localhost:8000/v1/sessions/<session_id> \
  -H "Authorization: Bearer $STT_API_KEY"
# HTTP/1.1 204 No Content  (404 if unknown)
```

### Finalize a session

Takes the session's raw Whisper transcript plus stored context (`draft_text`,
`chat_history`, `comments`), calls OpenAI `gpt-4o-mini` to fix transcription
errors, returns the cleaned text for appending to the user's draft, then
deletes the session.

```bash
curl -X POST http://localhost:8000/v1/sessions/<session_id>/finalize \
  -H "Authorization: Bearer $STT_API_KEY"
# HTTP/1.1 200 OK
# {"text":"cleaned text","raw_transcript":"original whisper output"}
```

Status codes:

| Status | Condition |
|--------|-----------|
| `200` | Cleaned text returned; session deleted |
| `400` | Session exists but `raw_transcript` is empty/whitespace |
| `401` | Missing or invalid Bearer token (when `STT_API_KEY` set) |
| `404` | Session not found |
| `502` | OpenAI call failed (rate limit, 5xx, network, invalid key); session is preserved for retry |
| `503` | `OPENAI_API_KEY` not configured |

The LLM prompt is built by `autobot_stt.services.llm_cleanup.cleanup_transcript`.
The model is hard-coded to `gpt-4o-mini` and is instructed to output **only**
the corrected spoken text (not the full draft), preserving technical terms
and meaning from chat history and comments.

### Authentication

- Header: `Authorization: Bearer <STT_API_KEY>`
- Applied to all `/v1/*` routes. `/health`, `/docs`, `/openapi.json` remain open.
- Missing or wrong token returns `401 Unauthorized`.

## Configuration

All configuration is via environment variables (or a `.env` file at the repo
root). See `.env.example` for the full list. None are required for this
subtask.

Key details:

- `Settings` in `config.py` uses `pydantic-settings` `BaseSettings` with
  `extra="ignore"` to ignore unrelated env vars.
- `get_settings()` is decorated with `@lru_cache` — settings are read once
  per process and reused.
- `LOG_LEVEL` is constrained to `"debug" | "info" | "warning" | "error" | "critical"`
  via a `LogLevel` `Literal` type. Invalid values raise `ValidationError` at
  construction time.

## Version management

The package version lives in a single source of truth:
`src/autobot_stt/__init__.py` defines `__version__`. It is consumed by:

- **hatch** (via `[tool.hatch.version]` in `pyproject.toml`) for package
  metadata
- **FastAPI** (via `main.py` importing `__version__`) for the API version
  field

To bump the version, edit `__version__` in `__init__.py` — no other file
needs changing.
