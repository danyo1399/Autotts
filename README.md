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
├── __init__.py             # package version (single source of truth)
├── main.py                 # FastAPI app factory, lifespan, /health, uvicorn entry
├── config.py               # pydantic-settings config, cached singleton
├── services/
│   ├── __init__.py
│   ├── audio_decoder.py    # WebM/Opus -> 16 kHz mono PCM (ffmpeg)
│   ├── llm_cleanup.py      # OpenAI gpt-4o-mini transcript cleanup
│   └── whisper_service.py  # faster-whisper wrapper + build_initial_prompt
├── models/
│   └── session.py          # ChatMessage, Comment, Session, request/response schemas
├── stores/
│   ├── base.py             # SessionStore Protocol
│   └── memory.py           # InMemorySessionStore (asyncio.Lock-guarded dict)
├── dependencies/
│   ├── auth.py             # verify_api_key (Bearer auth on /v1/*)
│   └── store.py            # get_session_store (app.state singleton)
└── routes/
    └── sessions.py      # POST /v1/sessions, DELETE /v1/sessions/{id}, POST /v1/sessions/{id}/finalize
tests/
├── fixtures/
│   └── sample.webm              # tiny WebM/Opus clip for decoder tests
├── conftest.py                  # shared fixtures: client, store override, auth_headers
├── test_health.py               # async health endpoint tests (httpx.AsyncClient)
├── test_config.py               # settings defaults, env loading, LogLevel validation
├── test_app.py                  # OpenAPI schema, run(), lifespan wiring tests
├── test_audio_decoder.py        # audio decoder success, sample rate, error paths
├── test_sessions.py             # POST/DELETE session REST contract
├── test_auth.py                 # Bearer auth enforcement on /v1/*
├── test_whisper_service.py      # build_initial_prompt + mocked transcribe tests
├── test_preload_whisper_model.py # GPU build-time preload script (cache check, CPU/int8)
└── test_docker_config.py        # docker-compose / Dockerfile / .dockerignore integration
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
| `test_app.py` | OpenAPI schema shape, `run()` delegates to uvicorn with correct args, lifespan loads/releases `WhisperService` |
| `test_audio_decoder.py` | WebM/Opus decode to mono float32 PCM, sample-rate handling, error paths; skips when ffmpeg is absent |
| `test_sessions.py` | Session create/delete REST contract, persistence, defaults, 422 on bad input |
| `test_finalize.py` | LLM finalize endpoint: success, 400/404/502/503, session deletion (incl. preserved on OpenAI error), empty-text response, OpenAI payload, `cleanup_transcript` unit tests (whitespace strip, empty choices, None content, error propagation, api_key + timeout passthrough, empty-context placeholders) |
| `test_auth.py` | Bearer auth enforced on `/v1/*` when `STT_API_KEY` set; skipped when empty |
| `test_whisper_service.py` | `build_initial_prompt` logic, mocked `WhisperModel.load`/`transcribe`, beam size/VAD kwargs, empty/multi-dim input handling |
| `test_preload_whisper_model.py` | GPU build-time preload: `_expected_cache_dir()` paths, `main()` CPU/int8 hardcoding, cache-miss exit code |
| `test_docker_config.py` | `docker-compose.yml` shape, `Dockerfile` / `Dockerfile.gpu` defaults and preload ordering, `.dockerignore` patterns, real `docker compose config` validation |

## Lint

```bash
uv run ruff check .
```

## Docker

Two images are provided: a CPU image for local development and a GPU image
for production with the Whisper `small` model baked in.

### Prerequisites

- Docker Engine with Docker Compose v2
- For the GPU image: an NVIDIA driver and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Environment variables

Both images read runtime configuration from a `.env` file (Compose) or
`--env-file` flag (`docker run`). Create one from the example:

```bash
cp .env.example .env
```

| Variable | Required | Notes |
|----------|----------|-------|
| `STT_API_KEY` | Production: yes | Bearer token for `/v1/*`; empty skips auth (local dev only) |
| `OPENAI_API_KEY` | When using OpenAI cleanup (subtask 7) | Post-processing |
| `WHISPER_MODEL` | No | Set by image defaults (`base` CPU, `small` GPU); override via env |
| `WHISPER_DEVICE` | No | Set by image defaults (`cpu` / `cuda`) |
| `LOG_LEVEL` | No | Default `info` |

### Local development (CPU)

```bash
cp .env.example .env
docker compose up --build
```

Health check (in another terminal):

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

The first start downloads the Whisper `base` model (~150 MB) into the
container's Hugging Face cache; subsequent starts reuse it.

### Production GPU image

Build the image (no GPU required during `docker build`):

```bash
docker build -f Dockerfile.gpu -t autobot-stt:gpu .
```

Run with GPU access:

```bash
docker run --gpus all -p 8000:8000 --env-file .env autobot-stt:gpu
```

The `small` model (~500 MB) is baked into the image during build, so runtime
startup does not hit the network. Verify the cache after build:

```bash
docker run --rm autobot-stt:gpu \
  find /root/.cache/huggingface -name '*faster-whisper-small*' | head
```

### GPU via Compose profile

```bash
docker compose --profile gpu up --build stt-gpu
```

### Notes

- The CPU and GPU services both bind `8000:8000`; only run one at a time.
- CI does not currently build Docker images.

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

## Whisper STT

`autobot_stt.services.whisper_service.WhisperService` wraps
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local speech
transcription. The model is loaded once in the FastAPI lifespan startup and
stored on `app.state.whisper_service` so subsequent requests can reuse it.

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | faster-whisper model size (`base` for dev/CI, `small` for production) |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda`. `compute_type` is derived (`int8` on CPU, `float16` on CUDA) |

**First-run download.** faster-whisper pulls model weights from the Hugging Face
Hub on first use and caches them under `~/.cache/huggingface/hub`. Approximate
sizes: ~150 MB for `base`, ~500 MB for `small`.

**Transcription defaults.** `WhisperService.transcribe()` calls
`WhisperModel.transcribe()` with `beam_size=5` and `vad_filter=True`, then
concatenates the segment texts. Empty input returns `""` without invoking the
model.

**Initial prompt.** `build_initial_prompt(draft_text, chat_history)` assembles
a context string from the user's current draft plus the last few chat messages
(capped at ~800 characters / ~224 tokens, preserving the most recent tail). The
WebSocket route (subtask 6) passes this string as Whisper's `initial_prompt` to
bias decoding toward in-domain vocabulary.

```python
from autobot_stt.services.whisper_service import build_initial_prompt

prompt = build_initial_prompt(
    draft_text="meeting notes",
    chat_history=[{"role": "user", "content": "discuss Q3 roadmap"}],
)
text = app.state.whisper_service.transcribe(pcm_array, initial_prompt=prompt or None)
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
