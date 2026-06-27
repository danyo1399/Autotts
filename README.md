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
└── services/
    ├── __init__.py
    └── audio_decoder.py # WebM/Opus -> 16 kHz mono PCM (ffmpeg)
tests/
├── fixtures/
│   └── sample.webm      # tiny WebM/Opus clip for decoder tests
├── test_health.py       # async health endpoint tests (httpx.AsyncClient)
├── test_config.py       # settings defaults, env loading, LogLevel validation
├── test_app.py          # OpenAPI schema, metadata, run() entry point tests
└── test_audio_decoder.py # audio decoder success, sample rate, error paths
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

## Lint

```bash
uv run ruff check .
```

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
