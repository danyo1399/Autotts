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
└── services/
    ├── __init__.py
    ├── audio_decoder.py    # WebM/Opus -> 16 kHz mono PCM (ffmpeg)
    └── whisper_service.py  # faster-whisper wrapper + build_initial_prompt
tests/
├── fixtures/
│   └── sample.webm         # tiny WebM/Opus clip for decoder tests
├── test_health.py          # async health endpoint tests (httpx.AsyncClient)
├── test_config.py          # settings defaults, env loading, LogLevel validation
├── test_app.py             # OpenAPI schema, run(), lifespan wiring tests
├── test_audio_decoder.py   # audio decoder success, sample rate, error paths
└── test_whisper_service.py # build_initial_prompt + mocked transcribe tests
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
| `test_whisper_service.py` | `build_initial_prompt` logic, mocked `WhisperModel.load`/`transcribe`, beam size/VAD kwargs, empty/multi-dim input handling |

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
