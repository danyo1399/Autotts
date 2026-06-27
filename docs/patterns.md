# autobot-stt Project Patterns

Reference document for project conventions and design decisions.

## Version management

**Single source of truth.** The package version is defined in
`src/autobot_stt/__init__.py` as `__version__`. Both hatch (build backend) and
FastAPI (app metadata) consume it from there.

**Why:** Prevents drift between package metadata and the API version reported
at runtime.

**How:** `pyproject.toml` sets `dynamic = ["version"]` with
`[tool.hatch.version] path = "src/autobot_stt/__init__.py"`. To bump, edit
`__init__.py` only.

## Settings / config caching

`get_settings()` in `config.py` is decorated with `@lru_cache` so
`Settings()` is constructed exactly once per process.

**Why:** Every call reads and parses `.env` file. Uncached, every request that
uses `Depends(get_settings)` re-parses env vars — wasteful and risks
inconsistent state if env changes mid-process.

**How:** The `get_settings` function acts as a cached factory. Its cache can
be cleared with `get_settings.cache_clear()` during testing. Add new settings
by adding a field to the `Settings` class and a matching entry in
`.env.example`.

## Async testing with FastAPI

Tests use `httpx.AsyncClient` with `ASGITransport` to talk to the FastAPI app
without a live server.

**Why:** The sync `starlette.testclient.TestClient` is deprecated and emits
`StarletteDeprecationWarning`. Async testing also aligns with the project's
async-first design (WebSocket, streaming).

**How:**
- `pytest-asyncio` is a dev dependency
- `pyproject.toml` sets `asyncio_mode = "auto"`
- Async tests are marked with `@pytest.mark.asyncio`
- HTTP calls use `httpx.AsyncClient` with `ASGITransport(app=app)`

```python
@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
```

## Constrained string values with Literal

For settings that accept a fixed set of string values, use `typing.Literal`
instead of bare `str`.

**Why:** A bare `str` field accepts any value — typos or invalid config pass
silently at startup, only failing at first use. `Literal` validates at
pydantic construction time, catching errors early.

**How:** Define a type alias and use it as the field annotation:

```python
from typing import Literal

LogLevel = Literal["debug", "info", "warning", "error", "critical"]

class Settings(BaseSettings):
    log_level: LogLevel = "info"
```

## Env variable testing with monkeypatch

Settings tests use pytest's built-in `monkeypatch` fixture to set env vars
before constructing `Settings()`.

**Why:** Avoids pollution across tests (monkeypatch restores automatically)
and needs no external files.

**How:**

```python
def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "small")
    settings = Settings()
    assert settings.whisper_model == "small"
```

## Entry point: `run()`

The `main.py` module exposes a `run()` function that invokes `uvicorn.run()`
with `reload=True` for development convenience.

**Why:** Allows `uv run autobot-stt` as a shorthand for the full uvicorn
command. The entry point is registered in `pyproject.toml` under
`[project.scripts]`.

```python
def run() -> None:
    uvicorn.run(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
```

## Session store: Protocol + dependency injection

Session storage is defined by a `SessionStore` `Protocol` in
`stores/base.py` and implemented by `InMemorySessionStore` in
`stores/memory.py`. Routes receive the store through FastAPI's `Depends()`.

**Why:** The `Protocol` enables structural subtyping — tests can provide
fakes without inheriting. The dependency boundary means the store can be
swapped (Redis, SQL, …) by changing only the lifespan setup in `main.py`,
with no route changes.

**How:**

- `main.py` lifespan instantiates one `InMemorySessionStore` and stores it
  on `app.state.session_store`.
- `dependencies/store.py` exposes `get_session_store(request)` which reads
  from `request.app.state.session_store`.
- Routes declare `store: SessionStore = Depends(get_session_store)`.
- Tests override the dependency via
  `app.dependency_overrides[get_session_store] = lambda: fake_store`.

`InMemorySessionStore` uses an `asyncio.Lock` to serialize dict mutation so
concurrent `create`/`delete` calls cannot corrupt the dict.

## Router-scoped Bearer auth

Authentication is enforced on `/v1/*` via a `verify_api_key` FastAPI
dependency attached to the v1 router, not via HTTP middleware.

**Why:** `Depends()` is the idiomatic FastAPI pattern for request
validation. Scoping it to the router (instead of the root app) automatically
protects `/v1/*` while leaving `/health`, `/docs`, `/redoc`, and
`/openapi.json` open.

**How:**

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.stt_api_key:
        return  # dev mode — auth skipped
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized")
    if credentials.credentials != settings.stt_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
```

Notes:

- `HTTPBearer(auto_error=False)` lets us return `401` (not `403`) with a
  consistent body when the header is missing.
- The error detail is always `"Unauthorized"` — we do not leak whether the
  header was missing, the scheme was wrong, or the key differed.
- When `STT_API_KEY` is empty/unset, auth is bypassed entirely. This is
  local-dev convenience only; production must set a non-empty key.
- Auth tests mutate `STT_API_KEY` via `monkeypatch.setenv` and clear the
  `get_settings` cache (`get_settings.cache_clear()`) so the new value is
  picked up. The `conftest._clear_settings_cache` autouse fixture clears
  the cache before and after every test to prevent leakage.

## WebSocket auth (separate from REST Bearer)

The WebSocket stream route (`routes/stream.py`) cannot share the router-scoped
`verify_api_key` dependency: WS clients authenticate via `?token=` query
parameter (or `Authorization` header in the handshake), and failure must
surface as a close frame (code 4401), not an HTTP 401.

**Why:** REST auth raises `HTTPException(401)` from a `Depends` chain that
runs after the request has been fully received. A WebSocket handshake has to
choose between `accept()` (proceed) and `close(code=...)` (reject) before
any application-level messages flow, so the handler must run its own check
and close with a WS-specific code.

**How:**

- `v1_router` is split: REST routes (`sessions_router`) keep
  `dependencies=[Depends(verify_api_key)]`; the stream route is mounted on
  `v1_router` without router-level auth.
- `dependencies/auth.py::check_ws_api_key(websocket, token, settings) -> bool`
  is called as a plain function (not via `Depends`) at the top of the
  handler. Returning `False` triggers `await websocket.close(code=4401)`.
- Unknown sessions close with code `4404`.
- When `STT_API_KEY` is empty, `check_ws_api_key` returns `True` and the
  handler proceeds (dev mode parity with REST).

## Blocking work off the event loop (`asyncio.to_thread`)

`decode_webm_opus_to_pcm` shells out to ffmpeg (subprocess) and
`WhisperService.transcribe` runs CPU/GPU inference. Both are synchronous
and would block the event loop, starving other WebSocket connections.

**Why:** FastAPI/Starlette runs the handler on a single event loop; a
blocking call inside a handler stalls every concurrent request. The decode
and transcribe calls are offloaded to a worker thread via
`asyncio.to_thread`, yielding control back to the loop between operations.

**How:** The stream handler wraps the calls:

```python
pcm = await asyncio.to_thread(decode_webm_opus_to_pcm, batch)
async with app.state.whisper_lock:
    text = await asyncio.to_thread(whisper.transcribe, pcm, initial_prompt)
```

`faster-whisper`'s model instance is not safe for concurrent transcribe
calls, so an `asyncio.Lock` (`app.state.whisper_lock`, created in lifespan)
serializes access across connections.

## WebSocket streaming flush logic

The stream handler (`routes/stream.py`) buffers incoming binary WebM
frames and decides when to decode + transcribe using two thresholds and
a silence timeout.

**Why:** Sending every 50 ms Opus frame to ffmpeg+Whisper would pin the
CPU without improving accuracy. Whisper performs best on a few seconds of
audio at once. The thresholds batch small frames into ~2 s windows while
still flushing promptly when the speaker pauses.

**How:**

- `STREAM_MIN_BYTES_FOR_DECODE = 1024` — byte threshold below which
  `flush(force=False)` is a no-op. Guards against tiny initial frames.
- `STREAM_FLUSH_SAMPLES = STREAM_CHUNK_SECONDS * PCM_SAMPLE_RATE` —
  sample threshold that triggers transcription after decode. Default
  `2 * 16000 = 32000` samples ≈ 2 s of audio.
- `STREAM_SILENCE_TIMEOUT_SECONDS = 1.5` — `asyncio.wait_for` wraps
  `websocket.receive()`; on `TimeoutError` the handler calls
  `flush(force=True)` to drain whatever has accumulated.

`flush(force=...)` decodes the buffered WebM bytes to PCM, then either
calls `_transcribe_and_emit` (when the threshold is met or `force=True`)
or returns early to wait for more audio. The WebM buffer is cleared as
soon as a successful decode is consumed by the transcribe step.

`_build_initial_prompt(session)` is computed **once per connection** from
`session.draft_text` + `session.chat_history` via
`services.whisper_service.build_initial_prompt`, then reused for every
transcribe call on that connection.

The cumulative `session.raw_transcript` is updated inside
`_transcribe_and_emit`: new text is appended with a separating space, the
same text is pushed onto `session.partial_transcripts`, and the entire
`raw_transcript` is sent in each `partial_transcript` event (`is_final`
is always `false` in this subtask — finalization is subtask 7).

## Pydantic schemas vs. domain models

HTTP request/response payloads use thin Pydantic `BaseModel` classes
(`models/session.py`). The internal `Session` model is also a `BaseModel`
but carries runtime-only fields (`raw_transcript`, `partial_transcripts`,
`created_at`) that are never accepted from the client.

**Why:** Keeping the REST contract thin prevents clients from setting
server-managed fields. The domain model can grow later (audio transcripts,
LLM-finalized text) without changing the public API.

**How:** Route handlers translate `CreateSessionRequest` → `Session` by
generating `id` (UUID4) and `created_at` (UTC now), then call
`store.create(session)` and return `CreateSessionResponse(session_id=...)`.
