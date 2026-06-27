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
WhisperDevice = Literal["cpu", "cuda"]

class Settings(BaseSettings):
    log_level: LogLevel = "info"
    whisper_device: WhisperDevice = "cpu"
```

`WhisperDevice` follows the same pattern as `LogLevel`: a typed alias that
catches invalid values (e.g. `"cuda:0"`, `"gpu"`) at construction time rather
than at model-load time.

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
for development convenience.

**Why:** Allows `uv run autobot-stt` as a shorthand for the full uvicorn
command. The entry point is registered in `pyproject.toml` under
`[project.scripts]`. `reload=True` is intentionally omitted from `run()` —
production safety takes precedence; use `uv run uvicorn ... --reload` explicitly
when hot-reload is needed during development.

```python
def run() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run(
        "autobot_stt.main:app",
        host="0.0.0.0",
        port=8000,
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

## External API error handling: OpenAI

Calls to external APIs (currently the OpenAI transcript-cleanup service) use a
consistent error-handling pattern:

**Map SDK exceptions to HTTP 502.** OpenAI raises `openai.OpenAIError` for
rate limits, 5xx, network failures, and invalid keys. The route catches it and
returns `502 Bad Gateway` with a descriptive message. The session is **not**
deleted on error — the caller can retry.

**Timeout configuration.** The OpenAI client is created with an explicit
`timeout` kwarg (currently 30 s in `_TIMEOUT_SECONDS`). This prevents a hung
API call from blocking the server indefinitely.

**Async context manager.** The client is used via `async with
AsyncOpenAI(...) as client` to guarantee proper connection cleanup.

**Why:** Without a timeout, a stuck connection holds the worker forever.
Without the 502→preserve pattern, a transient OpenAI failure destroys the
user's session and transcript — they'd have to re-record.

**How:**
```python
_TIMEOUT_SECONDS = 30.0

async with AsyncOpenAI(api_key=api_key, timeout=_TIMEOUT_SECONDS) as client:
    response = await client.chat.completions.create(...)
```

Route layer:
```python
try:
    cleaned = await cleanup_transcript(session, api_key=settings.openai_api_key)
except OpenAIError as exc:
    raise HTTPException(status_code=502, detail=f"Cleanup failed: {exc}") from exc
# Only delete session on success
await store.delete(session_id)
```

## Transcript snapshot on finalize (defensive copy)

The finalize endpoint copies the session before passing it to the OpenAI
cleanup call:

```python
cleanup_session = session.model_copy(update={"raw_transcript": raw_transcript})
try:
    cleaned = await cleanup_transcript(cleanup_session, api_key=...)
```

**Why:** While the OpenAI call is in flight, a concurrent WebSocket streaming
connection could mutate `session.raw_transcript`. Without the copy, the
transcript sent to cleanup would differ from the one returned to the client in
the response. The copy ensures both `raw_transcript` in the response and the
value seen by `cleanup_transcript` reflect the same point-in-time snapshot.

**How:** `session.model_copy(update={"raw_transcript": raw_transcript})` creates
a shallow copy of the `Session` pydantic model with the transcript field pinned
to the string that was read before the copy. The `raw_transcript` binding on
line 87 already holds an immutable `str`; the `model_copy` guards against other
attribute changes on the session (none today, but future-proof).

The session is deleted only on success (line 99), so the caller can retry on
OpenAI failures without losing the accumulated transcript.

## Guarding against empty API responses

When calling LLM APIs, handle two edge cases that the SDK types permit:

1. **Empty choices list** — the API returns success but no completions.
2. **`None` message content** — the choice exists but `message.content` is
   `None` (e.g. a tool-call-only response or streaming edge case).

Both cases return `""` rather than crashing.

**Why:** The OpenAI SDK types allow these values. Without explicit guards a
`None` content value would produce `"None"` (stringified) in the response.

**How:**
```python
if not response.choices:
    return ""
content = response.choices[0].message.content or ""
return content.strip()
```

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
    if not _keys_match(credentials.credentials, settings.stt_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
```

Notes:

- `HTTPBearer(auto_error=False)` lets us return `401` (not `403`) with a
  consistent body when the header is missing.
- The error detail is always `"Unauthorized"` — we do not leak whether the
  header was missing, the scheme was wrong, or the key differed.
- Both REST and WebSocket auth use the `_keys_match` helper for constant-time
  comparison (see the [WebSocket auth](#websocket-auth-separate-from-rest-bearer)
  section below for the implementation).
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
- Both the REST and WS paths compare keys with `hmac.compare_digest` (via
  the `_keys_match` helper) to avoid a timing side-channel on API-key guessing.

### `_keys_match` — constant-time comparison

The `_keys_match` helper wraps `hmac.compare_digest` with a `None` guard,
since `compare_digest` requires both arguments to be the same type:

```python
def _keys_match(provided: str | None, expected: str) -> bool:
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
```

The `None` check is essential: a missing credential must short-circuit rather
than passing `None` to `.encode()`. Both sides are UTF-8 encoded so Unicode
keys compare correctly.

### `_extract_ws_token` — credential source precedence

WebSocket clients authenticate via `?token=` query param **or**
`Authorization: Bearer` header. Query param wins when both are present:

```python
def _extract_ws_token(websocket: WebSocket, token: str | None) -> str | None:
    if token:
        return token
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[len("bearer "):].strip()
    return None
```

The `"bearer "` prefix length is expressed as `len("bearer ")` rather than
a magic number `7`, so the intent is self-documenting and survives scheme
changes. The scheme comparison is case-insensitive (`startswith` on `.lower()`).

### Query-string token exposure (deployment caveat)

The `?token=<STT_API_KEY>` contract is locked by the spec, but a query
string is recorded alongside the request line in many default access logs
(uvicorn's access log, nginx/ALB access logs, browser history). Mitigations:

- Prefer the `Authorization: Bearer ...` header when the client can set
  headers on the WS handshake (the server accepts both).
- In production, suppress or filter query strings from access logs (e.g.
  `--no-access-log`, or a log formatter that redacts `?token=`).

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

**Multi-worker caveat.** `asyncio.Lock` is per-process. Deploying behind
`uvicorn --workers N` (or any pre-fork model) loads the Whisper model once
per worker, each with its own lock. That is correct — the model is not
shared across processes, so no cross-worker lock is needed — but operators
should not expect `app.state.whisper_lock` to serialize work across
workers. Each worker also holds its own `session_store`; subtask 7 will
need to externalize state before horizontal scaling is meaningful.

### Whisper service/lock DI accessors

The stream handler receives `WhisperService` and `asyncio.Lock` via
`Depends()` helpers in `dependencies/store.py`, not by reaching into
`request.app.state` inside the handler:

```python
def get_whisper_service(conn: HTTPConnection) -> WhisperService:
    service = getattr(conn.app.state, "whisper_service", None)
    if service is None:
        raise RuntimeError("Whisper service is not initialized on app.state")
    return service

def get_whisper_lock(conn: HTTPConnection) -> asyncio.Lock:
    lock = getattr(conn.app.state, "whisper_lock", None)
    if lock is None:
        raise RuntimeError("Whisper lock is not initialized on app.state")
    return lock
```

**Why:** Keeps the stream handler focused on protocol logic. The `RuntimeError`
surfaces missing-lifespan-wiring early with a clear message, rather than an
opaque `AttributeError` at the first transcribe call. See also
[`HTTPConnection` for WS-safe DI](#httpconnection-for-websocket-safe-dependency-injection)
below.

### Transcript mutation within `whisper_lock` — emit outside lock

The `_transcribe_and_emit` function holds `whisper_lock` across **both** the
`whisper.transcribe()` call **and** the `session.raw_transcript`
read-modify-write, not just the model call:

```python
async with whisper_lock:
    if await store.get(session.id) is None:
        return False
    try:
        text = await asyncio.to_thread(whisper.transcribe, pcm, initial_prompt)
    except Exception:
        transcribe_error = True
    else:
        if text:
            if await store.get(session.id) is None:
                return False
            if session.raw_transcript:
                session.raw_transcript += " "
            session.raw_transcript += text
            session.partial_transcripts.append(text)
            cumulative = session.raw_transcript

# Send outside the lock so a slow client cannot stall other transcriptions.
if transcribe_error:
    await websocket.send_json({"type": "error", "message": "Transcription failed"})
elif cumulative is not None:
    await websocket.send_json({"type": "partial_transcript", "text": cumulative, ...})
```

**Why (lock scope):** If the transcribe call were inside the lock but the
transcript update were after releasing it, two concurrent WebSocket connections
to the same session could interleave their appends. One connection's output
would be overwritten. Locking the entire critical section makes the mutation
unambiguously safe — no `await` in the mutation today keeps it atomic under the
GIL, but the lock guarantees safety even if a future `await` is inserted
between the read and write.

**Why (emit outside):** If `websocket.send_json` were inside the lock, a slow
client that reads the socket slowly would hold the lock open, stalling other
sessions waiting to transcribe. Emitting outside the lock means lock duration is
bounded by model inference + dict append (~10-100 ms), not by network I/O.

**Session-deletion detection.** Before transcribing and again before mutating,
the handler checks `store.get(session.id)` — if a concurrent `finalize` or
`DELETE` removed the session, the handler returns `False` and the stream loop
stops. Without this, streaming could continue on a deleted session, appending
transcript text to an orphan object that will never be finalized.

**Error recovery.** Transcribe failures (`Exception`) are caught, logged, and
reported as an error event — the connection stays open and subsequent chunks
can still be processed. Only the session-gone case stops the stream.

## `HTTPConnection` for WebSocket-safe dependency injection

Dependency accessors used by both HTTP and WebSocket routes must type
their parameter as `HTTPConnection` (from Starlette), not `Request`.

**Why:** WebSocket connections are not `Request` objects. A dependency
declared with `def get_x(request: Request) -> X` raises
`AttributeError` when FastAPI tries to inject it into a WebSocket route
handler. `HTTPConnection` is the common parent of both `Request` and
`WebSocket`, so it resolves correctly for both.

**How:**

```python
from starlette.requests import HTTPConnection

def get_session_store(conn: HTTPConnection) -> SessionStore:
    return conn.app.state.session_store
```

The `dependencies/store.py` module uses this pattern for all three
injectors (`get_session_store`, `get_whisper_service`,
`get_whisper_lock`).

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

**Partial decode deferral.** An `AudioDecodeError` during a non-forced
flush does not close the connection — the handler logs at `DEBUG` level
and returns, waiting for more bytes. This is necessary because
`MediaRecorder` fragments may not be valid WebM independently; they only
form a valid stream once enough chunks have accumulated. On a forced
flush (silence timeout or disconnect), the same error is reported to the
client as `{"type": "error", "message": "Failed to decode audio"}`.
`FileNotFoundError` (ffmpeg missing) is always reported immediately.

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
