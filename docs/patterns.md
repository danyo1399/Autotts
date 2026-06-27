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
