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
