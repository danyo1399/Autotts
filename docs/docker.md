# autobot-stt Docker packaging

Reference for the Docker image strategy, build constraints, and Compose setup.

## Two-image strategy

Two images serve different deployment contexts:

| Image | File | Base image | Whisper model | Device | Use case |
|-------|------|------------|---------------|--------|----------|
| CPU | `Dockerfile` | `python:3.12-slim` | `base` (~150 MB) | `cpu` | Local dev, CI |
| GPU | `Dockerfile.gpu` | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | `small` (~500 MB) | `cuda` | Production |

**Why:** The CPU image is lightweight and starts fast. The GPU image bundles a
larger model and the CUDA/cuDNN runtime for production accuracy. The
separation keeps dev iteration fast.

## cuDNN version pinning

`Dockerfile.gpu` uses `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.

**Why:** `ctranslate2` 4.8.0 (the runtime backend for `faster-whisper`) requires
cuDNN 9. The plain `nvidia/cuda:12.4.1-runtime-ubuntu22.04` image lacks cuDNN and
crashes with `Unable to load libcudnn_ops.so.9`.

**Tag nuance:** For CUDA 12.4+ the `cudnn-runtime` tag ships cuDNN 9.x. The older
`cudnn9-runtime` spelling only exists on CUDA <= 12.3.

## Build-time model preload

`Dockerfile.gpu` runs `scripts/preload_whisper_model.py` during the build to
download and cache the `small` model weights into the image layer.

**CPU preload for GPU build:** The script always uses `device="cpu"` /
`compute_type="int8"`, even when `WHISPER_DEVICE=cuda` is set. This ensures
`docker build` succeeds on hosts without a GPU. The cached weights are
device-agnostic and reused at runtime with CUDA.

**Cache verification:** After `WhisperModel()` returns, the script checks that
the expected Hugging Face cache directory exists. If not, it exits with code 1
to fail the build.

**Preload ordering constraint:** `WHISPER_DEVICE=cuda` must be set AFTER the
preload `RUN` step. Setting it before would cause any refactor that reads
`WHISPER_DEVICE` at preload time to break builds on GPU-less hosts or CI.

### Preload script API

```
scripts/preload_whisper_model.py
```

- Reads `WHISPER_MODEL` env var (defaults to `small`)
- Downloads weights via `faster_whisper.WhisperModel(model, device="cpu", compute_type="int8")`
- Exits 1 if cache directory is missing after download

## Healthcheck configuration

Both Compose services define a `curl`-based healthcheck. Key difference:

| Service | `start_period` | Rationale |
|---------|----------------|-----------|
| `stt` (CPU) | 120s | First start downloads `base` model (~150 MB) from Hugging Face Hub |
| `stt-gpu` (GPU) | 60s | Model is baked into image; only cuDNN init + CUDA context load |

## Compose GPU profile

The `stt-gpu` service uses a Compose profile (`profiles: ["gpu"]`) so it is not
started by bare `docker compose up`. Start it explicitly:

```bash
docker compose --profile gpu up --build stt-gpu
```

The profile reserves all available NVIDIA GPUs through the device reservation
block in `deploy.resources.reservations.devices`.

## CPU first-run download

The CPU image does NOT pre-download the Whisper model at build time. On first
container start, `faster-whisper` downloads the `base` model into
`/root/.cache/huggingface/hub` (~150 MB). Subsequent starts reuse the cached
weights. For a pre-baked CPU image, replicate the `RUN` step from
`Dockerfile.gpu`.

## .dockerignore

The `.dockerignore` excludes build-context noise from the Docker build.
Key patterns:

| Pattern | Why |
|---------|-----|
| `.git`, `.github` | Version control and CI not needed in image |
| `.venv`, `venv`, `env` | uv creates its own venv inside the image |
| `__pycache__`, `*.pyc` | Python bytecode |
| `.env`, `.env.*` | Credentials must be supplied at runtime; `!.env.example` keeps the example |
| `tests`, `docs` | Not needed at runtime |
| `*.md` | Documentation; `!README.md` keeps the readme |
| `.autobot_*` | Task runtime artifacts |
