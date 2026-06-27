"""Download faster-whisper weights into the Hugging Face cache.

Used as a build-time step in ``Dockerfile.gpu`` so the ``small`` model is baked
into the production image. Run with ``device="cpu"`` so no GPU is required
during ``docker build``; the cached weights are device-agnostic and reused at
runtime when the container runs with ``WHISPER_DEVICE=cuda``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def _expected_cache_dir(model: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    # faster-whisper publishes weights as Systran/faster-whisper-{model} on HF Hub.
    cache_name = f"models--Systran--faster-whisper-{model}"
    return hf_home / "hub" / cache_name


def main() -> None:
    model = os.environ.get("WHISPER_MODEL", "small")
    # Always use CPU at build time; WHISPER_DEVICE is a runtime setting only.
    WhisperModel(model, device="cpu", compute_type="int8")

    cache_dir = _expected_cache_dir(model)
    if not cache_dir.is_dir():
        print(
            f"ERROR: Whisper model {model!r} was not cached at {cache_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Preloaded whisper model: {model} -> {cache_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
