"""Download faster-whisper weights into the Hugging Face cache.

Used as a build-time step in ``Dockerfile.gpu`` so the ``small`` model is baked
into the production image. Run with ``device="cpu"`` so no GPU is required
during ``docker build``; the cached weights are device-agnostic and reused at
runtime when the container runs with ``WHISPER_DEVICE=cuda``.
"""

from __future__ import annotations

import os
import sys

from faster_whisper import WhisperModel


def main() -> None:
    model = os.environ.get("WHISPER_MODEL", "small")
    WhisperModel(model, device="cpu", compute_type="int8")
    print(f"Preloaded whisper model: {model}", file=sys.stderr)


if __name__ == "__main__":
    main()
