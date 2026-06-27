"""Unit tests for the GPU image build-time preload script.

The script lives outside the ``autobot_stt`` package (in ``scripts/``), so we
load it via ``importlib``. ``main()`` itself is exercised at Docker build time
and depends on network I/O, so we only cover the pure path-construction helper
that the post-download verification check depends on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "preload_whisper_model.py"


@pytest.fixture(scope="module")
def preload_module():
    spec = importlib.util.spec_from_file_location("_preload_whisper_model", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_preload_whisper_model"] = module
    spec.loader.exec_module(module)
    return module


def test_expected_cache_dir_honors_hf_home(
    preload_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    cache_dir = preload_module._expected_cache_dir("small")
    assert cache_dir == tmp_path / "hub" / "models--Systran--faster-whisper-small"


def test_expected_cache_dir_falls_back_to_default(
    preload_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    cache_dir = preload_module._expected_cache_dir("base")
    assert cache_dir == (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-base"
    )
