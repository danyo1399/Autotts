"""Unit tests for the GPU image build-time preload script.

The script lives outside the ``autobot_stt`` package (in ``scripts/``), so we
load it via ``importlib``. ``main()``'s network I/O is mocked by patching the
``WhisperModel`` symbol bound at module load; the post-download cache check is
exercised by creating or omitting the expected cache directory under a
temporary ``HF_HOME``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

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


def _make_cache_dir(tmp_path: Path, model: str) -> Path:
    cache_dir = tmp_path / "hub" / f"models--Systran--faster-whisper-{model}"
    cache_dir.mkdir(parents=True)
    return cache_dir


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


# --- main() ----------------------------------------------------------------
#
# main() downloads weights via WhisperModel(...) and then verifies the cache
# directory exists. We mock WhisperModel to avoid network I/O and control the
# verification outcome by creating (or not) the expected cache path under
# HF_HOME.


def test_main_invokes_whisper_with_cpu_and_int8_regardless_of_device_env(
    preload_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preload must always use CPU + int8 so ``docker build`` works without a GPU.

    Regression guard: if a future refactor reads ``WHISPER_DEVICE`` at preload
    time, the build would fail on GPU-less CI/hosts.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")  # must be ignored by preload
    _make_cache_dir(tmp_path, "small")

    with patch.object(preload_module, "WhisperModel") as mock_cls:
        preload_module.main()

    mock_cls.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_main_exits_nonzero_when_cache_dir_missing(
    preload_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the cache dir is absent after ``WhisperModel`` returns, fail the build."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # empty -> cache dir won't exist
    monkeypatch.setenv("WHISPER_MODEL", "small")

    with patch.object(preload_module, "WhisperModel"):
        with pytest.raises(SystemExit) as exc_info:
            preload_module.main()

    assert exc_info.value.code == 1


def test_main_completes_when_cache_dir_present(
    preload_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_MODEL", "small")
    _make_cache_dir(tmp_path, "small")

    with patch.object(preload_module, "WhisperModel"):
        preload_module.main()  # must not raise SystemExit


def test_main_defaults_whisper_model_to_small(
    preload_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without WHISPER_MODEL set, main() defaults to ``small`` (matches Dockerfile.gpu)."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    _make_cache_dir(tmp_path, "small")

    with patch.object(preload_module, "WhisperModel") as mock_cls:
        preload_module.main()

    mock_cls.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_main_passes_through_whisper_model_override(
    preload_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("WHISPER_MODEL", "base")
    _make_cache_dir(tmp_path, "base")

    with patch.object(preload_module, "WhisperModel") as mock_cls:
        preload_module.main()

    mock_cls.assert_called_once_with("base", device="cpu", compute_type="int8")
