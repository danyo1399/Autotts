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


@pytest.mark.parametrize(
    ("model_env", "expected_model", "set_cuda_device_env"),
    [
        # Regression guard: WHISPER_DEVICE=cuda must be ignored so docker build
        # works on GPU-less hosts (commit 4079b6a).
        pytest.param("small", "small", True, id="ignores-cuda-device-env"),
        pytest.param(None, "small", False, id="defaults-to-small"),
        pytest.param("base", "base", False, id="passes-model-override"),
    ],
)
def test_main_calls_whisper_with_cpu_int8(
    preload_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_env: str | None,
    expected_model: str,
    set_cuda_device_env: bool,
) -> None:
    """main() always invokes WhisperModel with CPU + int8, regardless of env vars."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    if model_env is None:
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
    else:
        monkeypatch.setenv("WHISPER_MODEL", model_env)
    if set_cuda_device_env:
        monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    _make_cache_dir(tmp_path, expected_model)

    with patch.object(preload_module, "WhisperModel") as mock_cls:
        preload_module.main()  # must not raise SystemExit

    mock_cls.assert_called_once_with(expected_model, device="cpu", compute_type="int8")


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
