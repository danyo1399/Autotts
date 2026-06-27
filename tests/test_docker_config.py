"""Integration tests for the Docker packaging surface introduced in AT-9.

Validates the structure of ``docker-compose.yml``, the env defaults baked into
both ``Dockerfile`` and ``Dockerfile.gpu``, and the ``.dockerignore`` rules.
A real ``docker compose config`` run is included at the end so the YAML is
validated by Compose itself (not just by ``yaml.safe_load``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- docker-compose.yml ----------------------------------------------------


@pytest.fixture(scope="module")
def compose_config() -> dict:
    with (REPO_ROOT / "docker-compose.yml").open() as fh:
        loaded = yaml.safe_load(fh)
    assert isinstance(loaded, dict)
    return loaded


def test_compose_defines_cpu_and_gpu_services(compose_config: dict) -> None:
    services = compose_config.get("services", {})
    assert "stt" in services
    assert "stt-gpu" in services


def test_cpu_service_uses_dockerfile_and_exposes_8000(compose_config: dict) -> None:
    stt = compose_config["services"]["stt"]
    assert stt["build"]["dockerfile"] == "Dockerfile"
    assert stt["ports"] == ["8000:8000"]


def test_gpu_service_uses_dockerfile_gpu_and_gpu_profile(compose_config: dict) -> None:
    stt_gpu = compose_config["services"]["stt-gpu"]
    assert stt_gpu["build"]["dockerfile"] == "Dockerfile.gpu"
    assert stt_gpu["profiles"] == ["gpu"]
    assert stt_gpu["ports"] == ["8000:8000"]


def test_gpu_service_reserves_nvidia_gpu(compose_config: dict) -> None:
    devices = (
        compose_config["services"]["stt-gpu"]["deploy"]["resources"]["reservations"]["devices"]
    )
    assert len(devices) == 1
    assert devices[0]["driver"] == "nvidia"
    assert devices[0]["count"] == "all"
    assert devices[0]["capabilities"] == ["gpu"]


def test_gpu_service_overrides_whisper_to_small_cuda(compose_config: dict) -> None:
    env = compose_config["services"]["stt-gpu"]["environment"]
    assert env["WHISPER_MODEL"] == "small"
    assert env["WHISPER_DEVICE"] == "cuda"


def test_env_file_is_optional_on_both_services(compose_config: dict) -> None:
    """Both services must start without a ``.env`` file (Dockerfile defaults apply)."""
    expected = [{"path": ".env", "required": False}]
    assert compose_config["services"]["stt"]["env_file"] == expected
    assert compose_config["services"]["stt-gpu"]["env_file"] == expected


def test_both_services_have_curl_healthcheck_on_8000(compose_config: dict) -> None:
    for name in ("stt", "stt-gpu"):
        healthcheck = compose_config["services"][name]["healthcheck"]
        cmd_parts = healthcheck["test"]
        # Format: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
        assert "curl" in cmd_parts
        assert any("8000/health" in part for part in cmd_parts)
        assert healthcheck["interval"] == "30s"
        assert healthcheck["timeout"] == "10s"
        assert healthcheck["retries"] == 3


def test_cpu_healthcheck_start_period_accommodates_first_run_download(
    compose_config: dict
) -> None:
    """CPU ``start_period`` must be long enough for the first-run base model download."""
    start_period = compose_config["services"]["stt"]["healthcheck"]["start_period"]
    # ``base`` is ~150 MB; the plan locked 120s.
    assert start_period == "120s"


# --- Dockerfile (CPU / dev) ------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text()


def test_cpu_dockerfile_uses_python_3_12_slim_base() -> None:
    assert "FROM python:3.12-slim" in _read(REPO_ROOT / "Dockerfile")


def test_cpu_dockerfile_defaults_to_base_cpu() -> None:
    content = _read(REPO_ROOT / "Dockerfile")
    assert "WHISPER_MODEL=base" in content
    assert "WHISPER_DEVICE=cpu" in content


def test_cpu_dockerfile_installs_ffmpeg_and_uv() -> None:
    content = _read(REPO_ROOT / "Dockerfile")
    assert "ffmpeg" in content
    assert "ghcr.io/astral-sh/uv:" in content


def test_cpu_dockerfile_runs_uv_sync_frozen() -> None:
    content = _read(REPO_ROOT / "Dockerfile")
    assert "uv sync --frozen" in content


def test_cpu_dockerfile_exposes_8000_and_starts_uvicorn() -> None:
    content = _read(REPO_ROOT / "Dockerfile")
    assert "EXPOSE 8000" in content
    assert 'CMD ["uv", "run", "uvicorn", "autobot_stt.main:app"' in content


# --- Dockerfile.gpu (production) -------------------------------------------


def test_gpu_dockerfile_uses_cudnn_runtime_base() -> None:
    """ctranslate2 4.8.0 requires cuDNN 9; the ``cudnn-runtime`` tag ships it."""
    content = _read(REPO_ROOT / "Dockerfile.gpu")
    assert "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04" in content


def test_gpu_dockerfile_bakes_small_model_via_preload_script() -> None:
    content = _read(REPO_ROOT / "Dockerfile.gpu")
    assert "WHISPER_MODEL=small" in content
    assert "scripts/preload_whisper_model.py" in content


def test_gpu_dockerfile_defers_cuda_device_until_after_preload() -> None:
    """``WHISPER_DEVICE=cuda`` must come AFTER the preload step.

    Regression guard for the issue fixed in commit 4079b6a: setting
    ``WHISPER_DEVICE=cuda`` before preload would cause a future refactor that
    reads the env var to break builds on hosts without a GPU.
    """
    content = _read(REPO_ROOT / "Dockerfile.gpu")
    preload_idx = content.index("scripts/preload_whisper_model.py")
    cuda_idx = content.index("WHISPER_DEVICE=cuda")
    assert cuda_idx > preload_idx


def test_gpu_dockerfile_sets_hf_home_for_baked_cache() -> None:
    content = _read(REPO_ROOT / "Dockerfile.gpu")
    assert "HF_HOME=/root/.cache/huggingface" in content


def test_gpu_dockerfile_installs_python_3_12_via_uv() -> None:
    """CUDA base image does not ship Python; uv must install it explicitly."""
    content = _read(REPO_ROOT / "Dockerfile.gpu")
    assert "uv python install 3.12" in content
    assert "UV_PYTHON=3.12" in content


def test_gpu_dockerfile_exposes_8000_and_starts_uvicorn() -> None:
    content = _read(REPO_ROOT / "Dockerfile.gpu")
    assert "EXPOSE 8000" in content
    assert 'CMD ["uv", "run", "uvicorn", "autobot_stt.main:app"' in content


# --- .dockerignore ---------------------------------------------------------


def _dockerignore_patterns() -> list[str]:
    content = _read(REPO_ROOT / ".dockerignore")
    return [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize(
    "pattern",
    [
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".env",
        ".env.*",
        ".autobot_*",
        "tests",
        "docs",
    ],
)
def test_dockerignore_excludes_pattern(pattern: str) -> None:
    assert pattern in _dockerignore_patterns()


@pytest.mark.parametrize("pattern", ["!.env.example", "!README.md"])
def test_dockerignore_negates_pattern(pattern: str) -> None:
    """``.env.example`` and ``README.md`` must be kept despite blanket exclusions."""
    assert pattern in _dockerignore_patterns()


def test_dockerignore_negation_follows_parent_rule() -> None:
    """Negations must appear after the pattern they un-ignore (Docker rule)."""
    patterns = _dockerignore_patterns()
    env_star_idx = patterns.index(".env.*")
    env_example_idx = patterns.index("!.env.example")
    md_idx = patterns.index("*.md")
    readme_idx = patterns.index("!README.md")
    assert env_example_idx > env_star_idx
    assert readme_idx > md_idx


# --- docker compose config (real validation) -------------------------------


skip_if_no_docker = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker CLI not available",
)


def _compose_config(profile: str | None = None) -> subprocess.CompletedProcess[str]:
    cmd: list[str] = ["docker", "compose"]
    if profile is not None:
        cmd += ["--profile", profile]
    cmd += ["-f", str(REPO_ROOT / "docker-compose.yml"), "config", "--quiet"]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@skip_if_no_docker
def test_docker_compose_config_validates_cpu() -> None:
    """``docker compose config`` must accept the default CPU service definition."""
    result = _compose_config()
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"


@skip_if_no_docker
def test_docker_compose_config_validates_gpu_profile() -> None:
    """``docker compose config`` must accept the ``gpu`` profile service."""
    result = _compose_config(profile="gpu")
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
