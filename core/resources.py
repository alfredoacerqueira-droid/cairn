"""System resource detection and local model recommendation."""

from __future__ import annotations

import re
import subprocess


def get_system_resources() -> dict:
    """Detect system RAM, CPU, and GPU/VRAM resources. Never raises."""
    try:
        import psutil
    except ImportError:
        return {
            "ram_total_gb": 0.0,
            "ram_available_gb": 0.0,
            "cpu_count": 0,
            "vram_total_gb": None,
            "vram_free_gb": None,
            "gpu_name": None,
        }

    try:
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024**3), 1)
        ram_available_gb = round(mem.available / (1024**3), 1)
    except Exception:
        ram_total_gb = 0.0
        ram_available_gb = 0.0

    try:
        cpu_count = psutil.cpu_count() or 0
    except Exception:
        cpu_count = 0

    vram_total_gb = None
    vram_free_gb = None
    gpu_name = None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parsed = _parse_nvidia_smi_line(result.stdout.strip().split("\n")[0])
            gpu_name = parsed[0]
            vram_total_gb = round(parsed[1] / 1024.0, 1)
            vram_free_gb = round(parsed[2] / 1024.0, 1)
    except Exception:
        pass

    return {
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "cpu_count": cpu_count,
        "vram_total_gb": vram_total_gb,
        "vram_free_gb": vram_free_gb,
        "gpu_name": gpu_name,
    }


def _parse_nvidia_smi_line(line: str) -> tuple[str, float, float]:
    """Parse a single nvidia-smi CSV line into (name, total_mib, free_mib)."""
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if len(parts) > 0 else "Unknown"
    total = float(parts[1]) if len(parts) > 1 else 0.0
    free = float(parts[2]) if len(parts) > 2 else 0.0
    return name, total, free


def recommend_local_models(resources: dict, installed: list[dict]) -> dict:
    """Recommend worker/embed models based on available resources.

    Args:
        resources: dict from get_system_resources()
        installed: list of {"name": str, "size_gb": float}

    Returns:
        {"worker": {"model": str, "reason": str},
         "embed": {"model": str, "reason": str},
         "suggested_num_ctx": int,
         "budget_gb": float}
    """
    ram_avail = resources.get("ram_available_gb", 0.0) or 0.0
    vram_free = resources.get("vram_free_gb")

    if vram_free is not None:
        budget_gb = vram_free + ram_avail - 2.0
    else:
        budget_gb = ram_avail - 2.0

    budget_gb = max(round(budget_gb, 1), 0.0)

    embed_models = [m for m in installed if "embed" in m.get("name", "").lower()]
    worker_models = [m for m in installed if "embed" not in m.get("name", "").lower()]

    worker_rec = _recommend_worker(worker_models, budget_gb)
    embed_rec = _recommend_embed(embed_models, budget_gb)
    suggested_num_ctx = _suggest_num_ctx(resources, budget_gb)

    return {
        "worker": worker_rec,
        "embed": embed_rec,
        "suggested_num_ctx": suggested_num_ctx,
        "budget_gb": budget_gb,
    }


def _recommend_worker(worker_models: list[dict], budget_gb: float) -> dict:
    if not worker_models:
        return {
            "model": "qwen2.5-coder:3b",
            "reason": "no worker models installed; suggested default",
        }

    fitting = [m for m in worker_models if m.get("size_gb", 0.0) * 1.15 <= budget_gb]

    if fitting:
        fitting.sort(key=lambda m: m.get("size_gb", 0.0), reverse=True)
        best = fitting[0]
        return {
            "model": best["name"],
            "reason": f"largest worker fitting budget ({budget_gb:.1f} GB)",
        }

    worker_models.sort(key=lambda m: m.get("size_gb", 0.0))
    smallest = worker_models[0]
    return {
        "model": smallest["name"],
        "reason": (
            f"no worker fits {budget_gb:.1f} GB budget; "
            "smallest available (tight)"
        ),
    }


def _recommend_embed(embed_models: list[dict], budget_gb: float) -> dict:
    if not embed_models:
        return {
            "model": "nomic-embed-text",
            "reason": "no embed models installed; suggested default",
        }

    fitting = [m for m in embed_models if m.get("size_gb", 0.0) * 1.15 <= budget_gb]

    if fitting:
        fitting.sort(key=lambda m: m.get("size_gb", 0.0))
        best = fitting[0]
        return {
            "model": best["name"],
            "reason": f"smallest embedder fitting budget ({budget_gb:.1f} GB)",
        }

    embed_models.sort(key=lambda m: m.get("size_gb", 0.0))
    smallest = embed_models[0]
    return {
        "model": smallest["name"],
        "reason": (
            f"no embedder fits {budget_gb:.1f} GB budget; "
            "smallest available"
        ),
    }


def _suggest_num_ctx(resources: dict, budget_gb: float) -> int:
    if budget_gb >= 20:
        return 65536
    if budget_gb >= 12:
        return 32768
    if budget_gb < 8:
        return 8192
    return 16384


def list_installed_ollama_models() -> list[dict]:
    """List installed Ollama models with names and sizes. Returns [] on any failure."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        return _parse_ollama_list(result.stdout)
    except Exception:
        return []


def _parse_ollama_list(output: str) -> list[dict]:
    """Parse `ollama list` output into [{"name": str, "size_gb": float}]."""
    models = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("NAME"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        size_gb = _parse_size_gb(" ".join(parts[1:]))
        models.append({"name": name, "size_gb": size_gb})
    return models


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(GB|MB|TB)", re.IGNORECASE)


def _parse_size_gb(text: str) -> float:
    """Parse a size string like '9.6 GB' or '500 MB' into float GB."""
    m = _SIZE_RE.search(text)
    if not m:
        return 0.0
    value = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "TB":
        return round(value * 1024.0, 1)
    if unit == "MB":
        return round(value / 1024.0, 1)
    return round(value, 1)
