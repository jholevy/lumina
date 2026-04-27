"""Configuration centralisée — détection GPU/CPU, limites mémoire, chemins modèles."""

import os
import platform
import subprocess
import sys
from pathlib import Path

# ── Chemins ────────────────────────────────────────────────────────────

HOME = Path.home()
APP_DIR = HOME / ".lumina"
MODELS_DIR = APP_DIR / "models"
CACHE_DIR = APP_DIR / "cache"
LOGS_DIR = APP_DIR / "logs"
SAMPLES_DIR = APP_DIR / "samples"

for d in [APP_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR, SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Détection hardware ──────────────────────────────────────────────────

SYSTEM = platform.system()
MACHINE = platform.machine()
IS_MAC = SYSTEM == "Darwin"
IS_MAC_ARM = IS_MAC and MACHINE == "arm64"
IS_LINUX = SYSTEM == "Linux"
IS_WINDOWS = SYSTEM == "Windows"

def _has_nvidia_gpu() -> bool:
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return True
    except FileNotFoundError:
        return False

def _has_apple_gpu() -> bool:
    """Vérifie si Metal Performance Shaders est dispo (Apple Silicon ou AMD)"""
    if not IS_MAC:
        return False
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        return "Metal" in result.stdout
    except Exception:
        return False

HAS_NVIDIA_GPU = _has_nvidia_gpu()
HAS_APPLE_GPU = _has_apple_gpu()
HAS_GPU = HAS_NVIDIA_GPU or HAS_APPLE_GPU

RAM_TOTAL_GB = 0
if IS_MAC:
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=5
        )
        RAM_TOTAL_GB = int(out.strip()) // (1024**3)
    except Exception:
        RAM_TOTAL_GB = 8  # fallback
elif IS_LINUX:
    try:
        import psutil
        RAM_TOTAL_GB = psutil.virtual_memory().total // (1024**3)
    except ImportError:
        RAM_TOTAL_GB = 8
elif IS_WINDOWS:
    try:
        import psutil
        RAM_TOTAL_GB = psutil.virtual_memory().total // (1024**3)
    except ImportError:
        RAM_TOTAL_GB = 8

# Limite de sécurité : on ne dépasse jamais 70% de la RAM totale
SAFETY_LIMIT_GB = int(RAM_TOTAL_GB * 0.7)


def get_device() -> str:
    """Retourne le device PyTorch optimal."""
    if HAS_NVIDIA_GPU:
        return "cuda"
    if HAS_APPLE_GPU:
        return "mps"
    return "cpu"


def available_engines() -> list[str]:
    """Moteurs disponibles pour chaque module."""
    engines = ["pytorch"]
    if HAS_NVIDIA_GPU:
        engines.append("tensorrt")
    if HAS_APPLE_GPU:
        engines.append("mps")
    # NCNN / Vulkan dispo partout avec un binaire standalone
    engines.append("ncnn")
    return engines
