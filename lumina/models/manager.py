"""Gestionnaire de modèles IA — version simplifiée sans dépendances lourdes."""

import logging
from pathlib import Path
from typing import Any, Optional

from lumina.config import MODELS_DIR

logger = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "real_esrgan": {
        "name": "Real-ESRGAN",
        "description": "Upscaling photo/vidéo x2/x4 + denoising",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "size_mb": 67,
        "ram_gb": 2,
        "type": "upscale",
    },
    "codeformer": {
        "name": "CodeFormer",
        "description": "Restauration de visages après upscaling",
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "filename": "codeformer.pth",
        "size_mb": 180,
        "ram_gb": 2,
        "type": "face",
    },
    "basicvsr_pp": {
        "name": "BasicVSR++",
        "description": "Denoising vidéo temporel",
        "url": "https://github.com/ckkelvinchan/BasicVSR_PlusPlus/releases/download/v1.0.0/BasicVSRPP_REDS4.pth",
        "filename": "BasicVSRPP_REDS4.pth",
        "size_mb": 200,
        "ram_gb": 3,
        "type": "denoise",
    },
    "rife": {
        "name": "RIFE",
        "description": "Frame interpolation vidéo",
        "url": "https://github.com/hzwer/ECCV2022-RIFE/releases/download/v4.6/flownet.pkl",
        "filename": "rife_v4.6.pth",
        "size_mb": 20,
        "ram_gb": 1,
        "type": "interpolation",
    },
}


class ModelManager:
    """Gère le téléchargement et le cache des modèles."""

    def __init__(self):
        self._loaded: dict[str, Any] = {}

    def is_downloaded(self, model_id: str) -> bool:
        entry = MODEL_REGISTRY.get(model_id)
        if not entry or not entry.get("filename"):
            return True
        return (MODELS_DIR / entry["filename"]).exists()

    def download(self, model_id: str) -> Path:
        entry = MODEL_REGISTRY.get(model_id)
        if not entry:
            raise ValueError(f"Modèle inconnu : {model_id}")

        model_path = MODELS_DIR / entry["filename"]

        if model_path.exists():
            logger.info(f"Modèle déjà en cache : {model_path.name}")
            return model_path

        if not entry.get("url"):
            raise ValueError(f"Pas d'URL pour {model_id}")

        logger.info(f"Téléchargement : {entry['name']} ({entry['size_mb']} MB)...")
        import requests
        r = requests.get(entry["url"], stream=True, timeout=300)
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total
                    bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
                    print(f"\r  [{bar}] {pct*100:.0f}%", end="", flush=True)
        if total:
            print()
        logger.info(f"✓ Modèle téléchargé : {model_path}")
        return model_path

    def load(self, model_id: str) -> Any:
        raise NotImplementedError(
            "Les modèles lourds nécessitent PyTorch. "
            "Installez avec : pip install torch torchvision"
        )

    def unload(self, model_id: str) -> None:
        self._loaded.pop(model_id, None)
        import gc
        gc.collect()

    def unload_all(self) -> None:
        for mid in list(self._loaded):
            self.unload(mid)


model_manager = ModelManager()
