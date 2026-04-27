"""Gestionnaire de modèles IA — téléchargement, cache, chargement mémoire."""

import hashlib
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

from lumina.config import MODELS_DIR, SAFETY_LIMIT_GB, get_device

logger = logging.getLogger(__name__)


# ── Registre des modèles disponibles ────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "real_esrgan": {
        "name": "Real-ESRGAN",
        "description": "Upscaling général photo/vidéo x2/x4 + denoising",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "size_mb": 67,
        "ram_gb": 2,
        "type": "upscale",
    },
    "real_esrgan_anime": {
        "name": "Real-ESRGAN Anime",
        "description": "Upscaling optimisé pour animations/mangas",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "size_mb": 17,
        "ram_gb": 1.5,
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
    "gfpgan": {
        "name": "GFPGAN",
        "description": "Restauration de visages (fallback plus rapide)",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "filename": "GFPGANv1.3.pth",
        "size_mb": 340,
        "ram_gb": 1.5,
        "type": "face",
    },
    "rife": {
        "name": "RIFE",
        "description": "Frame interpolation vidéo (doublement FPS)",
        "url": "https://github.com/hzwer/ECCV2022-RIFE/releases/download/v4.6/flownet.pkl",
        "filename": "rife_v4.6.pth",
        "size_mb": 20,
        "ram_gb": 1,
        "type": "interpolation",
    },
    "basicvsr_pp": {
        "name": "BasicVSR++",
        "description": "Denoising vidéo avec cohérence temporelle",
        "url": "https://github.com/ckkelvinchan/BasicVSR_PlusPlus/releases/download/v1.0.0/BasicVSRPP_REDS4.pth",
        "filename": "BasicVSRPP_REDS4.pth",
        "size_mb": 200,
        "ram_gb": 3,
        "type": "denoise",
    },
    "color_cnn": {
        "name": "ColorNet",
        "description": "Correction couleur / balance auto",
        "url": None,  # Sera entraîné ou remplacé par algo classique
        "filename": None,
        "size_mb": 5,
        "ram_gb": 0.5,
        "type": "color",
    },
}


class ModelManager:
    """Gère le téléchargement, cache et chargement des modèles en mémoire."""

    def __init__(self):
        self._loaded: dict[str, Any] = {}
        self._current_ram_gb = 0.0

    # ── Téléchargement ──────────────────────────────────────────────────

    def is_downloaded(self, model_id: str) -> bool:
        entry = MODEL_REGISTRY.get(model_id)
        if not entry or not entry.get("filename"):
            return True  # pas de téléchargement nécessaire
        model_path = MODELS_DIR / entry["filename"]
        return model_path.exists()

    def download(self, model_id: str, progress_cb: Optional[Callable] = None) -> Path:
        """Télécharge un modèle s'il n'est pas déjà en cache."""
        import requests

        entry = MODEL_REGISTRY.get(model_id)
        if not entry:
            raise ValueError(f"Modèle inconnu : {model_id}")

        if not entry.get("url"):
            raise ValueError(f"Pas d'URL pour {model_id}")

        model_path = MODELS_DIR / entry["filename"]

        if model_path.exists():
            logger.info(f"Modèle déjà en cache : {model_path}")
            return model_path

        logger.info(f"Téléchargement de {entry['name']} ({entry['size_mb']} MB)...")
        logger.info(f"  URL : {entry['url']}")

        response = requests.get(entry["url"], stream=True, timeout=300)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if progress_cb and total > 0:
                    progress_cb(downloaded / total)

        logger.info(f"✓ Modèle téléchargé : {model_path}")
        return model_path

    def download_all(self, progress_cb: Optional[Callable] = None):
        """Télécharge tous les modèles non présents."""
        total_models = len(MODEL_REGISTRY)
        for i, model_id in enumerate(MODEL_REGISTRY):
            if not self.is_downloaded(model_id):
                self.download(model_id, progress_cb)
            if progress_cb:
                progress_cb((i + 1) / total_models)

    # ── Chargement mémoire avec swap ────────────────────────────────────

    def _ensure_ram_headroom(self, needed_gb: float) -> None:
        """Libère de la RAM en déchargeant des modèles si nécessaire."""
        while (self._current_ram_gb + needed_gb) > SAFETY_LIMIT_GB:
            if not self._loaded:
                raise MemoryError(
                    f"Besoin de {needed_gb} GB mais même après avoir tout déchargé, "
                    f"le total ({SAFETY_LIMIT_GB} GB) est insuffisant."
                )
            # Décharge le plus récent (LIFO — swap séquentiel)
            model_id = next(iter(self._loaded))
            self.unload(model_id)

    def load(self, model_id: str, device: Optional[str] = None) -> Any:
        """Charge un modèle en mémoire, swap si nécessaire."""
        entry = MODEL_REGISTRY.get(model_id)
        if not entry:
            raise ValueError(f"Modèle inconnu : {model_id}")

        if model_id in self._loaded:
            return self._loaded[model_id]

        ram_needed = entry["ram_gb"]
        self._ensure_ram_headroom(ram_needed)

        logger.info(f"Chargement de {entry['name']}...")
        model_obj = self._load_model(model_id, device or get_device())

        self._loaded[model_id] = model_obj
        self._current_ram_gb += ram_needed
        logger.info(f"✓ {entry['name']} chargé. RAM modèles : {self._current_ram_gb:.1f} GB")
        return model_obj

    def unload(self, model_id: str) -> None:
        """Décharge un modèle de la mémoire."""
        if model_id not in self._loaded:
            return
        entry = MODEL_REGISTRY.get(model_id)
        ram_freed = entry["ram_gb"] if entry else 0
        del self._loaded[model_id]
        self._current_ram_gb = max(0, self._current_ram_gb - ram_freed)
        import gc; gc.collect()
        logger.info(f"Déchargé : {entry['name']} ({ram_freed} GB libérés)")

    def unload_all(self) -> None:
        """Décharge tous les modèles."""
        for model_id in list(self._loaded.keys()):
            self.unload(model_id)

    def _load_model(self, model_id: str, device: str) -> Any:
        """Charge effectivement un modèle PyTorch."""
        import torch

        entry = MODEL_REGISTRY[model_id]
        model_path = MODELS_DIR / entry["filename"]

        if model_id == "real_esrgan":
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=None,  # sera créé automatiquement
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=device != "cpu",
                device=device,
            )
            return model

        elif model_id == "codeformer":
            from codeformer import CodeFormer as CF
            model = CF(
                dim_latent=512,
                num_heads=8,
                max_iter=3,
                fidelity=0.7,
            )
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["params_ema"])
            model = model.to(device).eval()
            return model

        elif model_id == "rife":
            # RIFE — modèle de frame interpolation
            import torch
            from model.RIFE_arch import IFNet
            model = IFNet()
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model = model.to(device).eval()
            return model

        elif model_id == "basicvsr_pp":
            import torch
            from basicsr.archs.basicvsr_arch import BasicVSR
            model = BasicVSR(
                mid_channels=64,
                num_blocks=30,
                max_residue_magnitude=10,
                spynet_pretrained=None,
            )
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["params"])
            model = model.to(device).eval()
            return model

        elif model_id == "gfpgan":
            from gfpgan import GFPGANer
            model = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=device,
            )
            return model

        else:
            raise NotImplementedError(f"Chargement non implémenté pour {model_id}")


# Singleton
model_manager = ModelManager()
