#!/usr/bin/env python3
"""Lumina — Point d'entrée."""

import logging
import sys
from pathlib import Path

import gradio as gr

# S'assurer que le package est importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumina.config import APP_DIR, LOGS_DIR, get_device
from lumina.models.manager import model_manager
from lumina.ui.app import CSS, build_app


def setup_logging():
    log_file = LOGS_DIR / "lumina.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("lumina")


def check_models():
    """Vérifie quels modèles sont téléchargés."""
    from lumina.models.manager import MODEL_REGISTRY
    logger = logging.getLogger("lumina")
    logger.info("Vérification des modèles…")
    for mid, entry in MODEL_REGISTRY.items():
        if entry.get("filename") and entry.get("url"):
            if model_manager.is_downloaded(mid):
                logger.info(f"  ✓ {entry['name']}")
            else:
                logger.info(f"  — {entry['name']} (à télécharger au premier usage)")


def main():
    logger = setup_logging()
    logger.info(f"✦ Lumina v0.1.0 — Device: {get_device()}")
    logger.info(f"  RAM: {APP_DIR}")

    check_models()
    logger.info("Lancement de l'interface…")

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        allowed_paths=[str(APP_DIR / "results")],
    )


if __name__ == "__main__":
    main()
