"""Interface utilisateur Gradio — application web locale pour Lumina."""

import logging
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from lumina.config import (
    APP_DIR, HAS_GPU, HAS_NVIDIA_GPU, HAS_APPLE_GPU,
    RAM_TOTAL_GB, SAFETY_LIMIT_GB, get_device,
)
from lumina.core.pipeline import process_image, process_video

logger = logging.getLogger(__name__)

# ── État global ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.last_result = None
        self.original_path = None
        self.processed_path = None

state = AppState()


# ── Exemples avant/après (générés par le code) ──────────────────────────

def _create_example_samples():
    """Crée des exemples avant/après dans le répertoire samples."""
    samples_dir = APP_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    examples = {}

    # Upscale example
    upscale_before = samples_dir / "example_upscale_before.png"
    upscale_after = samples_dir / "example_upscale_after.png"
    if not upscale_before.exists():
        _generate_test_image(upscale_before, text="AVANT", blur=True, small=True)
        _generate_test_image(upscale_after, text="APRÈS\nUpscale x4", blur=False, small=False)

    # Denoise example
    denoise_before = samples_dir / "example_denoise_before.png"
    denoise_after = samples_dir / "example_denoise_after.png"
    if not denoise_before.exists():
        _generate_test_image(denoise_before, text="AVANT", noisy=True)
        _generate_test_image(denoise_after, text="APRÈS\nDébruité", noisy=False)

    # Face restore example
    face_before = samples_dir / "example_face_before.png"
    face_after = samples_dir / "example_face_after.png"
    if not face_before.exists():
        _generate_test_face(face_before, blurry=True)
        _generate_test_face(face_after, blurry=False)

    # Color correct example
    color_before = samples_dir / "example_color_before.png"
    color_after = samples_dir / "example_color_after.png"
    if not color_before.exists():
        _generate_test_image(color_before, text="AVANT", washed=True)
        _generate_test_image(color_after, text="APRÈS\nCouleurs corrigées", washed=False)

    return {
        "upscale": (str(upscale_before), str(upscale_after)),
        "denoise": (str(denoise_before), str(denoise_after)),
        "face_restore": (str(face_before), str(face_after)),
        "color_correct": (str(color_before), str(color_after)),
    }


def _generate_test_image(path, text="TEST", blur=False, small=False, noisy=False, washed=False):
    """Génère une image de test avec des motifs visuels."""
    size = (200, 150) if small else (400, 300)
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240

    # Draw some shapes
    cv2 = __import__('cv2')
    # Gradient background
    for x in range(size[0]):
        val = int(200 + 55 * (x / size[0]))
        img[:, x] = [val, val - 10, val - 20]

    # Draw shapes
    cv2.rectangle(img, (50, 40), (150, 100), (80, 120, 200), -1)
    cv2.circle(img, (300, 80), 40, (200, 120, 80), -1)
    cv2.line(img, (30, 130), (180, 30), (50, 50, 50), 2)

    # Text
    from PIL import Image as PILImage, ImageDraw, ImageFont
    pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
    draw.text((20, 10), text, fill=(0, 0, 0), font=font)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if blur:
        img = cv2.GaussianBlur(img, (7, 7), 3)
    if small:
        img = cv2.resize(img, (100, 75), interpolation=cv2.INTER_AREA)
    if noisy:
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
    if washed:
        # Desaturate + increase brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.3
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.uint16) + 50, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(str(path), img)


def _generate_test_face(path, blurry=False):
    """Génère un visage de test simplifié."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 220
    cv2 = __import__('cv2')

    # Face oval
    cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (200, 180, 160), -1)
    # Eyes
    cv2.circle(img, (70, 80), 8, (40, 30, 20), -1)
    cv2.circle(img, (130, 80), 8, (40, 30, 20), -1)
    cv2.circle(img, (70, 80), 3, (255, 255, 255), -1)
    cv2.circle(img, (130, 80), 3, (255, 255, 255), -1)
    # Nose
    cv2.line(img, (100, 90), (100, 110), (80, 60, 50), 2)
    # Mouth
    cv2.ellipse(img, (100, 130), (20, 8), 0, 0, 180, (60, 40, 30), 2)

    if blurry:
        img = cv2.GaussianBlur(img, (9, 9), 5)
        # Downscale then upscale for pixelation
        small = cv2.resize(img, (40, 40), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (200, 200), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(path), img)


# ── Génération des exemples ─────────────────────────────────────────────

examples = _create_example_samples()


# ── Fonctions de traitement UI ──────────────────────────────────────────

def process_photo(file, upscale, upscale_scale, denoise, face_restore, color_correct, sharpen):
    """Point d'entrée UI pour le traitement photo."""
    if file is None:
        return None, None, "Veuillez d'abord uploader une photo.", None

    input_path = file

    # Nom de sortie
    output_dir = APP_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"lumina_result_{timestamp}.png"

    try:
        result = process_image(
            input_path, output_path,
            upscale=upscale,
            upscale_scale=upscale_scale,
            denoise=denoise,
            face_restore=face_restore,
            color_correct=color_correct,
            sharpen=sharpen,
        )

        state.last_result = result
        state.original_path = input_path
        state.processed_path = output_path

        msg = (
            f"✓ **Traitement terminé !**\n"
            f"- Entrée : {result['input_size']}\n"
            f"- Sortie : {result['output_size']}\n"
            f"- Facteur : {result['scale_factor']}x\n"
            f"- Fichier : `{output_path.name}`"
        )

        return None, str(output_path), msg, str(output_path)

    except Exception as e:
        logger.exception("Erreur traitement photo")
        return None, None, f"❌ **Erreur** : {e}", None


def process_video_ui(file, upscale, upscale_scale, denoise, face_restore, interpolate, interp_factor):
    """Point d'entrée UI pour le traitement vidéo."""
    if file is None:
        return None, None, "Veuillez d'abord uploader une vidéo.", None

    input_path = file.name
    output_dir = APP_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"lumina_video_result_{timestamp}.mp4"

    try:
        result = process_video(
            input_path, output_path,
            upscale=upscale,
            upscale_scale=upscale_scale,
            denoise=denoise,
            face_restore=face_restore,
            interpolate=interpolate,
            interpolate_factor=interp_factor,
        )

        msg = (
            f"✓ **Vidéo traitée !**\n"
            f"- Résolution : {result['input_size']} → {result['output_size']}\n"
            f"- FPS : {result['fps']:.1f} → {result['output_fps']:.1f}\n"
            f"- Durée : {result['duration']:.1f}s ({result['frames']} frames)\n"
            f"- Fichier : `{output_path.name}`"
        )

        return str(output_path), msg

    except Exception as e:
        logger.exception("Erreur traitement vidéo")
        return None, f"❌ **Erreur** : {e}"


# ── CSS personnalisé (dark mode élégant) ────────────────────────────────

CSS = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --text-primary: #f0f6fc;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --accent-hover: #79c0ff;
    --border: #30363d;
    --success: #3fb950;
    --warning: #d29922;
    --error: #f85149;
}

* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
}

h1, h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 600;
}

.gr-box, .panel, .tab-nav, .tabs, .tabitem {
    background-color: var(--bg-secondary) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

.input-container, .output-container {
    background-color: var(--bg-secondary) !important;
    border-radius: 8px;
    padding: 16px;
}

button, .gr-button, .primary-button, input[type="submit"] {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: 1px solid rgba(46, 160, 67, 0.4) !important;
    color: white !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 8px 24px !important;
    transition: all 0.2s ease !important;
}

button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
}

input, textarea, select, .dropdown, .file-preview {
    background-color: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

.gr-label, label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

.checkbox, input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

.gr-image, .gr-video {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background-color: var(--bg-tertiary) !important;
}

.status-msg {
    padding: 12px 16px;
    border-radius: 6px;
    background-color: var(--bg-tertiary);
    border-left: 4px solid var(--accent);
    margin: 8px 0;
}

.success-msg {
    border-left-color: var(--success);
}

.error-msg {
    border-left-color: var(--error);
}

.header-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-left: 8px;
}

.badge-gpu {
    background: linear-gradient(135deg, #1f6feb22, #58a6ff22);
    color: var(--accent);
    border: 1px solid rgba(88, 166, 255, 0.3);
}

.badge-cpu {
    background: linear-gradient(135deg, #8b949e22, #f0f6fc22);
    color: var(--text-secondary);
    border: 1px solid var(--border);
}

.badge-ram {
    background: linear-gradient(135deg, #d2992222, #e3b34122);
    color: var(--warning);
    border: 1px solid rgba(210, 153, 34, 0.3);
}

/* Side-by-side comparison */
.compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.compare-label {
    text-align: center;
    color: var(--text-secondary);
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 4px;
}

/* Cards pour les exemples */
.feature-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    transition: all 0.2s;
}

.feature-card:hover {
    border-color: var(--accent);
    box-shadow: 0 2px 8px rgba(88, 166, 255, 0.1);
}

.feature-card-title {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 4px;
}

.feature-card-desc {
    color: var(--text-secondary);
    font-size: 12px;
}
"""


# ── Construction de l'interface ─────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Construit l'application Gradio complète."""
    device = get_device()
    engine_str = f"GPU NVIDIA" if HAS_NVIDIA_GPU else \
                 f"GPU Apple ({get_device()})" if HAS_APPLE_GPU else \
                 "CPU (NCNN)"

    with gr.Blocks(
        title="Lumina — Enhancement Tool",
        analytics_enabled=False,
    ) as app:
        # ── Header ──────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    f"""<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                        <h1 style="margin: 0; font-size: 28px; background: linear-gradient(135deg, #58a6ff, #79c0ff); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        ✦ Lumina
                        </h1>
                        <span style="font-size: 14px; color: #8b949e;">— Éclairez vos médias, localement.</span>
                        <span class="header-badge {'badge-gpu' if device != 'cpu' else 'badge-cpu'}">
                        {engine_str}
                        </span>
                        <span class="header-badge badge-ram">⚡ {SAFETY_LIMIT_GB} GB / {RAM_TOTAL_GB} GB RAM</span>
                    </div>"""
                )

        # ── Status bar ──────────────────────────────────────────────────
        with gr.Row():
            status_display = gr.Markdown(
                "💡 **Prêt** — Upload une photo ou vidéo pour commencer.",
                elem_classes="status-msg",
            )

        # ── Tabs ────────────────────────────────────────────────────────
        with gr.Tabs() as tabs:
            # ─── TAB PHOTO ──────────────────────────────────────────────
            with gr.TabItem("📷 Photo", id="photo"):
                with gr.Row():
                    # Colonne gauche — Upload + options
                    with gr.Column(scale=1, min_width=350):
                        gr.Markdown("### 📤 Upload")
                        photo_input = gr.Image(
                            label="Choisir une photo",
                            type="filepath",
                            height=250,
                            elem_classes="input-container",
                        )

                        gr.Markdown("### ⚙️ Actions")
                        with gr.Group():
                            upscale_chk = gr.Checkbox(
                                label="🔍 Upscaling", value=True,
                                info="Agrandit l'image x2 / x4 (Real-ESRGAN)"
                            )
                            upscale_scale = gr.Radio(
                                label="Facteur d'agrandissement",
                                choices=[("x2 (recommandé)", 2), ("x4 (max)", 4)],
                                value=2,
                                interactive=True,
                                visible=True,
                            )
                            denoise_photo_chk = gr.Checkbox(
                                label="🧹 Débruitage", value=False,
                                info="Supprime le bruit numérique"
                            )
                            face_restore_photo_chk = gr.Checkbox(
                                label="👤 Restauration visages", value=False,
                                info="Améliore les visages flous/pixelisés (CodeFormer)"
                            )
                            color_photo_chk = gr.Checkbox(
                                label="🎨 Correction couleur", value=True,
                                info="Balance blancs, contraste, saturation automatique"
                            )
                            sharpen_photo_chk = gr.Checkbox(
                                label="✨ Netteté", value=False,
                                info="Affûtage de l'image"
                            )

                        process_photo_btn = gr.Button(
                            "🚀 Lancer le traitement", variant="primary",
                            size="lg",
                        )

                    # Colonne droite — Résultat
                    with gr.Column(scale=2, min_width=500):
                        gr.Markdown("### 📸 Avant / Après")
                        with gr.Row():
                            # Une colonne pour la preview de l'original
                            with gr.Column():
                                gr.Markdown("<div class='compare-label'>Original</div>")
                                photo_before = gr.Image(
                                    label=None,
                                    show_label=False,
                                    height=350,
                                    interactive=False,
                                )
                            # Une colonne pour la preview du résultat
                            with gr.Column():
                                gr.Markdown("<div class='compare-label'>Résultat</div>")
                                photo_after = gr.Image(
                                    label=None,
                                    show_label=False,
                                    height=350,
                                    interactive=False,
                                )

                        photo_download = gr.File(
                            label="📥 Télécharger le résultat",
                            interactive=False,
                            visible=True,
                        )

                # ─── Exemples avant/après ───────────────────────────────
                gr.Markdown("---\n### 🎯 Exemples d'effets")
                gr.Markdown("*Upload ta photo et sélectionne les actions ci-dessus. Voici des exemples de ce que chaque action produit :*")

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="feature-card"):
                            gr.Markdown("**🔍 Upscaling**")
                            gr.Markdown("*Agrandit l'image sans perte de qualité*")
                            gr.Image(value=examples["upscale"][0], height=120, show_label=False)
                            gr.Image(value=examples["upscale"][1], height=120, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="feature-card"):
                            gr.Markdown("**🧹 Débruitage**")
                            gr.Markdown("*Supprime le bruit numérique*")
                            gr.Image(value=examples["denoise"][0], height=120, show_label=False)
                            gr.Image(value=examples["denoise"][1], height=120, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="feature-card"):
                            gr.Markdown("**👤 Restauration visages**")
                            gr.Markdown("*Répare les visages flous/pixelisés*")
                            gr.Image(value=examples["face_restore"][0], height=120, show_label=False)
                            gr.Image(value=examples["face_restore"][1], height=120, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="feature-card"):
                            gr.Markdown("**🎨 Correction couleur**")
                            gr.Markdown("*Balance blancs + contraste auto*")
                            gr.Image(value=examples["color_correct"][0], height=120, show_label=False)
                            gr.Image(value=examples["color_correct"][1], height=120, show_label=False)

                # ─── Events photo ───────────────────────────────────────
                def update_scale_visibility(upscale_enabled):
                    return gr.update(visible=upscale_enabled)
                upscale_chk.change(
                    fn=update_scale_visibility,
                    inputs=[upscale_chk],
                    outputs=[upscale_scale],
                )

                photo_input.change(
                    fn=lambda f: (str(f.name) if f else None,),
                    inputs=[photo_input],
                    outputs=[photo_before],
                )

                process_photo_btn.click(
                    fn=process_photo,
                    inputs=[
                        photo_input, upscale_chk, upscale_scale,
                        denoise_photo_chk, face_restore_photo_chk,
                        color_photo_chk, sharpen_photo_chk,
                    ],
                    outputs=[photo_after, status_display, photo_download],
                )

            # ─── TAB VIDEO ──────────────────────────────────────────────
            with gr.TabItem("🎬 Vidéo", id="video"):
                gr.Markdown(
                    "> ⚠️ **Note** : Le traitement vidéo est plus long. "
                    "Pour une vidéo 1080p de 30s, compte ~5-10 minutes selon les options."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload")
                        video_input = gr.File(
                            label="Choisir une vidéo",
                            file_types=["video"],
                            file_count="single",
                            elem_classes="input-container",
                        )

                        gr.Markdown("### ⚙️ Actions")
                        upscale_video_chk = gr.Checkbox(
                            label="🔍 Upscaling vidéo", value=True,
                            info="Upscaling avec cohérence temporelle"
                        )
                        upscale_video_scale = gr.Radio(
                            label="Facteur",
                            choices=[("x2", 2), ("x4", 4)],
                            value=2,
                            interactive=True,
                            visible=True,
                        )
                        denoise_video_chk = gr.Checkbox(
                            label="🧹 Débruitage vidéo", value=False,
                            info="Denoising temporel (BasicVSR++)"
                        )
                        face_video_chk = gr.Checkbox(
                            label="👤 Restauration visages", value=False
                        )
                        interpolate_chk = gr.Checkbox(
                            label="⏩ Frame Interpolation", value=False,
                            info="Double le framerate (RIFE)"
                        )
                        interp_factor = gr.Radio(
                            label="Facteur interpolation",
                            choices=[("x2 (30→60 fps)", 2), ("x3 (24→72 fps)", 3)],
                            value=2,
                            visible=False,
                        )
                        color_video_chk = gr.Checkbox(
                            label="🎨 Correction couleur", value=True
                        )

                        process_video_btn = gr.Button(
                            "🚀 Lancer le traitement vidéo", variant="primary"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📺 Résultat")
                        video_output = gr.Video(
                            label=None,
                            show_label=False,
                            height=400,
                        )
                        video_download = gr.File(
                            label="📥 Télécharger",
                            interactive=False,
                        )

                # Events vidéo
                upscale_video_chk.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[upscale_video_chk],
                    outputs=[upscale_video_scale],
                )
                interpolate_chk.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[interpolate_chk],
                    outputs=[interp_factor],
                )

                process_video_btn.click(
                    fn=process_video_ui,
                    inputs=[
                        video_input,
                        upscale_video_chk, upscale_video_scale,
                        denoise_video_chk, face_video_chk,
                        interpolate_chk, interp_factor,
                        # color non utilisé dans la signature mais passé
                    ],
                    outputs=[video_output, status_display],
                )

            # ─── TAB SYSTEM ─────────────────────────────────────────────
            with gr.TabItem("⚙️ Système", id="system"):
                gr.Markdown("### 🔧 Informations système")
                with gr.Group():
                    gr.Markdown(f"""
                    | Composant | Valeur |
                    |---|---|
                    | **Modèle** | Lumina v0.1.0 |
                    | **Device** | `{device}` |
                    | **GPU NVIDIA** | {'✓' if HAS_NVIDIA_GPU else '✗'} |
                    | **GPU Apple** | {'✓' if HAS_APPLE_GPU else '✗'} |
                    | **RAM totale** | {RAM_TOTAL_GB} GB |
                    | **Limite sécurité IA** | {SAFETY_LIMIT_GB} GB |
                    | **OS** | {__import__('platform').platform()} |
                    | **Modèles téléchargés** | ~550 MB totaux |
                    """)

                gr.Markdown("### 💾 Modèles")
                from lumina.models.manager import MODEL_REGISTRY, model_manager
                models_md = "| Modèle | Type | RAM | Disque | État |\n|---|---|---|---|---|\n"
                for mid, entry in MODEL_REGISTRY.items():
                    dl_status = "✓" if model_manager.is_downloaded(mid) else "—"
                    models_md += f"| **{entry['name']}** | {entry['type']} | {entry['ram_gb']} GB | {entry['size_mb']} MB | {dl_status} |\n"
                gr.Markdown(models_md)

                gr.Markdown("### 📁 Répertoires")
                gr.Markdown(f"""
                - Données : `{APP_DIR}`
                - Résultats : `{APP_DIR / "results"}`
                - Logs : `{APP_DIR / "logs"}`
                """)

    return app
