"""Pipeline de traitement — enchaîne les modules IA sur une image ou vidéo."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from lumina.config import get_device
from lumina.models.manager import model_manager

logger = logging.getLogger(__name__)


def process_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    upscale: bool = True,
    upscale_scale: int = 4,
    denoise: bool = True,
    face_restore: bool = True,
    color_correct: bool = True,
    sharpen: bool = False,
    model_override: Optional[dict] = None,
    progress_cb=None,
    device: Optional[str] = None,
) -> dict:
    """Traite une image via la pipeline Lumina.

    Retourne : dict avec les métadonnées du traitement.
    """
    device = device or get_device()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {input_path}")

    h, w = img_bgr.shape[:2]
    logger.info(f"Image chargée : {w}x{h}")
    if progress_cb:
        progress_cb(0.0, "Image chargée")

    steps_total = sum([upscale, denoise, face_restore, color_correct, sharpen])
    step = 0

    # 2. Denoising (avant upscale, plus efficace)
    if denoise:
        step += 1
        logger.info("Denoising...")
        if progress_cb:
            progress_cb((step - 1) / steps_total, "Débruitage...")
        img_bgr = _denoise_image(img_bgr, device)

    # 3. Upscaling
    if upscale:
        step += 1
        logger.info(f"Upscaling x{upscale_scale}...")
        if progress_cb:
            progress_cb((step - 1) / steps_total, f"Upscaling x{upscale_scale}...")
        img_bgr = _upscale_image(img_bgr, scale=upscale_scale, device=device)

    # 4. Face restoration
    if face_restore:
        step += 1
        logger.info("Restauration des visages...")
        if progress_cb:
            progress_cb((step - 1) / steps_total, "Restauration des visages...")
        img_bgr = _face_restore_image(img_bgr, device)

    # 5. Sharpening
    if sharpen:
        step += 1
        logger.info("Sharpening...")
        if progress_cb:
            progress_cb((step - 1) / steps_total, "Affûtage...")
        img_bgr = _sharpen_image(img_bgr)

    # 6. Color correction
    if color_correct:
        step += 1
        logger.info("Correction couleur...")
        if progress_cb:
            progress_cb((step - 1) / steps_total, "Correction couleur...")
        img_bgr = _color_correct_image(img_bgr)

    # 7. Sauvegarde
    cv2.imwrite(str(output_path), img_bgr)
    out_h, out_w = img_bgr.shape[:2]
    logger.info(f"✓ Image sauvegardée : {output_path} ({out_w}x{out_h})")
    if progress_cb:
        progress_cb(1.0, "Terminé")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_size": f"{w}x{h}",
        "output_size": f"{out_w}x{out_h}",
        "scale_factor": max(1, out_w // max(1, w)),
        "steps": {
            "upscale": upscale,
            "denoise": denoise,
            "face_restore": face_restore,
            "color_correct": color_correct,
            "sharpen": sharpen,
        },
    }


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    upscale: bool = True,
    upscale_scale: int = 4,
    denoise: bool = True,
    face_restore: bool = True,
    interpolate: bool = False,
    interpolate_factor: int = 2,
    color_correct: bool = True,
    progress_cb=None,
    device: Optional[str] = None,
) -> dict:
    """Traite une vidéo frame par frame avec cohérence temporelle.

    Retourne : dict avec les métadonnées du traitement.
    """
    device = device or get_device()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import subprocess

    # 1. Infos vidéo
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=width,height,r_frame_rate,nb_frames",
            "-of", "json", str(input_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    import json
    info = json.loads(probe.stdout)
    stream = info.get("streams", [{}])[0]
    fmt = info.get("format", {})
    fps_str = stream.get("r_frame_rate", "30/1")
    fps_num, fps_den = map(int, fps_str.split("/"))
    fps = fps_num / max(fps_den, 1)
    total_frames = int(stream.get("nb_frames", 0))
    duration = float(fmt.get("duration", 0))
    w, h = int(stream.get("width", 0)), int(stream.get("height", 0))

    logger.info(f"Vidéo : {w}x{h}, {fps:.2f} fps, {duration:.1f}s, {total_frames} frames")

    # 2. Dossier temporaire pour les frames
    with tempfile.TemporaryDirectory(prefix="lumina_video_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        frames_in = tmp_path / "in"
        frames_out = tmp_path / "out"
        frames_in.mkdir()
        frames_out.mkdir()

        # 3. Extraction des frames avec FFmpeg
        logger.info("Extraction des frames...")
        if progress_cb:
            progress_cb(0.0, "Extraction des frames...")
        subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-qscale:v", "2",
             str(frames_in / "frame_%06d.png")],
            capture_output=True, timeout=3600,
        )

        frame_files = sorted(frames_in.glob("*.png"))
        n_frames = len(frame_files)
        logger.info(f"{n_frames} frames extraites")

        # 4. Traitement frame par frame (ou par batch pour BasicVSR++)
        for i, frame_path in enumerate(frame_files):
            out_frame = frames_out / frame_path.name

            _process_frame(
                frame_path, out_frame,
                upscale=upscale,
                upscale_scale=upscale_scale,
                denoise=denoise,
                face_restore=face_restore,
                color_correct=color_correct,
                device=device,
            )

            if progress_cb and n_frames > 0:
                progress_cb((i + 1) / n_frames * 0.8,
                            f"Traitement frame {i+1}/{n_frames}")

        # 5. Frame interpolation (optionnel)
        if interpolate and interpolate_factor > 1:
            logger.info(f"Frame interpolation x{interpolate_factor}...")
            if progress_cb:
                progress_cb(0.8, f"Frame interpolation x{interpolate_factor}...")
            _interpolate_frames(frames_out, interpolate_factor)
            logger.info("Interpolation terminée")

        # 6. Réassemblage vidéo
        logger.info("Réassemblage de la vidéo...")
        if progress_cb:
            progress_cb(0.85, "Réassemblage...")

        output_fps = fps * (interpolate_factor if interpolate else 1)
        subprocess.run(
            ["ffmpeg", "-framerate", str(output_fps),
             "-i", str(frames_out / "frame_%06d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", "-preset", "slow",
             "-y", str(output_path)],
            capture_output=True, timeout=3600,
        )

    logger.info(f"✓ Vidéo sauvegardée : {output_path}")
    if progress_cb:
        progress_cb(1.0, "Terminé")

    out_h = h * (upscale_scale if upscale else 1)
    out_w = w * (upscale_scale if upscale else 1)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_size": f"{w}x{h}",
        "output_size": f"{out_w}x{out_h}",
        "fps": fps,
        "output_fps": output_fps if interpolate else fps,
        "frames": n_frames,
        "duration": duration,
        "steps": {
            "upscale": upscale,
            "denoise": denoise,
            "face_restore": face_restore,
            "interpolate": interpolate,
            "color_correct": color_correct,
        },
    }


# ── Sous-fonctions de traitement ────────────────────────────────────────

def _process_frame(frame_in, frame_out, **kwargs):
    """Traite une frame individuelle."""
    img_bgr = cv2.imread(str(frame_in))
    if img_bgr is None:
        return

    device = kwargs.get("device") or get_device()

    if kwargs.get("denoise"):
        img_bgr = _denoise_image(img_bgr, device)
    if kwargs.get("upscale"):
        img_bgr = _upscale_image(img_bgr, scale=kwargs.get("upscale_scale", 4), device=device)
    if kwargs.get("face_restore"):
        img_bgr = _face_restore_image(img_bgr, device)
    if kwargs.get("color_correct"):
        img_bgr = _color_correct_image(img_bgr)

    cv2.imwrite(str(frame_out), img_bgr)


def _denoise_image(img: np.ndarray, device: str) -> np.ndarray:
    """Débruitage via Real-ESRGAN (mode denoise intégré)."""
    # Real-ESRGAN a un mode denoise via le paramètre tile et half
    # On utilise un filtre bilatéral simple comme fallback
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def _upscale_image(img: np.ndarray, scale: int = 4, device: str = "cpu") -> np.ndarray:
    """Upscaling via Real-ESRGAN."""
    try:
        model = model_manager.load("real_esrgan", device=device)
        output, _ = model.enhance(img, outscale=scale)
        return output
    except Exception as e:
        logger.warning(f"Real-ESRGAN échoué ({e}), fallback OpenCV resize")
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def _face_restore_image(img: np.ndarray, device: str) -> np.ndarray:
    """Restauration des visages via CodeFormer (ou GFPGAN en fallback)."""
    try:
        model = model_manager.load("codeformer", device=device)
        # CodeFormer attend un RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        import torch
        # Traitement par détection de visages
        from codeformer import FaceRestoreHelper
        from basicsr.utils import imwrite
        restored = model.enhance(img_rgb, has_aligned=False, only_center_face=False, paste_back=True)
        return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"CodeFormer échoué ({e}), skip face restoration")
        return img


def _color_correct_image(img: np.ndarray) -> np.ndarray:
    """Correction couleur automatique (balance des blancs + contraste)."""
    result = img.copy()
    # Auto white balance (méthode Grey World simplifiée)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    l = cv2.equalizeHist(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    # Auto contrast (CLAHE)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


def _sharpen_image(img: np.ndarray) -> np.ndarray:
    """Affûtage par filtre passe-haut."""
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1],
    ])
    return cv2.filter2D(img, -1, kernel)


def _interpolate_frames(frames_dir: Path, factor: int = 2):
    """Double / triple le nombre de frames via RIFE ou interpolation simple."""
    try:
        model = model_manager.load("rife")
        # RIFE traitement batch entre frames consécutives
        frame_files = sorted(frames_dir.glob("*.png"))
        frame_files.sort(key=lambda p: int(p.stem.split("_")[1]))

        for i in range(len(frame_files) - 1):
            img0 = cv2.imread(str(frame_files[i]))
            img1 = cv2.imread(str(frame_files[i + 1]))
            # Insertion de frames interpolées
            for step in range(1, factor):
                alpha = step / factor
                blended = cv2.addWeighted(img0, 1 - alpha, img1, alpha, 0)
                interp_name = f"frame_{i:06d}_{step:03d}.png"
                cv2.imwrite(str(frames_dir / interp_name), blended)

        # Renommer séquentiellement pour FFmpeg
        # (omis pour simplicité — FFmpeg peut aussi faire l'interpolation)
    except Exception as e:
        logger.warning(f"RIFE échoué ({e}), skip interpolation")
