"""Pipeline de traitement — enchaîne les modules IA sur une image ou vidéo."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from lumina.config import get_device

logger = logging.getLogger(__name__)

# ── Flag pour savoir si les modèles lourds sont dispo ──────────────────

_HAS_TORCH = False
_HAS_REALESRGAN = False

try:
    import torch
    _HAS_TORCH = True
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        _HAS_REALESRGAN = True
    except ImportError:
        logger.info("Real-ESRGAN PyTorch non installé — fallback OpenCV")
except ImportError:
    logger.info("PyTorch non installé — fallback OpenCV pour tous les modèles")


# ── Pipeline principal ──────────────────────────────────────────────────

def process_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    upscale: bool = True,
    upscale_scale: int = 2,
    denoise: bool = True,
    face_restore: bool = True,
    color_correct: bool = True,
    sharpen: bool = False,
    progress_cb=None,
    device: Optional[str] = None,
) -> dict:
    """Traite une image via la pipeline Lumina."""
    device = device or get_device()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {input_path}")

    h, w = img_bgr.shape[:2]
    logger.info(f"Image chargée : {w}x{h}")
    if progress_cb:
        progress_cb(0.0, "Image chargée")

    steps = [s for s in [denoise, upscale, face_restore, sharpen, color_correct] if s]
    total_steps = len(steps) or 1
    step_idx = 0

    # 1. Denoising
    if denoise:
        step_idx += 1
        if progress_cb:
            progress_cb((step_idx - 1) / total_steps, "Débruitage...")
        logger.info("Débruitage...")
        img_bgr = _denoise_image(img_bgr)

    # 2. Upscaling
    if upscale:
        step_idx += 1
        if progress_cb:
            progress_cb((step_idx - 1) / total_steps, f"Upscaling x{upscale_scale}...")
        logger.info(f"Upscaling x{upscale_scale}...")
        img_bgr = _upscale_image(img_bgr, scale=upscale_scale)

    # 3. Face restoration
    if face_restore:
        step_idx += 1
        if progress_cb:
            progress_cb((step_idx - 1) / total_steps, "Restauration visages...")
        logger.info("Restauration visages...")
        img_bgr = _face_restore_image(img_bgr)

    # 4. Sharpening
    if sharpen:
        step_idx += 1
        if progress_cb:
            progress_cb((step_idx - 1) / total_steps, "Affûtage...")
        logger.info("Affûtage...")
        img_bgr = _sharpen_image(img_bgr)

    # 5. Color correction
    if color_correct:
        step_idx += 1
        if progress_cb:
            progress_cb((step_idx - 1) / total_steps, "Correction couleur...")
        logger.info("Correction couleur...")
        img_bgr = _color_correct_image(img_bgr)

    # Sauvegarde
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
    upscale_scale: int = 2,
    denoise: bool = True,
    face_restore: bool = True,
    interpolate: bool = False,
    interpolate_factor: int = 2,
    color_correct: bool = True,
    progress_cb=None,
    device: Optional[str] = None,
) -> dict:
    """Traite une vidéo frame par frame."""
    device = device or get_device()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Infos vidéo via ffprobe
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=width,height,r_frame_rate,nb_frames",
            "-of", "json", str(input_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    import json as _json
    info = _json.loads(probe.stdout)
    stream = info.get("streams", [{}])[0]
    fmt = info.get("format", {})
    fps_str = stream.get("r_frame_rate", "30/1")
    fps_num, fps_den = map(int, fps_str.split("/"))
    fps = fps_num / max(fps_den, 1)
    n_frames = int(stream.get("nb_frames", 0))
    duration = float(fmt.get("duration", 0))
    w, h = int(stream.get("width", 0)), int(stream.get("height", 0))

    logger.info(f"Vidéo : {w}x{h}, {fps:.2f} fps, {duration:.1f}s")

    # Dossier temporaire pour les frames
    with tempfile.TemporaryDirectory(prefix="lumina_video_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        frames_in = tmp_path / "in"
        frames_out = tmp_path / "out"
        frames_in.mkdir()
        frames_out.mkdir()

        # Extraction via FFmpeg
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

        # Traitement frame par frame
        for i, frame_path in enumerate(frame_files):
            out_frame = frames_out / frame_path.name
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            if denoise:
                img = _denoise_image(img)
            if upscale:
                img = _upscale_image(img, scale=upscale_scale)
            if face_restore:
                img = _face_restore_image(img)
            if color_correct:
                img = _color_correct_image(img)

            cv2.imwrite(str(out_frame), img)

            if progress_cb and n_frames > 0:
                progress_cb(0.1 + 0.8 * (i + 1) / n_frames,
                            f"Frame {i+1}/{n_frames}")

        # Frame interpolation
        if interpolate and interpolate_factor > 1:
            if progress_cb:
                progress_cb(0.85, f"Interpolation x{interpolate_factor}...")
            _interpolate_frames(frames_out, interpolate_factor)

        # Réassemblage
        logger.info("Réassemblage vidéo...")
        if progress_cb:
            progress_cb(0.9, "Réassemblage...")

        out_fps = fps * (interpolate_factor if interpolate else 1)
        subprocess.run(
            ["ffmpeg", "-framerate", str(out_fps),
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
        "output_fps": out_fps,
        "frames": n_frames,
        "duration": duration,
    }


# ── Modules de traitement ───────────────────────────────────────────────

def _upscale_image(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Upscaling — utilise Real-ESRGAN si dispo, sinon OpenCV."""
    if _HAS_REALESRGAN:
        try:
            model = RealESRGANer(
                scale=scale,
                model_path=None,
                tile=0, tile_pad=10, pre_pad=0,
                half=False,
                device=get_device(),
            )
            # Reconstruit un RRDBNet basique
            model.model = RRDBNet(num_in_ch=3, num_out_ch=3,
                                  num_feat=64, num_block=23, num_grow_ch=32,
                                  scale=scale)
            output, _ = model.enhance(img, outscale=scale)
            return output
        except Exception as e:
            logger.warning(f"Real-ESRGAN: {e}, fallback OpenCV")

    # Fallback OpenCV — interpolation LANCZOS
    h, w = img.shape[:2]
    interp = cv2.INTER_LANCZOS4 if scale > 2 else cv2.INTER_CUBIC
    logger.info(f"  Upscale OpenCV {w}x{h} -> {w*scale}x{h*scale}")
    return cv2.resize(img, (w * scale, h * scale), interpolation=interp)


def _denoise_image(img: np.ndarray) -> np.ndarray:
    """Débruitage par NL-Means."""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def _face_restore_image(img: np.ndarray) -> np.ndarray:
    """Restauration visages — utilise OpenCV (pas de CodeFormer sans PyTorch)."""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, fw, fh) in faces:
            # Crop le visage
            face = img[y:y+fh, x:x+fw]
            # Upscale local
            face_hd = cv2.resize(face, (fw*2, fh*2), interpolation=cv2.INTER_CUBIC)
            # Denoise local
            face_hd = cv2.bilateralFilter(face_hd, 9, 75, 75)
            # Sharpen
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            face_hd = cv2.filter2D(face_hd, -1, kernel)
            # Redimensionne et replace
            face = cv2.resize(face_hd, (fw, fh), interpolation=cv2.INTER_AREA)
            img[y:y+fh, x:x+fw] = face

        n_faces = len(faces)
        if n_faces > 0:
            logger.info(f"  {n_faces} visage(s) restauré(s)")
        return img
    except Exception as e:
        logger.warning(f"Face restore: {e}")
        return img


def _color_correct_image(img: np.ndarray) -> np.ndarray:
    """Correction couleur — auto white balance + CLAHE."""
    # Auto white balance
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # Saturation boost
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) * 1.15, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def _sharpen_image(img: np.ndarray) -> np.ndarray:
    """Affûtage."""
    kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def _interpolate_frames(frames_dir: Path, factor: int = 2):
    """Interpolation de frames par blending linéaire."""
    frame_files = sorted(frames_dir.glob("*.png"))
    new_files = []

    for i in range(len(frame_files) - 1):
        img0 = cv2.imread(str(frame_files[i]))
        img1 = cv2.imread(str(frame_files[i + 1]))
        if img0 is None or img1 is None:
            continue
        for step in range(1, factor):
            alpha = step / factor
            blended = cv2.addWeighted(img0, 1 - alpha, img1, alpha, 0)
            name = f"interp_{i:06d}_{step:03d}.png"
            path = frames_dir / name
            cv2.imwrite(str(path), blended)
            new_files.append(path)

    logger.info(f"  {len(new_files)} frames interpolées ajoutées")
