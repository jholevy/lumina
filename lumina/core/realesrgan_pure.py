"""Upscaling par Real-ESRGAN via PyTorch pur (sans basicsr)."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RRDBNet(nn.Module):
        """Real-ESRGAN RRDB network — implémentation minimale."""
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=4):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

            self.body = nn.ModuleList()
            for _ in range(num_block):
                self.body.append(RRDB(num_feat, num_grow_ch))

            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

            # Upsampling
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if scale == 4:
                self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

            self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        def forward(self, x):
            feat = self.conv_first(x)

            for block in self.body:
                feat = block(feat)

            feat = self.conv_body(feat)

            # Upsampling
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            if self.scale == 4:
                feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))

            out = self.lrelu(self.conv_hr(feat))
            out = self.conv_last(out)
            return out

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    REALESRGAN_PYTORCH_OK = True

except ImportError:
    REALESRGAN_PYTORCH_OK = False


def upscale_with_realesrgan(img: np.ndarray, scale: int = 2,
                            device: str = "mps") -> np.ndarray | None:
    """Upscale une image avec Real-ESRGAN via PyTorch MPS."""
    if not REALESRGAN_PYTORCH_OK:
        return None

    model_path = Path.home() / ".lumina" / "models" / "RealESRGAN_x4plus.pth"
    if not model_path.exists():
        logger.info("  Modèle Real-ESRGAN non trouvé, téléchargement...")
        import requests
        url = ("https://github.com/xinntao/Real-ESRGAN/"
               "releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        logger.info(f"  Modèle téléchargé: {model_path} ({model_path.stat().st_size // 1024**2} MB)")

    try:
        h, w = img.shape[:2]

        # Chargement du modèle
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        state = torch.load(model_path, map_location="cpu", weights_only=True)

        # Nettoyage des clés du checkpoint
        new_state = {}
        for k, v in state["params_ema"].items():
            # Enlever le préfixe "module." si présent
            k = k.replace("module.", "")
            if k.startswith("conv_body"):
                k = "conv_body." + k.split(".", 1)[1]
            new_state[k] = v

        # On charge les poids disponibles
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in new_state.items() if k in model_dict}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)

        device_t = torch.device(device)
        model = model.to(device_t).eval()

        # Préparation de l'image
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device_t)

        # Upscale par étapes (le modèle est x4, on peut faire x2 en resize)
        with torch.no_grad():
            output = model(img_t)
            if scale == 2:
                # Redimensionner le résultat x4 en x2
                output = F.interpolate(output, scale_factor=0.5,
                                       mode="bicubic", align_corners=False)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        logger.info(f"  Upscale Real-ESRGAN PyTorch ({device}) "
                    f"{w}x{h} -> {output.shape[1]}x{output.shape[0]}")
        return output

    except Exception as e:
        logger.warning(f"  Real-ESRGAN PyTorch échoué: {e}")
        return None
