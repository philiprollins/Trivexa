"""Foreground extraction with BiRefNet (vendored model code + local weights)."""

import logging
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from ..config import MODELS_DIR
from ..vendor.birefnet.BiRefNet_config import BiRefNetConfig
from ..vendor.birefnet.birefnet import BiRefNet

log = logging.getLogger(__name__)

WEIGHTS = MODELS_DIR / "background_removal" / "birefnet.safetensors"
INPUT_SIZE = 1024
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BackgroundRemover:
    """Kept in system RAM and moved to the GPU only while running, so it doesn't compete with diffusion models."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model: BiRefNet | None = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        return WEIGHTS.is_file()

    def _load(self) -> BiRefNet:
        if self.model is None:
            log.info("Loading BiRefNet from %s", WEIGHTS)
            model = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
            model.load_state_dict(load_file(str(WEIGHTS)))
            self.model = model.eval().half()
        return self.model

    @torch.inference_mode()
    def remove(self, image: Image.Image) -> Image.Image:
        """Return the image as RGBA with the background made transparent."""
        with self._lock:
            model = self._load().to(self.device)
            try:
                rgb = image.convert("RGB")
                x = torch.from_numpy(np.asarray(rgb.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR))).permute(2, 0, 1)
                x = (x.unsqueeze(0).float() / 255 - _MEAN) / _STD
                pred = model(x.to(self.device, torch.float16))[-1].sigmoid().float()
                mask = F.interpolate(pred, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)[0, 0]
                alpha = Image.fromarray((mask.clamp(0, 1) * 255).byte().cpu().numpy(), mode="L")
            finally:
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        out = rgb.copy()
        out.putalpha(alpha)
        return out
