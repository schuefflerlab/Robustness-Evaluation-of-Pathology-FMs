from __future__ import annotations

from typing import Any
import torch
import numpy as np
from PIL import Image


class PepperNoise:
    """
    Pepper noise usable in BOTH:
      - EVA style: transform(image, target) -> (image, target)
      - torchvision style: transform(image) -> image
    Sets a fraction p of pixel locations to black (all channels).
    """
    def __init__(self, p: float):
        p = float(p)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        self.p = p

    def _apply(self, image: Any) -> Any:
        if self.p == 0.0:
            return image

        # PIL path (rare in EVA if you use ToImage, but keep for safety)
        if isinstance(image, Image.Image):
            arr = np.array(image)  # HWC uint8
            if arr.ndim == 2:
                arr = arr[..., None]
            h, w = arr.shape[:2]
            mask = (np.random.rand(h, w) < self.p)
            if arr.shape[2] == 1:
                arr[mask, 0] = 0
            else:
                arr[mask] = 0
            return Image.fromarray(arr.astype(np.uint8))

        # Tensor path (fast)
        if torch.is_tensor(image):
            x = image
            if x.ndim != 3:
                raise ValueError(f"Expected 3D tensor, got shape {tuple(x.shape)}")

            # Prefer CHW (torchvision v2 ToImage typically produces CHW tv_tensors.Image)
            if x.shape[0] in (1, 3) and x.shape[1] > 8 and x.shape[2] > 8:
                c, h, w = x.shape
                mask = (torch.rand((1, h, w), device=x.device) < self.p).expand(c, h, w)
            else:
                # fallback HWC
                h, w, c = x.shape
                mask = (torch.rand((h, w, 1), device=x.device) < self.p).expand(h, w, c)

            out = x.clone()
            if out.dtype.is_floating_point:
                out.masked_fill_(mask, 0.0)
            else:
                out.masked_fill_(mask, 0)
            return out

        raise TypeError(f"PepperNoise expects PIL.Image or torch.Tensor, got {type(image)}")

    def __call__(self, *args):
        # torchvision.Compose calls transform(image)
        if len(args) == 1:
            return self._apply(args[0])

        # EVA calls transform(image, target)
        if len(args) == 2:
            image, target = args
            return self._apply(image), target

        raise TypeError(f"PepperNoise expected 1 or 2 args, got {len(args)}")
