from __future__ import annotations
from typing import Any, Sequence, Tuple
import torch
import os

class RandomRotate90:
    """
    Randomly rotate by one of {90,180,270} (or provided angles).
    Works for torch.Tensor and tv_tensors.Image (subclass of Tensor).
    For non-tensor (e.g., PIL), it falls back to PIL rotate.
    Returns (image, target) to match EVA transform signature patterns you use for gamma.
    """
    def __init__(self, angles: Sequence[int] = (90, 180, 270), p: float = 1.0):
        self.angles = [int(a) for a in angles]
        for a in self.angles:
            if a % 90 != 0:
                raise ValueError(f"Only multiples of 90 supported, got {a}")
        self.p = float(p)
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")

        self._dbg_count = 0  # for optional debug prints

    def __call__(self, image: Any, target: Any = None) -> Tuple[Any, Any]:
        if self.p < 1.0 and torch.rand(()) > self.p:
            return image, target

        idx = int(torch.randint(0, len(self.angles), ()).item())
        angle = self.angles[idx]
        k = (angle // 90) % 4

        # Optional debug (enable by env var if you want)
        debug = os.environ.get("ROTATE_DEBUG", "0")
        if self._dbg_count < 5 and debug not in ("0", "", "false", "False", "no", "No"):
            print(f"[RandomRotate90] angle={angle} k={k} type={type(image)} shape={getattr(image, 'shape', None)}")
            self._dbg_count += 1

        # Tensor / tv_tensor path (most common in torchvision v2 pipelines)
        if isinstance(image, torch.Tensor):
            # If CHW: rotate over H,W = dims 1,2. If HWC: dims 0,1.
            if image.ndim == 3:
                # heuristics: if first dim is 1/3/4 assume CHW
                if image.shape[0] in (1, 3, 4):
                    image = torch.rot90(image, k=k, dims=(1, 2))
                else:
                    image = torch.rot90(image, k=k, dims=(0, 1))
            elif image.ndim == 2:
                image = torch.rot90(image, k=k, dims=(0, 1))
            return image, target

        # PIL fallback
        try:
            image = image.rotate(angle, expand=False)
        except Exception:
            # if some unexpected type, just return unchanged (but this should not happen)
            return image, target

        return image, target
