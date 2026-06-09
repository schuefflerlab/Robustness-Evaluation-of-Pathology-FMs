from __future__ import annotations

from typing import Any, Tuple
from torchvision.transforms import functional as F

class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1.0):
        gamma = float(gamma)
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.gamma = gamma
        self.gain = float(gain)

    def _apply(self, image: Any) -> Any:
        return F.adjust_gamma(image, gamma=self.gamma, gain=self.gain)

    def __call__(self, *args):
        if len(args) == 1:
            return self._apply(args[0])
        if len(args) == 2:
            image, target = args
            return self._apply(image), target
        raise TypeError(f"Expected 1 or 2 args, got {len(args)}")