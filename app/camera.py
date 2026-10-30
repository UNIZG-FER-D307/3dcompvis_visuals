from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RGBDFrame:
    rgb: np.ndarray  # uint8 [H, W, 3]
    depth_m: np.ndarray  # float32 [H, W], metres, 0 = invalid
    K: np.ndarray  # float64 [3, 3]


class Camera(Protocol):
    name: str

    def latest_rgb(self) -> np.ndarray: ...

    def capture(self) -> RGBDFrame: ...
