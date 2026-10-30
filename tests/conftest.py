import numpy as np
import pytest

from app.camera import RGBDFrame

H, W = 48, 64


@pytest.fixture
def K() -> np.ndarray:
    return np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def frame(K: np.ndarray) -> RGBDFrame:
    """Tilted plane with a nearer box in the middle and a hole of invalid depth."""
    v, u = np.mgrid[0:H, 0:W]
    depth = (2.0 + 0.01 * u).astype(np.float32)
    depth[18:30, 24:40] = 1.0
    depth[0:4, 0:4] = 0.0
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return RGBDFrame(rgb=rgb, depth_m=depth, K=K)
