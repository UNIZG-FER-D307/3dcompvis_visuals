"""Synthetic RGBD camera: a ball resting on a checkerboard floor, ray-cast once."""

import numpy as np

from app.camera import RGBDFrame

# Scene in camera coordinates (x right, y down, z forward), D435-like intrinsics.
WIDTH, HEIGHT = 640, 480
K = np.array([[615.0, 0.0, 318.0], [0.0, 615.0, 244.0], [0.0, 0.0, 1.0]])
CAMERA_HEIGHT = 0.5  # m above the floor
PITCH = np.radians(10.0)  # camera tilted down
FLOOR_NORMAL = np.array([0.0, -np.cos(PITCH), -np.sin(PITCH)])  # world "up"
FLOOR_FORWARD = np.array([0.0, -np.sin(PITCH), np.cos(PITCH)])  # along the floor
SPHERE_RADIUS = 0.25
SPHERE_CENTER = 1.4 * FLOOR_FORWARD + (SPHERE_RADIUS - CAMERA_HEIGHT) * FLOOR_NORMAL
MAX_DEPTH = 6.0  # floor beyond this is treated as sky (no depth)
CHECKER = 0.25  # m

SKY_RGB = np.array([170, 195, 225])
FLOOR_RGB = np.array([[200, 200, 190], [90, 110, 90]])
BALL_RGB = np.array([255, 140, 40])
LIGHT = np.array([0.4, -1.0, -0.3]) / np.linalg.norm([0.4, -1.0, -0.3])


def _render() -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH]
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).astype(np.float64)
    d = pix @ np.linalg.inv(K).T  # rays with d_z = 1, so t is depth
    rgb = np.tile(SKY_RGB, (len(d), 1)).astype(np.float64)
    depth = np.zeros(len(d))

    # Floor: n·(t d) = -h.
    nd = d @ FLOOR_NORMAL
    with np.errstate(divide="ignore"):
        t_floor = np.where(nd < 0, -CAMERA_HEIGHT / nd, np.inf)
    floor = t_floor <= MAX_DEPTH
    hit = t_floor[floor, None] * d[floor]
    cell = np.floor(hit[:, 0] / CHECKER) + np.floor(hit @ FLOOR_FORWARD / CHECKER)
    rgb[floor] = FLOOR_RGB[cell.astype(np.int64) % 2]
    depth[floor] = t_floor[floor]

    # Ball: nearer root of |t d - C|² = r², drawn where it is in front of the floor.
    a = np.einsum("ij,ij->i", d, d)
    b = -2 * d @ SPHERE_CENTER
    c = SPHERE_CENTER @ SPHERE_CENTER - SPHERE_RADIUS**2
    disc = b * b - 4 * a * c
    t_ball = np.full(len(d), np.inf)
    ok = disc >= 0
    t_ball[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
    ball = (
        np.isfinite(t_ball) & (t_ball > 0) & (t_ball < np.where(floor, t_floor, np.inf))
    )
    normal = (t_ball[ball, None] * d[ball] - SPHERE_CENTER) / SPHERE_RADIUS
    shade = 0.25 + 0.75 * np.clip(normal @ LIGHT, 0, 1)
    rgb[ball] = shade[:, None] * BALL_RGB
    depth[ball] = t_ball[ball]

    return (
        rgb.round().astype(np.uint8).reshape(HEIGHT, WIDTH, 3),
        depth.astype(np.float32).reshape(HEIGHT, WIDTH),
    )


class SyntheticCamera:
    """Static ball-on-floor scene used when no RealSense is available."""

    name = "synthetic"

    def __init__(self) -> None:
        rgb, depth = _render()
        self._frame = RGBDFrame(rgb=rgb, depth_m=depth, K=K.copy())

    def latest_rgb(self) -> np.ndarray:
        return self._frame.rgb

    def capture(self) -> RGBDFrame:
        return self._frame
