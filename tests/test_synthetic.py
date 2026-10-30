import numpy as np
import pytest

from app import synthetic
from app.synthetic import SyntheticCamera


@pytest.fixture(scope="module")
def cam() -> SyntheticCamera:
    return SyntheticCamera()


def _ray(K: np.ndarray, u: float, v: float) -> np.ndarray:
    return np.linalg.inv(K) @ [u, v, 1.0]


def test_frame_shape_and_name(cam: SyntheticCamera) -> None:
    frame = cam.capture()
    assert cam.name == "synthetic"
    assert frame.rgb.shape == (480, 640, 3) and frame.rgb.dtype == np.uint8
    assert frame.depth_m.shape == (480, 640) and frame.depth_m.dtype == np.float32
    np.testing.assert_array_equal(cam.latest_rgb(), frame.rgb)


def test_ball_rests_on_floor() -> None:
    n, c = synthetic.FLOOR_NORMAL, synthetic.SPHERE_CENTER
    # Floor is n·X = -h; the ball's centre is one radius above it.
    assert n @ c == pytest.approx(-synthetic.CAMERA_HEIGHT + synthetic.SPHERE_RADIUS)


def test_ray_through_ball_centre_hits_its_front(cam: SyntheticCamera) -> None:
    frame = cam.capture()
    uvw = frame.K @ synthetic.SPHERE_CENTER
    u, v = uvw[:2] / uvw[2]
    d = _ray(frame.K, round(u), round(v))
    # Nearer root of |t·d - C|² = r² (t is depth since d_z = 1).
    a, b = d @ d, -2 * d @ synthetic.SPHERE_CENTER
    c = synthetic.SPHERE_CENTER @ synthetic.SPHERE_CENTER - synthetic.SPHERE_RADIUS**2
    t = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert frame.depth_m[round(v), round(u)] == pytest.approx(t, rel=1e-5)


def test_floor_pixel_depth_is_ray_plane_intersection(cam: SyntheticCamera) -> None:
    frame = cam.capture()
    u, v = 20, 470  # bottom-left: floor, away from the ball
    d = _ray(frame.K, u, v)
    t = -synthetic.CAMERA_HEIGHT / (synthetic.FLOOR_NORMAL @ d)
    assert frame.depth_m[v, u] == pytest.approx(t, rel=1e-5)


def test_sky_has_no_depth(cam: SyntheticCamera) -> None:
    frame = cam.capture()
    assert np.all(frame.depth_m[0] == 0)  # top row looks above the horizon
    assert frame.rgb[0].any()  # but is not black
    assert np.all(frame.depth_m[frame.depth_m > 0] <= synthetic.MAX_DEPTH)
