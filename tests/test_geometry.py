import io

import numpy as np
import pytest
import trimesh
from matplotlib import colormaps

import gltf
from app.camera import RGBDFrame
from app.geometry import (
    backproject,
    colorize_depth,
    epipolar_plane,
    epipoles,
    essential_matrix,
    frustum_segments,
    fundamental_matrix,
    project,
    rotation_matrix,
    skew,
    splat_sizes,
    to_glb,
    warp,
)


def _cloud(frame: RGBDFrame) -> tuple[np.ndarray, np.ndarray]:
    points, idx = backproject(frame.depth_m, frame.K)
    return points, frame.rgb.reshape(-1, 3)[idx]


def test_backproject_skips_invalid_depth(frame: RGBDFrame) -> None:
    points, idx = backproject(frame.depth_m, frame.K)
    assert len(points) == len(idx) == np.count_nonzero(frame.depth_m > 0)
    np.testing.assert_allclose(points[:, 2], frame.depth_m.reshape(-1)[idx])


def test_roundtrip_with_original_K(frame: RGBDFrame) -> None:
    h, w = frame.depth_m.shape
    rgb, depth = project(*_cloud(frame), frame.K, h, w)
    valid = frame.depth_m > 0
    np.testing.assert_array_equal(rgb[valid], frame.rgb[valid])
    np.testing.assert_allclose(depth, frame.depth_m, rtol=1e-6)


def test_doubling_focal_zooms_about_principal_point(frame: RGBDFrame) -> None:
    h, w = frame.depth_m.shape
    K2 = frame.K.copy()
    K2[0, 0] *= 2
    K2[1, 1] *= 2
    rgb, _ = project(*_cloud(frame), K2, h, w)
    cx, cy = int(frame.K[0, 2]), int(frame.K[1, 2])
    # Pixel at offset (+5, +3) from the principal point moves to offset (+10, +6).
    np.testing.assert_array_equal(rgb[cy + 6, cx + 10], frame.rgb[cy + 3, cx + 5])
    np.testing.assert_array_equal(rgb[cy, cx], frame.rgb[cy, cx])


def test_shifting_cx_shifts_image(frame: RGBDFrame) -> None:
    h, w = frame.depth_m.shape
    K2 = frame.K.copy()
    K2[0, 2] += 7
    rgb, depth = project(*_cloud(frame), K2, h, w)
    np.testing.assert_array_equal(rgb[10:40, 17:50], frame.rgb[10:40, 10:43])
    assert np.all(depth[:, :7] == 0)


def test_skew_shears_rows_by_y(frame: RGBDFrame) -> None:
    h, w = frame.depth_m.shape
    K2 = frame.K.copy()
    K2[0, 1] = 0.25 * K2[0, 0]  # u shifts by 0.25 * (v - cy) on the plane
    rgb, _ = project(*_cloud(frame), K2, h, w)
    cy = int(frame.K[1, 2])
    np.testing.assert_array_equal(rgb[cy + 8, 22], frame.rgb[cy + 8, 20])
    np.testing.assert_array_equal(rgb[cy - 8, 18], frame.rgb[cy - 8, 20])


def test_zbuffer_keeps_nearest() -> None:
    K = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]])
    points = np.array([[0.0, 0, 1.0], [0.0, 0, 3.0], [0.0, 0, 2.0]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    rgb, depth = project(points, colors, K, 10, 10)
    np.testing.assert_array_equal(rgb[5, 5], [255, 0, 0])
    assert depth[5, 5] == 1.0


def test_drops_points_behind_or_outside() -> None:
    K = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]])
    points = np.array([[0.0, 0, -1.0], [100.0, 0, 1.0], [0.0, 0, 0.0]])
    colors = np.full((3, 3), 255, dtype=np.uint8)
    rgb, depth = project(points, colors, K, 10, 10)
    assert not rgb.any() and not depth.any()


def test_colorize_depth_uses_plasma_and_black_for_invalid() -> None:
    depth = np.array([[0.0, 1.0, 3.0]], dtype=np.float32)
    out = colorize_depth(depth, vmin=1.0, vmax=3.0)
    assert out.dtype == np.uint8 and out.shape == (1, 3, 3)
    plasma = colormaps["plasma"]
    np.testing.assert_array_equal(out[0, 0], [0, 0, 0])
    np.testing.assert_allclose(out[0, 1], np.array(plasma(0.0)[:3]) * 255, atol=1)
    np.testing.assert_allclose(out[0, 2], np.array(plasma(1.0)[:3]) * 255, atol=1)


def test_to_glb_flips_axes() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    scene = trimesh.load(io.BytesIO(to_glb(points, colors)), file_type="glb")
    cloud = next(iter(scene.geometry.values()))
    np.testing.assert_allclose(cloud.vertices, points * [1, -1, -1], rtol=1e-6)


def test_to_glb_subsamples_large_clouds() -> None:
    points = np.random.default_rng(0).random((1000, 3))
    colors = np.zeros((1000, 3), dtype=np.uint8)
    scene = trimesh.load(
        io.BytesIO(to_glb(points, colors, max_points=100)), file_type="glb"
    )
    assert len(next(iter(scene.geometry.values())).vertices) == 100


def test_warp_with_original_K_is_identity(frame: RGBDFrame) -> None:
    rgb, depth = warp(frame.rgb, frame.depth_m, frame.K, frame.K)
    valid = frame.depth_m > 0
    np.testing.assert_array_equal(rgb[valid], frame.rgb[valid])
    assert not rgb[~valid].any()
    np.testing.assert_allclose(depth, frame.depth_m, rtol=1e-6)


def test_warp_zoom_has_no_holes(frame: RGBDFrame) -> None:
    K2 = frame.K.copy()
    K2[0, 0] *= 2
    K2[1, 1] *= 2
    rgb, depth = warp(frame.rgb, frame.depth_m, frame.K, K2)
    assert np.all(depth > 0)
    cx, cy = int(frame.K[0, 2]), int(frame.K[1, 2])
    np.testing.assert_array_equal(rgb[cy + 6, cx + 10], frame.rgb[cy + 3, cx + 5])


def test_warp_half_pixel_shift_averages_neighbours() -> None:
    K = np.array([[10.0, 0, 1], [0, 10.0, 0], [0, 0, 1]])
    rgb = np.array([[[0, 0, 0], [100, 50, 10], [200, 150, 30]]], dtype=np.uint8)
    depth = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    K2 = K.copy()
    K2[0, 2] -= 0.5  # output pixel u samples source u + 0.5
    out_rgb, out_depth = warp(rgb, depth, K, K2)
    np.testing.assert_array_equal(out_rgb[0, 0], [50, 25, 5])
    np.testing.assert_array_equal(out_rgb[0, 1], [150, 100, 20])
    np.testing.assert_allclose(out_depth[0, :2], [1.5, 3.0])
    assert out_depth[0, 2] == 4.0  # right neighbour is outside but has weight 0.5


def test_warp_ignores_invalid_neighbours() -> None:
    K = np.array([[10.0, 0, 1], [0, 10.0, 0], [0, 0, 1]])
    rgb = np.array([[[90, 90, 90], [10, 10, 10], [0, 0, 0]]], dtype=np.uint8)
    depth = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
    K2 = K.copy()
    K2[0, 2] -= 0.5
    out_rgb, out_depth = warp(rgb, depth, K, K2)
    # Pixel 0 samples between a valid and an invalid pixel: only the valid one counts.
    np.testing.assert_array_equal(out_rgb[0, 0], [90, 90, 90])
    assert out_depth[0, 0] == 3.0
    # Pixel 1 samples between two invalid pixels: empty.
    assert not out_rgb[0, 1].any() and out_depth[0, 1] == 0


def test_warp_outside_source_is_empty(frame: RGBDFrame) -> None:
    K2 = frame.K.copy()
    K2[0, 2] += 10
    rgb, depth = warp(frame.rgb, frame.depth_m, frame.K, K2)
    assert not rgb[:, :10].any() and not depth[:, :10].any()


def test_warp_identity_keeps_invalid_pixels_empty_with_real_K(
    frame: RGBDFrame,
) -> None:
    # Non-round intrinsics (from a D435) make K @ inv(K) inexact.
    K = np.array(
        [
            [615.0182495117188, 0, 31.006012],
            [0, 615.2573852539062, 24.068741],
            [0, 0, 1],
        ]
    )
    depth = frame.depth_m.copy()
    depth[::3, ::2] = 0.0  # many invalid pixels next to valid ones
    rgb, out_depth = warp(frame.rgb, depth, K, K)
    valid = depth > 0
    assert not out_depth[~valid].any() and not rgb[~valid].any()
    np.testing.assert_array_equal(rgb[valid], frame.rgb[valid])


def test_frustum_segments_reach_backprojected_corners() -> None:
    K = np.array([[100.0, 20.0, 30.0], [0, 80.0, 25.0], [0, 0, 1]])
    seg = frustum_segments(K, 64, 48, depth=2.0)
    assert seg.shape == (8, 2, 3)
    corners = [
        np.linalg.inv(K) @ [u, v, 1] * 2.0
        for u, v in [(0, 0), (64, 0), (64, 48), (0, 48)]
    ]
    # 4 rays from the camera centre, then the 4 edges of the far rectangle.
    np.testing.assert_allclose(seg[:4, 0], 0)
    np.testing.assert_allclose(seg[:4, 1], corners)
    np.testing.assert_allclose(seg[4:, 0], corners)
    np.testing.assert_allclose(seg[4:, 1], np.roll(corners, -1, axis=0))


def test_to_glb_adds_coloured_frustum_lines() -> None:
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    seg = frustum_segments(K, 64, 48, depth=1.0)
    glb = to_glb(
        np.ones((3, 3)),
        np.zeros((3, 3), np.uint8),
        frustums=[(seg, (255, 255, 255)), (seg, (0, 255, 255))],
    )
    lines = gltf.primitives(glb, mode=1)
    assert [len(prim["positions"]) for prim in lines] == [16, 16]
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    paths = [g for g in scene.geometry.values() if isinstance(g, trimesh.path.Path3D)]
    flipped = seg.reshape(-1, 3) * [1, -1, -1]
    assert any(np.allclose(np.sort(p.vertices, 0), np.sort(flipped, 0)) for p in paths)


def test_to_glb_image_plane_is_textured_quad(frame: RGBDFrame) -> None:
    h, w = frame.depth_m.shape
    corners = frustum_segments(frame.K, w, h, depth=0.5)[:4, 1]  # TL, TR, BR, BL
    glb = to_glb(
        np.ones((3, 3)), np.zeros((3, 3), np.uint8), image_plane=(corners, frame.rgb)
    )
    positions, uv, material, image = gltf.textured_quad(glb)
    np.testing.assert_allclose(positions, corners * [1, -1, -1], rtol=1e-6)
    # glTF UV origin is the image's top-left pixel.
    np.testing.assert_array_equal(uv, [[0, 0], [1, 0], [1, 1], [0, 1]])
    np.testing.assert_array_equal(image, frame.rgb)
    assert material["alphaMode"] == "BLEND" and material["doubleSided"]
    assert 0 < material["pbrMetallicRoughness"]["baseColorFactor"][3] < 1


def _plane_frame(K: np.ndarray, z: float = 2.0) -> RGBDFrame:
    rgb = np.random.default_rng(1).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    return RGBDFrame(rgb=rgb, depth_m=np.full((48, 64), z, np.float32), K=K)


def test_rotation_matrix_is_a_rotation() -> None:
    R = rotation_matrix(10, -20, 30)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(
        rotation_matrix(0, 0, 90) @ [1, 0, 0], [0, 1, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        rotation_matrix(90, 0, 0) @ [0, 1, 0], [0, 0, 1], atol=1e-12
    )
    # R = Rz @ Ry @ Rx
    np.testing.assert_allclose(
        rotation_matrix(10, 20, 30),
        rotation_matrix(0, 0, 30)
        @ rotation_matrix(0, 20, 0)
        @ rotation_matrix(10, 0, 0),
    )


def test_translation_shifts_plane_by_fx_tx_over_z(K: np.ndarray) -> None:
    frame = _plane_frame(K, z=2.0)
    points, colors = _cloud(frame)
    rgb, _ = project(points + [0.2, 0, 0], colors, K, 48, 64)  # fx·tx/z = 5 px
    np.testing.assert_array_equal(rgb[:, 5:], frame.rgb[:, :-5])
    assert not rgb[:, :5].any()


def test_translation_gives_parallax(frame: RGBDFrame) -> None:
    points, colors = _cloud(frame)
    rgb, _ = project(points + [0.1, 0, 0], colors, frame.K, 48, 64)
    # Box at 1 m moves 50·0.1/1 = 5 px; the plane at ~2.1 m only ~2.4 px.
    np.testing.assert_array_equal(rgb[24, 37], frame.rgb[24, 32])
    np.testing.assert_array_equal(rgb[40, 12], frame.rgb[40, 10])


def test_project_without_sizes_is_unchanged(frame: RGBDFrame) -> None:
    points, colors = _cloud(frame)
    ones = np.ones(len(points), dtype=np.int64)
    for a, b in zip(
        project(points, colors, frame.K, 48, 64),
        project(points, colors, frame.K, 48, 64, sizes=ones),
    ):
        np.testing.assert_array_equal(a, b)


def test_splatting_fills_holes_when_moving_closer(K: np.ndarray) -> None:
    frame = _plane_frame(K, z=2.0)
    points, colors = _cloud(frame)
    closer = points + [0, 0, -1.0]  # plane now at 1 m: 2x magnification
    _, depth = project(closer, colors, K, 48, 64)
    centre = (slice(14, 34), slice(22, 42))
    assert (depth[centre] == 0).any()
    sizes = splat_sizes(K, K, points[:, 2], closer[:, 2])
    _, filled = project(closer, colors, K, 48, 64, sizes=sizes)
    assert (filled[centre] > 0).all()


def test_splatting_keeps_nearest_in_front() -> None:
    K = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]])
    points = np.array([[0.0, 0, 1.0], [0.1, 0, 2.0]])  # near at (5,5), far at (5.5→6,5)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    rgb, _ = project(points, colors, K, 10, 10, sizes=np.array([3, 3]))
    np.testing.assert_array_equal(rgb[5, 5], [255, 0, 0])
    np.testing.assert_array_equal(rgb[5, 6], [255, 0, 0])  # near splat wins overlap
    np.testing.assert_array_equal(rgb[5, 7], [0, 255, 0])


def test_splat_sizes_track_magnification(K: np.ndarray) -> None:
    K2 = K.copy()
    K2[0, 0] *= 1.5
    sizes = splat_sizes(
        K, K2, np.array([2.0, 2.0, 2.0, 2.0]), np.array([2.0, 1.0, 4.0, 0.1])
    )
    np.testing.assert_array_equal(sizes, [2, 3, 1, 8])  # 1.5, 3, 0.75→1, 30→cap 8


@pytest.fixture
def two_views(K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """R, t of view 2 and matching pixels x1, x2 (homogeneous) of points seen by both."""
    R, t = rotation_matrix(5, -10, 3), np.array([-0.3, 0.05, 0.1])
    X = np.random.default_rng(2).uniform([-0.5, -0.4, 1.5], [0.5, 0.4, 3.0], (20, 3))
    x1 = X @ K.T
    x2 = (X @ R.T + t) @ K.T
    return R, t, x1 / x1[:, 2:], x2 / x2[:, 2:]


def test_skew_is_cross_product() -> None:
    t, x = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.7, -1.1])
    np.testing.assert_allclose(skew(t) @ x, np.cross(t, x))


def test_essential_and_fundamental_constraints(K: np.ndarray, two_views) -> None:
    R, t, x1, x2 = two_views
    E = essential_matrix(R, t)
    np.testing.assert_allclose(E, skew(t) @ R)
    Kinv = np.linalg.inv(K)
    n1, n2 = x1 @ Kinv.T, x2 @ Kinv.T
    np.testing.assert_allclose(np.einsum("ij,jk,ik->i", n2, E, n1), 0, atol=1e-12)
    F = fundamental_matrix(K, R, t)
    np.testing.assert_allclose(np.einsum("ij,jk,ik->i", x2, F, x1), 0, atol=1e-9)


def test_epipoles_are_null_vectors_of_F(K: np.ndarray, two_views) -> None:
    R, t, _, _ = two_views
    F = fundamental_matrix(K, R, t)
    e1, e2 = epipoles(K, R, t)
    np.testing.assert_allclose(F @ e1, 0, atol=1e-9)
    np.testing.assert_allclose(F.T @ e2, 0, atol=1e-9)
    np.testing.assert_allclose(e1, K @ (-R.T @ t))  # image of C2 in view 1
    np.testing.assert_allclose(e2, K @ t)  # image of C1 in view 2


@pytest.mark.parametrize("view", [1, 2])
def test_epipolar_plane_contains_centres_ray_and_match(
    K: np.ndarray, two_views, view: int
) -> None:
    R, t, x1, x2 = two_views
    x = (x1 if view == 1 else x2)[0]
    tri = epipolar_plane(K, R, t, view, x[0], x[1], depth=4.0)
    C1, C2, X = tri
    np.testing.assert_allclose(C1, 0)
    np.testing.assert_allclose(C2, -R.T @ t)
    # X is on the hovered ray, at the given depth in that view's camera frame.
    X_cam = X if view == 1 else R @ X + t
    assert X_cam[2] == pytest.approx(4.0)
    uvw = K @ X_cam
    np.testing.assert_allclose(uvw[:2] / uvw[2], x[:2], atol=1e-9)
    # The true 3D point of this correspondence lies on the plane.
    normal = np.cross(C2 - C1, X - C1)
    X_true = np.linalg.inv(K) @ x1[0] * 1.0  # direction from C1; any depth works
    assert abs(normal @ X_true) / np.linalg.norm(normal) < 1e-9
