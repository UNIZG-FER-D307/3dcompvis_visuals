import numpy as np
import pytest
from fastapi.testclient import TestClient

import gltf
from app import mvg
from app.server import create_app

# Slider defaults of the three tabs (app/static/mvg.js).
TRI = {"ry": 10, "tx": -0.2}
PLANE = {"rx": 25, "ry": -15, "rz": 5, "tz": 1.0}
TWO = {"ry": 10, "tx": -0.2}


class NoCamera:
    name = "none"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(NoCamera()))


def _json(client: TestClient, url: str, **params) -> dict:
    resp = client.get(url, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_mvg_script_served(client: TestClient) -> None:
    resp = client.get("/mvg.js")
    assert resp.status_code == 200 and resp.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    "url, pose, target",
    [
        ("/mvg/triangulation.json", TRI, mvg.BOOK_TARGET),
        ("/mvg/plane.json", PLANE, mvg.PLANE_TARGET),
        ("/mvg/twoview.json", TWO, mvg.PLANE_TARGET),
    ],
)
def test_default_poses_show_every_corner(client, url, pose, target) -> None:
    d = _json(client, url, **pose)
    assert all(d["selectable"]) and len(d["selectable"]) == len(target.corners)
    for view in d["views"]:
        assert len(view["squares"]) == len(target.squares)


def test_triangulation_without_noise_recovers_the_corner(client) -> None:
    d = _json(client, "/mvg/triangulation.json", **TRI, point=7)
    np.testing.assert_allclose(d["Q"], mvg.BOOK_TARGET.corners[7], atol=1e-9)
    np.testing.assert_allclose(d["Q_true"], mvg.BOOK_TARGET.corners[7])
    assert np.array(d["A"]).shape == (4, 4) and max(d["reprojection"]) < 1e-6


def test_triangulation_with_noise_is_off_but_close(client) -> None:
    d = _json(client, "/mvg/triangulation.json", **TRI, point=7, noise=2.0)
    err = np.linalg.norm(np.array(d["Q"]) - d["Q_true"])
    assert 1e-5 < err < 0.05 and d["reprojection"][0] > 1e-3


def test_triangulation_drops_a_hidden_point(client) -> None:
    d = _json(client, "/mvg/triangulation.json", tx=-0.2, ty=0.9, point=0)
    assert not d["selectable"][0] and d["point"] is None


def test_plane_needs_four_points(client) -> None:
    d = _json(client, "/mvg/plane.json", **PLANE, ids="0,10,20")
    assert "at least 4" in d["message"] and "candidates" not in d


def test_plane_four_points_give_one_valid_pose_equal_to_truth(client) -> None:
    d = _json(client, "/mvg/plane.json", **PLANE, ids="0,6,56,62")
    assert np.array(d["A"]).shape == (8, 9)
    np.testing.assert_allclose(d["H"], d["H_true"], rtol=1e-6, atol=1e-9)
    (valid,) = [c for c in d["candidates"] if c["valid"]]
    np.testing.assert_allclose(valid["R"], d["R_true"], atol=1e-9)
    np.testing.assert_allclose(valid["t"], d["t_true"], atol=1e-9)


@pytest.mark.parametrize("noise", [0.0, 3.0])
def test_plane_collinear_points_are_degenerate(client, noise: float) -> None:
    d = _json(client, "/mvg/plane.json", **PLANE, ids="0,1,2,3", noise=noise)
    assert "Degenerate" in d["message"] and "candidates" not in d  # one column


@pytest.mark.parametrize("noise", [0.0, 3.0])
def test_twoview_three_collinear_of_four_are_degenerate(client, noise) -> None:
    # 6, 31 and 56 lie on the board's diagonal through its centre.
    d = _json(client, "/mvg/twoview.json", **TWO, ids="0,6,31,56", noise=noise)
    assert "Degenerate" in d["message"] and "candidates" not in d


def test_plane_drops_duplicate_and_hidden_ids(client) -> None:
    d = _json(client, "/mvg/plane.json", **PLANE, ids="3,3,999,5")
    assert d["ids"] == [3, 5]


def test_twoview_candidates_include_the_truth(client) -> None:
    d = _json(client, "/mvg/twoview.json", **TWO, ids="0,6,31,56,62")
    np.testing.assert_allclose(d["H21"], d["H21_true"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(d["Hc"], d["H_true"], atol=1e-9)
    assert len(d["candidates"]) == 4
    truth = d["candidates"][d["truth"]]
    assert truth["valid"] and 1 <= sum(c["valid"] for c in d["candidates"]) <= 2
    np.testing.assert_allclose(truth["R"], d["R_true"], atol=1e-9)
    np.testing.assert_allclose(truth["t"], d["t_d_true"], atol=1e-9)
    np.testing.assert_allclose(truth["n"], d["n_true"], atol=1e-9)


def test_bad_ids_are_rejected(client) -> None:
    assert client.get("/mvg/plane.json", params={"ids": "a,b"}).status_code == 422


def _segments(glb: bytes) -> int:
    return sum(len(p["positions"]) // 2 for p in gltf.primitives(glb, 1))


def _glb(client: TestClient, url: str, **params) -> bytes:
    resp = client.get(url, params=params)
    assert resp.status_code == 200 and resp.headers["content-type"] == (
        "model/gltf-binary"
    )
    return resp.content


def test_triangulation_glb_adds_rays_and_markers(client) -> None:
    plain = _glb(client, "/mvg/triangulation.glb", **TRI)
    (cloud,) = gltf.primitives(plain, 0)
    assert len(cloud["positions"]) == len(mvg.target_samples(mvg.BOOK_TARGET)[0])
    assert _segments(plain) == 2 * 8  # two frustums
    picked = _glb(client, "/mvg/triangulation.glb", **TRI, point=7)
    assert _segments(picked) == 2 * 8 + 2 + 2 * 3  # rays, two crosses


def test_plane_glb_shows_both_candidates(client) -> None:
    plain = _glb(client, "/mvg/plane.glb", **PLANE)
    assert _segments(plain) == 8
    assert len(gltf.flat_triangles(plain)) == 2  # the image plane
    glb = _glb(client, "/mvg/plane.glb", **PLANE, ids="0,6,56,62")
    # Candidates, plus per picked point: the true ray, a + at the true and an x
    # at the noisy pixel on the image plane (no true -> noisy link without noise).
    assert _segments(glb) == 3 * 8 + 4 * 5
    noisy = _glb(client, "/mvg/plane.glb", **PLANE, ids="0,6,56,62", noise=1)
    assert _segments(noisy) == 3 * 8 + 4 * 6


def _point_primitives(glb: bytes) -> list[np.ndarray]:
    """Per picked point, its 6 segments [6, 2, 3] in camera (not glTF) axes."""
    flip = np.array([1, -1, -1])
    return [
        p["positions"].reshape(-1, 2, 3) * flip
        for p in gltf.primitives(glb, 1)
        if len(p["positions"]) == 2 * 6
    ]


def test_plane_glb_rays_through_true_pixels_and_noisy_marks(client) -> None:
    glb = _glb(client, "/mvg/plane.glb", **PLANE, ids="0,62", noise=3)
    R, t = mvg.rotation_matrix(25, -15, 5), np.array([0, 0, 1.0])
    centre = -R.T @ t
    points = _point_primitives(glb)
    assert len(points) == 2
    for segments, corner in zip(points, mvg.PLANE_TARGET.corners[[0, 62]]):
        ray, link = segments[0], segments[1]  # with noise, the link is second
        np.testing.assert_allclose(ray, [centre, corner], atol=1e-6)
        # The link runs from the true ray's crossing with the image plane (at
        # FRUSTUM_DEPTH in the camera) to the noisy pixel's crossing.
        true_cam, noisy_cam = link @ R.T + t
        np.testing.assert_allclose([true_cam[2], noisy_cam[2]], mvg.FRUSTUM_DEPTH)
        np.testing.assert_allclose(np.cross(true_cam, R @ corner + t), 0, atol=1e-6)
        shift_px = np.linalg.norm(true_cam - noisy_cam) / mvg.FRUSTUM_DEPTH * 615
        assert 0.1 < shift_px < 3 * 4


def test_plane_glb_points_follow_the_rainbow_colormap(client) -> None:
    d = _json(client, "/mvg/plane.json", **PLANE, ids="0,6,56,62")
    expected = mvg.rainbow(4)
    assert d["colors"] == expected.tolist()
    assert expected[0].tolist() == [128, 0, 255] and expected[-1].tolist() == [
        255,
        0,
        0,
    ]
    glb = _glb(client, "/mvg/plane.glb", **PLANE, ids="0,6,56,62")
    colours = gltf.line_colors(glb)
    for rgb in expected.tolist():
        assert tuple(rgb) in colours


def test_candidate_colours_are_distinct(client) -> None:
    d = _json(client, "/mvg/twoview.json", **TWO, ids="0,6,31,56,62")
    colours = [tuple(c["rgb"]) for c in d["candidates"]]
    assert len(set(colours)) == len(colours) == 4


def test_twoview_glb_shows_candidate_cameras_and_planes(client) -> None:
    glb = _glb(client, "/mvg/twoview.glb", **TWO, ids="0,6,31,56,62")
    # Frustums, plus per camera and picked point 5 segments (see above).
    assert _segments(glb) == 6 * 8 + 2 * 5 * 5
    # Candidate planes and the two image planes.
    assert len(gltf.flat_triangles(glb)) == 4 * 2 + 2 * 2


SCALE_IDS = "0,6,31,56,62"


def test_scale_needs_four_points(client) -> None:
    d = _json(client, "/mvg/scale.json", **TWO, ids="0,6")
    assert "at least 4" in d["message"] and "d_metric" not in d


def test_scale_without_noise_recovers_metric_pose_and_plane(client) -> None:
    d = _json(client, "/mvg/scale.json", **TWO, ids=SCALE_IDS)
    t_true = [TWO["tx"], 0, 0]
    assert d["d_metric"] == pytest.approx(d["d_true"])
    np.testing.assert_allclose(d["n_metric"], d["n_true"], atol=1e-9)
    np.testing.assert_allclose(d["R_metric"], d["R_true"], atol=1e-9)
    np.testing.assert_allclose(d["t_metric"], t_true, atol=1e-9)
    # The H21 route: the candidate chosen by the metric normal, times d.
    assert d["chosen"] == d["truth"]
    np.testing.assert_allclose(d["t_scaled"], t_true, atol=1e-9)
    # Each view's metric pose of the target, as in the Plane pose tab.
    R1, t1 = mvg.CAMERA_1
    R = mvg.rotation_matrix(0, TWO["ry"], 0)
    for pose, (Rv, tv) in zip(d["poses"], [(R1, t1), (R @ R1, R @ t1 + t_true)]):
        np.testing.assert_allclose(pose["R"], Rv, atol=1e-9)
        np.testing.assert_allclose(pose["t"], tv, atol=1e-9)


def test_scale_with_noise_is_close_and_picks_the_true_candidate(client) -> None:
    d = _json(client, "/mvg/scale.json", **TWO, ids=SCALE_IDS, noise=2)
    assert d["chosen"] == d["truth"]
    assert abs(d["d_metric"] - d["d_true"]) < 0.05
    assert np.linalg.norm(np.subtract(d["t_scaled"], [TWO["tx"], 0, 0])) < 0.05


def test_scale_glb_shows_cameras_and_d(client) -> None:
    glb = _glb(client, "/mvg/scale.glb", **TWO, ids=SCALE_IDS)
    # Camera 1, true camera 2, metric camera 2, H21-scaled camera 2, and the
    # segment of length d from camera 1 to the plane.
    assert _segments(glb) == 4 * 8 + 1
    assert len(gltf.flat_triangles(glb)) == 2  # the recovered plane
    assert _segments(_glb(client, "/mvg/scale.glb", **TWO)) == 2 * 8


@pytest.mark.parametrize("url", ["/mvg/twoview.json", "/mvg/scale.json"])
def test_two_view_target_is_parallel_to_camera_1_image_plane(client, url) -> None:
    d = _json(client, url, **TWO)
    np.testing.assert_allclose(d["n_true"], [0, 0, 1])
    assert d["d_true"] == pytest.approx(1.0)
