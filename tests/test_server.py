import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import gltf
from app.camera import RGBDFrame
from app.geometry import colorize_depth
from app.server import create_app, main, mjpeg_frames


class FakeCamera:
    name = "fake"

    def __init__(self, frame: RGBDFrame) -> None:
        self.frame = frame

    def latest_rgb(self) -> np.ndarray:
        return self.frame.rgb

    def capture(self) -> RGBDFrame:
        return self.frame


@pytest.fixture
def client(frame: RGBDFrame) -> TestClient:
    return TestClient(create_app(FakeCamera(frame)))


def _png(resp) -> np.ndarray:
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    return np.asarray(Image.open(io.BytesIO(resp.content)))


def _k_params(K: np.ndarray) -> dict[str, float]:
    return {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "s": K[0, 1]}


def test_index_served(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200 and "<html" in resp.text.lower()


def test_endpoints_before_capture_return_409(client: TestClient, K) -> None:
    assert client.get("/render.png", params=_k_params(K)).status_code == 409
    assert client.get("/capture/rgb.png").status_code == 409


def test_capture_returns_intrinsics(client: TestClient, frame: RGBDFrame) -> None:
    resp = client.post("/capture")
    assert resp.status_code == 200
    body = resp.json()
    np.testing.assert_allclose(body["K"], frame.K)
    assert (body["height"], body["width"]) == frame.depth_m.shape


def test_capture_assets(client: TestClient, frame: RGBDFrame) -> None:
    client.post("/capture")
    np.testing.assert_array_equal(_png(client.get("/capture/rgb.png")), frame.rgb)
    assert _png(client.get("/capture/depth.png")).shape == frame.rgb.shape
    glb = client.get("/capture/cloud.glb")
    assert glb.status_code == 200
    assert glb.headers["content-type"] == "model/gltf-binary"
    assert glb.content[:4] == b"glTF"


def test_render_with_original_K_matches_capture(
    client: TestClient, frame: RGBDFrame
) -> None:
    client.post("/capture")
    params = _k_params(frame.K)
    valid = frame.depth_m > 0
    rgb = _png(client.get("/render.png", params=params))
    np.testing.assert_array_equal(rgb[valid], frame.rgb[valid])
    depth = _png(client.get("/render.png", params={**params, "kind": "depth"}))
    captured = _png(client.get("/capture/depth.png"))
    np.testing.assert_array_equal(depth, captured)


def test_stream_yields_jpeg(frame: RGBDFrame) -> None:
    chunk = next(mjpeg_frames(FakeCamera(frame)))
    head, jpeg = chunk.split(b"\r\n\r\n", 1)
    assert head == b"--frame\r\nContent-Type: image/jpeg"
    assert Image.open(io.BytesIO(jpeg)).size == frame.rgb.shape[1::-1]


def test_render_interp_with_original_K_matches_capture(
    client: TestClient, frame: RGBDFrame
) -> None:
    client.post("/capture")
    params = {**_k_params(frame.K), "interp": "true"}
    valid = frame.depth_m > 0
    rgb = _png(client.get("/render.png", params=params))
    np.testing.assert_array_equal(rgb[valid], frame.rgb[valid])


def test_render_interp_fills_zoom_holes(client: TestClient, frame: RGBDFrame) -> None:
    client.post("/capture")
    zoom = {**_k_params(frame.K), "fx": 3 * frame.K[0, 0], "fy": 3 * frame.K[1, 1]}
    holes = (_png(client.get("/render.png", params=zoom)) == 0).all(-1).sum()
    filled = _png(client.get("/render.png", params={**zoom, "interp": "true"}))
    assert holes > 0 and not (filled == 0).all(-1).any()


def test_capture_returns_depth_range(client: TestClient, frame: RGBDFrame) -> None:
    body = client.post("/capture").json()
    valid = frame.depth_m[frame.depth_m > 0]
    np.testing.assert_allclose(
        [body["depth_min"], body["depth_max"]], np.percentile(valid, [2, 98])
    )


def test_colorbar_matches_depth_colormap(client: TestClient) -> None:
    bar = _png(client.get("/colorbar.png"))
    assert bar.shape == (1, 256, 3)
    ramp = colorize_depth(np.linspace(1.0, 2.0, 256)[None], vmin=1.0, vmax=2.0)
    np.testing.assert_array_equal(bar, ramp)


def test_cloud_has_capture_frustum_and_optional_edited_one(
    client: TestClient, frame: RGBDFrame
) -> None:
    client.post("/capture")
    assert len(gltf.primitives(client.get("/capture/cloud.glb").content, 1)) == 1
    params = {**_k_params(frame.K), "fx": 2 * frame.K[0, 0]}
    glb = client.get("/capture/cloud.glb", params=params).content
    assert len(gltf.primitives(glb, mode=1)) == 2


def test_cloud_image_plane_shows_capture_by_default(
    client: TestClient, frame: RGBDFrame
) -> None:
    body = client.post("/capture").json()
    glb = client.get("/capture/cloud.glb").content
    positions, uv, _, image = gltf.textured_quad(glb)
    np.testing.assert_array_equal(image, frame.rgb)
    # Behind the optical centre, like a physical sensor (glTF looks down -z) ...
    np.testing.assert_allclose(positions[:, 2], body["image_plane_depth"], rtol=1e-5)
    # ... and inverted: the image's top-left pixel (glTF UV 0,0) lands at the
    # sensor's bottom-right, i.e. camera +x, +y (glTF +x, -y).
    top_left = positions[np.flatnonzero((uv == 0).all(1))[0]]
    assert top_left[0] > 0 and top_left[1] < 0


def test_capture_returns_image_plane_depth(client: TestClient) -> None:
    body = client.post("/capture").json()
    assert body["image_plane_depth"] == pytest.approx(0.3 * body["depth_max"])


def test_cloud_rays_continue_through_centre_to_image_plane(
    client: TestClient, frame: RGBDFrame
) -> None:
    d = client.post("/capture").json()["image_plane_depth"]
    params = {**_k_params(frame.K), "fx": 1.5 * frame.K[0, 0]}
    glb = client.get("/capture/cloud.glb", params=params).content
    positions, _, _, _ = gltf.textured_quad(glb)
    _, edited = gltf.primitives(glb, mode=1)
    # The edited camera's lines reach the plane's corners behind the centre.
    for corner in positions:
        assert np.isclose(edited["positions"], corner, atol=1e-5).all(1).any()
    assert np.isclose(edited["positions"][:, 2].max(), 1.5 * d, rtol=1e-5)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_cloud_image_plane_distance_follows_fx(
    client: TestClient, frame: RGBDFrame, scale: float
) -> None:
    # The sensor sits at f = fx * pixel width behind the centre; the pixel width is
    # fixed, so the plane moves with fx and keeps its width.
    d = client.post("/capture").json()["image_plane_depth"]
    base, _, _, _ = gltf.textured_quad(client.get("/capture/cloud.glb").content)
    params = {**_k_params(frame.K), "fx": scale * frame.K[0, 0]}
    glb = client.get("/capture/cloud.glb", params=params).content
    positions, _, _, _ = gltf.textured_quad(glb)
    np.testing.assert_allclose(positions[:, 2], scale * d, rtol=1e-5)
    np.testing.assert_allclose(np.ptp(positions[:, 0]), np.ptp(base[:, 0]), rtol=1e-5)


def test_cloud_image_plane_fy_changes_pixel_height_not_distance(
    client: TestClient, frame: RGBDFrame
) -> None:
    d = client.post("/capture").json()["image_plane_depth"]
    base, _, _, _ = gltf.textured_quad(client.get("/capture/cloud.glb").content)
    params = {**_k_params(frame.K), "fy": 2 * frame.K[1, 1]}
    glb = client.get("/capture/cloud.glb", params=params).content
    positions, _, _, _ = gltf.textured_quad(glb)
    np.testing.assert_allclose(positions[:, 2], d, rtol=1e-5)
    np.testing.assert_allclose(
        np.ptp(positions[:, 1]), np.ptp(base[:, 1]) / 2, rtol=1e-5
    )


@pytest.mark.parametrize("interp", ["false", "true"])
def test_cloud_image_plane_shows_edited_render(
    client: TestClient, frame: RGBDFrame, interp: str
) -> None:
    client.post("/capture")
    params = {**_k_params(frame.K), "fx": 2 * frame.K[0, 0], "interp": interp}
    glb = client.get("/capture/cloud.glb", params=params).content
    _, _, _, image = gltf.textured_quad(glb)
    expected = _png(client.get("/render.png", params=params))
    np.testing.assert_array_equal(image, expected)


def test_camera_endpoint_reports_name(frame: RGBDFrame) -> None:
    client = TestClient(create_app(FakeCamera(frame)))
    assert client.get("/camera").json() == {"name": "fake"}


def test_main_falls_back_to_synthetic_without_realsense(monkeypatch) -> None:
    import app.realsense

    def no_device() -> None:
        raise RuntimeError("No device connected")

    monkeypatch.setattr(app.realsense, "RealSenseCamera", no_device)
    client = TestClient(main())
    assert client.get("/camera").json() == {"name": "synthetic"}
    assert client.post("/capture").status_code == 200


def test_render_with_zero_extrinsics_is_unchanged(
    client: TestClient, frame: RGBDFrame
) -> None:
    client.post("/capture")
    params = _k_params(frame.K)
    zero = {"rx": 0, "ry": 0, "rz": 0, "tx": 0, "ty": 0, "tz": 0}
    for interp in ("false", "true"):
        a = _png(client.get("/render.png", params={**params, "interp": interp}))
        b = _png(client.get("/render.png", params={**params, **zero, "interp": interp}))
        np.testing.assert_array_equal(a, b)


def test_render_extrinsics_interp_splats_holes(
    client: TestClient, frame: RGBDFrame
) -> None:
    client.post("/capture")
    params = {**_k_params(frame.K), "tz": -0.5, "ry": 5}  # move closer, turn
    holes = (_png(client.get("/render.png", params=params)) == 0).all(-1).sum()
    splat = _png(client.get("/render.png", params={**params, "interp": "true"}))
    assert (splat == 0).all(-1).sum() < holes / 4


def test_cloud_edited_frustum_and_plane_follow_extrinsics(
    client: TestClient, frame: RGBDFrame
) -> None:
    from app.geometry import frustum_segments, rotation_matrix

    depth_max = client.post("/capture").json()["depth_max"]
    ext = {"rx": 10, "ry": -20, "rz": 30, "tx": 0.3, "ty": -0.1, "tz": 0.2}
    glb = client.get("/capture/cloud.glb", params={**_k_params(frame.K), **ext}).content
    R = rotation_matrix(ext["rx"], ext["ry"], ext["rz"])
    t = np.array([ext["tx"], ext["ty"], ext["tz"]])
    flip = np.array([1, -1, -1])
    capture_lines, edited_lines = gltf.primitives(glb, mode=1)
    np.testing.assert_allclose(
        capture_lines["positions"].min(0)[2], -depth_max, rtol=1e-5
    )
    apex = (-R.T @ t) * flip
    assert np.isclose(edited_lines["positions"], apex, atol=1e-5).all(1).any()
    positions, _, _, _ = gltf.textured_quad(glb)
    corners = frustum_segments(frame.K, 64, 48, -0.3 * depth_max)[:4, 1]
    np.testing.assert_allclose(positions, ((corners - t) @ R) * flip, atol=1e-5)


ES = {"rx": 4, "ry": -8, "rz": 2, "tx": -0.2, "ty": 0.03, "tz": 0.05}


def _Rt(ext: dict) -> tuple[np.ndarray, np.ndarray]:
    from app.geometry import rotation_matrix

    R = rotation_matrix(ext["rx"], ext["ry"], ext["rz"])
    return R, np.array([ext["tx"], ext["ty"], ext["tz"]])


def test_matrices_json(client: TestClient, frame: RGBDFrame) -> None:
    from app.geometry import epipoles, essential_matrix, fundamental_matrix

    body = client.get("/matrices.json", params={**_k_params(frame.K), **ES}).json()
    R, t = _Rt(ES)
    K = frame.K
    np.testing.assert_allclose(body["P1"], K @ np.hstack([np.eye(3), np.zeros((3, 1))]))
    np.testing.assert_allclose(body["P2"], K @ np.hstack([R, t[:, None]]))
    np.testing.assert_allclose(body["E"], essential_matrix(R, t))
    np.testing.assert_allclose(body["F"], fundamental_matrix(K, R, t))
    e1, e2 = epipoles(K, R, t)
    np.testing.assert_allclose(body["e1"], e1)
    np.testing.assert_allclose(body["e2"], e2)


def test_matrices_json_without_translation_has_zero_E(
    client: TestClient, frame: RGBDFrame
) -> None:
    body = client.get("/matrices.json", params={**_k_params(frame.K), "ry": 10}).json()
    assert not np.any(body["E"]) and not np.any(body["F"])


def test_cloud_pair_shows_both_views_without_image_plane(
    client: TestClient, frame: RGBDFrame
) -> None:
    client.post("/capture")
    params = {**_k_params(frame.K), **ES, "pair": "true"}
    glb = client.get("/capture/cloud.glb", params=params).content
    view1, view2 = gltf.primitives(glb, mode=1)
    flip = np.array([1, -1, -1])
    R, t = _Rt(ES)
    assert np.isclose(view1["positions"], 0, atol=1e-6).all(1).any()
    assert np.isclose(view2["positions"], (-R.T @ t) * flip, atol=1e-5).all(1).any()
    assert gltf.flat_triangles(glb) == []
    assert not [p for p in gltf.primitives(glb, mode=4)]  # no textured plane


@pytest.mark.parametrize("view", [1, 2])
def test_cloud_pair_hover_adds_epipolar_plane(
    client: TestClient, frame: RGBDFrame, view: int
) -> None:
    from app.geometry import epipolar_plane

    depth_max = client.post("/capture").json()["depth_max"]
    hover = {"hover_view": view, "hover_u": 40, "hover_v": 20}
    params = {**_k_params(frame.K), **ES, "pair": "true", **hover}
    glb = client.get("/capture/cloud.glb", params=params).content
    ((positions, material),) = gltf.flat_triangles(glb)
    R, t = _Rt(ES)
    expected = epipolar_plane(frame.K, R, t, view, 40, 20, depth_max)
    np.testing.assert_allclose(positions, expected * [1, -1, -1], atol=1e-5)
    assert material["alphaMode"] == "BLEND" and material["doubleSided"]
    assert len(gltf.primitives(glb, mode=1)) == 3  # 2 frustums + the plane's rays


def test_cloud_pair_hover_without_baseline_has_no_plane(
    client: TestClient, frame: RGBDFrame
) -> None:
    client.post("/capture")
    params = {**_k_params(frame.K), "ry": 10, "pair": "true", "hover_view": 1}
    glb = client.get(
        "/capture/cloud.glb", params={**params, "hover_u": 5, "hover_v": 5}
    ).content
    assert gltf.flat_triangles(glb) == []


@pytest.mark.parametrize("path", ["/", "/app.js"])
def test_page_and_script_are_revalidated(client: TestClient, path: str) -> None:
    # Browsers must not keep running a stale app.js after the code changes.
    assert client.get(path).headers["cache-control"] == "no-cache"
