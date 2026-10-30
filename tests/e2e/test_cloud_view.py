"""Browser test of the point cloud pane (needs the e2e image and internet access
for model-viewer from unpkg): ./run.sh e2e"""

import io
import threading
import time
from typing import Iterator

import numpy as np
import pytest
import uvicorn
from PIL import Image

from app.camera import RGBDFrame
from app.server import create_app

sync_api = pytest.importorskip("playwright.sync_api")

H, W = 480, 640
K = np.array([[615.0, 0, 318.0], [0, 615.3, 244.1], [0, 0, 1]])
PORT = 8765


class SceneCamera:
    """A wall at 3 m with a box at 1.2 m, coloured by pixel position."""

    name = "scene"

    def __init__(self) -> None:
        v, u = np.mgrid[0:H, 0:W]
        depth = np.full((H, W), 3.0, np.float32)
        depth[150:350, 200:450] = 1.2
        rgb = np.stack([u * 255 // W, v * 255 // H, np.full_like(u, 160)], -1)
        self.frame = RGBDFrame(rgb.astype(np.uint8), depth, K)

    def latest_rgb(self) -> np.ndarray:
        return self.frame.rgb

    def capture(self) -> RGBDFrame:
        return self.frame


@pytest.fixture(scope="module")
def page() -> Iterator["sync_api.Page"]:
    server = uvicorn.Server(
        uvicorn.Config(create_app(SceneCamera()), port=PORT, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(f"http://127.0.0.1:{PORT}/")
        page.click("#capture")
        page.wait_for_function("document.getElementById('cloud').loaded", timeout=30000)
        time.sleep(2)  # let the camera settle after fitView
        yield page
        browser.close()
    server.should_exit = True


def test_cloud_pane_is_4_by_3(page) -> None:
    w, h = page.evaluate(
        "(() => { const e = document.getElementById('cloud');"
        " return [e.clientWidth, e.clientHeight]; })()"
    )
    assert h == pytest.approx(w * 3 / 4, abs=1)


def test_cloud_view_uses_camera_fov(page) -> None:
    fov = page.evaluate("document.getElementById('cloud').getFieldOfView()")
    vfov = np.degrees(2 * np.arctan(H / 2 / K[1, 1]))
    assert fov == pytest.approx(vfov, abs=0.1)


def test_cloud_fills_pane_without_overflow(page) -> None:
    shot = np.asarray(Image.open(io.BytesIO(page.locator("#cloud").screenshot())))
    content = shot[..., :3].max(-1) > 20
    rows, cols = np.flatnonzero(content.any(1)), np.flatnonzero(content.any(0))
    h, w = content.shape
    # Fills one dimension (the binding one) ...
    assert max(np.ptp(rows) / h, np.ptp(cols) / w) > 0.9
    # ... and is not cut off on all four sides.
    edges = [
        content[:2].any(),
        content[-2:].any(),
        content[:, :2].any(),
        content[:, -2:].any(),
    ]
    assert not all(edges)


def _slide(page, name: str, value: float) -> None:
    page.evaluate(
        """([name, value]) => { const s = document.getElementById(`sl-${name}`);
           s.value = value; s.dispatchEvent(new Event("input")); }""",
        [name, value],
    )
    time.sleep(0.5)  # 100 ms debounce + render


def _rendered(page) -> np.ndarray:
    page.wait_for_function("document.getElementById('rrgb').complete")
    return np.asarray(Image.open(io.BytesIO(page.locator("#rrgb").screenshot())))


def test_extrinsics_tab_uses_intrinsics_from_first_tab(page) -> None:
    _slide(page, "fx", 700)
    page.click(".tab[data-tab=extr]")
    assert page.is_visible("#tab-extr") and not page.is_visible("#tab-intr")
    assert "700.0" in page.inner_text("#kmat-ro")
    before = _rendered(page)
    _slide(page, "tx", 0.3)
    src = page.get_attribute("#rrgb", "src")
    assert "fx=700" in src and "tx=0.3" in src
    assert "0.300" in page.inner_text("#rtmat")
    assert not np.array_equal(before, _rendered(page))


def test_intrinsics_tab_ignores_extrinsics(page) -> None:
    page.click(".tab[data-tab=intr]")
    time.sleep(0.3)
    assert "tx=" not in page.get_attribute("#rrgb", "src")
    assert page.input_value("#sl-tx") == "0.3"  # kept for when the tab returns


def test_extrinsics_tab_shows_projection_matrix(page) -> None:
    page.click(".tab[data-tab=extr]")
    page.wait_for_function("document.getElementById('pmat').textContent.length > 0")
    time.sleep(0.5)
    text = page.inner_text("#pmat")
    # Row 1, column 4 of K[R|t] with fx = 700 and t = (0.3, 0, 0): 700 * 0.3 = 210.
    assert "P = K[R|t]" in text and "210.00" in text


def test_extrinsics_tab_shows_camera_to_world_and_handedness(page) -> None:
    page.click(".tab[data-tab=extr]")
    text = page.inner_text("#c2w")
    # Inverse of X_cam = X + (0.3, 0, 0): X = X_cam - t, centre C = -R^T t.
    assert "[R^T | -R^T t]" in text and "C = -R^T t = (-0.300, 0.000, 0.000)" in text
    assert "right-handed" in page.inner_text("#tab-extr")


def _painted(page, canvas: str) -> int:
    return page.evaluate(
        """(id) => { const c = document.getElementById(id);
           const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
           let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] > 0; return n; }""",
        canvas,
    )


def test_essential_tab_hover_draws_epipolar_geometry(page) -> None:
    page.click(".tab[data-tab=ess]")
    time.sleep(0.5)
    assert page.is_visible("#panel-view1") and page.is_visible("#panel-view2")
    assert not page.is_visible("#panel-rgb") and not page.is_visible("#panel-rrgb")
    assert "E = [t]x R" in page.inner_text("#es-E")
    assert "F = K^-T E K^-1" in page.inner_text("#es-F")
    # Bottom row split: sliders on the left, matrices on the right.
    left = page.locator("#tab-ess").bounding_box()
    right = page.locator("#panel-ess-mats").bounding_box()
    assert right["x"] > left["x"] + left["width"] and abs(right["y"] - left["y"]) < 50
    box = page.locator("#panel-view1 .stack").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.4)
    assert _painted(page, "epi1") > 0 and _painted(page, "epi2") > 0
    time.sleep(0.5)  # hover pause -> cloud reload with the epipolar plane
    src = page.evaluate("document.getElementById('cloud').src")
    assert "pair=true" in src and "hover_view=1" in src
    page.mouse.move(5, 5)  # leave the image
    assert _painted(page, "epi1") == 0 and _painted(page, "epi2") == 0


def _magenta(page, canvas: str) -> int:
    return page.evaluate(
        """(id) => { const c = document.getElementById(id);
           const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
           let n = 0;
           for (let i = 0; i < d.length; i += 4)
             n += d[i] > 200 && d[i + 1] < 80 && d[i + 2] > 200 && d[i + 3] > 200;
           return n; }""",
        canvas,
    )


def test_essential_tab_marks_epipoles_inside_image(page) -> None:
    page.click(".tab[data-tab=ess]")
    time.sleep(0.5)
    # Default pure tx baseline: epipoles at infinity, nothing marked.
    assert _magenta(page, "epi1") == 0 and _magenta(page, "epi2") == 0
    _slide(page, "es-tx", 0)
    _slide(page, "es-tz", -0.2)  # forward motion: both epipoles at the principal point
    assert _magenta(page, "epi1") > 0 and _magenta(page, "epi2") > 0
    assert "camera 2 -> camera 1" in page.inner_text("#es-P")
