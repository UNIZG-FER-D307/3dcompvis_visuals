"""Browser test of the reconstruction demos (needs the e2e image and internet
access for model-viewer and KaTeX): ./run.sh e2e"""

import threading
import time
from typing import Iterator

import numpy as np
import pytest
import uvicorn

from app.server import create_app

sync_api = pytest.importorskip("playwright.sync_api")

PORT = 8766


class BlankCamera:
    name = "blank"

    def latest_rgb(self) -> np.ndarray:
        return np.zeros((48, 64, 3), np.uint8)


@pytest.fixture(scope="module")
def page() -> Iterator["sync_api.Page"]:
    server = uvicorn.Server(
        uvicorn.Config(create_app(BlankCamera()), port=PORT, log_level="warning")
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
        page.click("#mvg-open")  # no capture needed
        page.wait_for_function("mdata !== null")
        yield page
        browser.close()
    server.should_exit = True


def _click_corner(page, view: int, corner: int) -> None:
    """Click a few pixels off a corner, so it has to snap."""
    u, v = page.evaluate(f"mdata.views[{view}].q[{corner}]")
    box = page.locator(f"#mvg-c{view + 1}").bounding_box()
    page.mouse.click(
        box["x"] + (u + 4) / 640 * box["width"],
        box["y"] + (v - 3) / 480 * box["height"],
    )


def _wait_picked(page, js: str) -> None:
    page.wait_for_function(js)
    page.wait_for_function("document.querySelector('#mvg-math .katex') !== null")


def test_section_opens_without_capture(page) -> None:
    assert page.is_hidden("#live") and page.is_visible("#mvg")
    assert page.is_visible("#mvg-c1") and page.is_visible("#mvg-c2")
    assert "Click a corner" in page.inner_text("#mvg-math")


def test_triangulation_click_solves_for_q(page) -> None:
    _click_corner(page, 1, 7)
    _wait_picked(page, "picked.tri === 7 && mdata.point === 7")
    text = page.inner_text("#mvg-math")
    assert "Structure reconstruction" in text and "mm" in text
    assert "point=7" in page.evaluate("document.getElementById('mvg-cloud').src")
    page.wait_for_function("document.getElementById('mvg-cloud').loaded", timeout=30000)


def test_plane_pose_four_corners_give_one_valid_candidate(page) -> None:
    page.click(".mtab[data-mtab=plane]")
    page.wait_for_function("mdata !== null && mdata.ids !== undefined")
    assert page.is_hidden("#mvg-panel2")
    for k, corner in enumerate([0, 6, 56, 62]):
        _click_corner(page, 0, corner)
        page.wait_for_function(f"mdata.ids.length === {k + 1}")
    _wait_picked(page, "mdata.candidates !== undefined")
    text = page.inner_text("#mvg-math")
    assert text.count("✔ valid") == 1 and text.count("✘ invalid") == 1


def test_two_views_of_a_plane_list_candidates(page) -> None:
    page.click(".mtab[data-mtab=two]")
    page.wait_for_function("mdata !== null && mdata.q1 !== undefined")
    for k, (view, corner) in enumerate([(0, 0), (1, 6), (0, 31), (1, 56), (0, 62)]):
        _click_corner(page, view, corner)
        page.wait_for_function(f"mdata.ids.length === {k + 1}")
    _wait_picked(page, "mdata.candidates !== undefined")
    text = page.inner_text("#mvg-math")
    assert text.count("valid") == 4 and 1 <= text.count("✔ valid") <= 2
    assert "★" in text
    page.evaluate(
        """() => { const s = document.getElementById("sl-noise");
           s.value = 2; s.dispatchEvent(new Event("input")); }"""
    )
    page.wait_for_function("mdata.ids.length === 5 && mvgQuery().get('noise') === '2'")
    time.sleep(0.5)
    assert "★" in page.inner_text("#mvg-math")


def test_metric_scale_recovers_t_from_t_over_d(page) -> None:
    page.click(".mtab[data-mtab=scale]")
    page.wait_for_function("mdata !== null && mdata.t_true !== undefined")
    for k, (view, corner) in enumerate([(0, 0), (1, 6), (0, 31), (1, 56), (0, 62)]):
        _click_corner(page, view, corner)
        page.wait_for_function(f"mdata.ids.length === {k + 1}")
    _wait_picked(page, "mdata.d_metric !== undefined")
    text = page.inner_text("#mvg-math")
    assert "The known target fixes the scale" in text and "the true one" in text
    assert "1.0000" in text  # the true d (the shared noise slider is still at 2)
    assert "scale.glb" in page.evaluate("document.getElementById('mvg-cloud').src")


def test_back_to_live(page) -> None:
    page.click("#mvg-back")
    assert page.is_visible("#live") and page.is_hidden("#mvg")
