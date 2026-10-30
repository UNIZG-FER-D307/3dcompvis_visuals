import io
import logging
import time
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image

from app import mvg
from app.camera import Camera, RGBDFrame
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
    splat_sizes,
    to_glb,
    warp,
)

STATIC = Path(__file__).parent / "static"
CAPTURE_FRUSTUM_RGB = (255, 255, 255)
EDITED_FRUSTUM_RGB = (0, 255, 255)
IMAGE_PLANE_DEPTH = 0.3  # behind the centre, fraction of frustum depth, at capture fx
EPIPOLAR_RGB = (255, 220, 0)
EPIPOLAR_PLANE_ALPHA = 110


def _encode(rgb: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt)
    return buf.getvalue()


def _png(rgb: np.ndarray) -> Response:
    return Response(_encode(rgb, "PNG"), media_type="image/png")


def mjpeg_frames(camera: Camera, fps: float = 30.0) -> Iterator[bytes]:
    """Endless multipart chunks of the camera's latest RGB frame as JPEG."""
    while True:
        jpeg = _encode(camera.latest_rgb(), "JPEG")
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / fps)


Extrinsics = tuple[np.ndarray, np.ndarray] | None  # (R, t) world→camera


def extrinsics(
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
) -> Extrinsics:
    """[R|t] from rotation angles (degrees) and translation (m); None if identity."""
    if not any((rx, ry, rz, tx, ty, tz)):
        return None
    return rotation_matrix(rx, ry, rz), np.array([tx, ty, tz])


def intrinsics(
    fx: float, fy: float, cx: float, cy: float, s: float = 0.0
) -> np.ndarray:
    return np.array([[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def to_world(points: np.ndarray, ext: Extrinsics) -> np.ndarray:
    """Camera-frame points to the world (capture) frame: R^T (X - t)."""
    if ext is None:
        return points
    R, t = ext
    return (points - t) @ R


class Capture:
    """A captured frame with its point cloud and fixed depth colour range."""

    def __init__(self, frame: RGBDFrame) -> None:
        self.frame = frame
        self.points, idx = backproject(frame.depth_m, frame.K)
        self.colors = frame.rgb.reshape(-1, 3)[idx]
        self.vmin, self.vmax = np.percentile(self.points[:, 2], [2, 98])

    def depth_rgb(self, depth: np.ndarray) -> np.ndarray:
        return colorize_depth(depth, self.vmin, self.vmax)

    def render(
        self, K: np.ndarray, interp: bool, ext: Extrinsics = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """RGB and depth seen through K[R|t].

        Without extrinsics, interpolation is the exact inverse warp; with them the
        camera centre moves (parallax, occlusion), so it splats points instead.
        """
        h, w = self.frame.depth_m.shape
        if ext is None:
            if interp:
                return warp(self.frame.rgb, self.frame.depth_m, self.frame.K, K)
            return project(self.points, self.colors, K, h, w)
        R, t = ext
        points = self.points @ R.T + t
        sizes = None
        if interp:
            sizes = splat_sizes(self.frame.K, K, self.points[:, 2], points[:, 2])
        return project(points, self.colors, K, h, w, sizes)


def create_app(camera: Camera) -> FastAPI:
    app = FastAPI()
    state: dict[str, Capture] = {}

    def current() -> Capture:
        if "capture" not in state:
            raise HTTPException(409, "No frame captured yet")
        return state["capture"]

    # no-cache: always revalidate, so a code change is picked up on reload.
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            STATIC / "index.html", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/app.js")
    def script() -> FileResponse:
        return FileResponse(STATIC / "app.js", headers={"Cache-Control": "no-cache"})

    @app.get("/mvg.js")
    def mvg_script() -> FileResponse:
        return FileResponse(STATIC / "mvg.js", headers={"Cache-Control": "no-cache"})

    @app.get("/camera")
    def camera_info() -> dict:
        return {"name": camera.name}

    @app.get("/stream")
    def stream() -> StreamingResponse:
        return StreamingResponse(
            mjpeg_frames(camera), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.post("/capture")
    def capture() -> dict:
        cap = Capture(camera.capture())
        state["capture"] = cap
        h, w = cap.frame.depth_m.shape
        return {
            "K": cap.frame.K.tolist(),
            "width": w,
            "height": h,
            "depth_min": float(cap.vmin),
            "depth_max": float(cap.vmax),
            "image_plane_depth": IMAGE_PLANE_DEPTH * float(cap.vmax),
        }

    @app.get("/colorbar.png")
    def colorbar() -> Response:
        return _png(colorize_depth(np.linspace(1.0, 2.0, 256)[None], 1.0, 2.0))

    @app.get("/capture/rgb.png")
    def capture_rgb() -> Response:
        return _png(current().frame.rgb)

    @app.get("/capture/depth.png")
    def capture_depth() -> Response:
        cap = current()
        return _png(cap.depth_rgb(cap.frame.depth_m))

    @app.get("/capture/cloud.glb")
    def capture_cloud(
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        s: float = 0.0,
        interp: bool = False,
        ext: Extrinsics = Depends(extrinsics),
        pair: bool = False,
        hover_view: int | None = Query(None, ge=1, le=2),
        hover_u: float = 0.0,
        hover_v: float = 0.0,
    ) -> Response:
        """Point cloud with the capture frustum, plus the edited-K frustum if given.

        The image plane sits in the edited frustum (else the capture one) and shows
        the image seen through that K. With extrinsics, the edited frustum and plane
        are placed at the camera pose: X_world = R^T (X_cam - t).

        With `pair`, it shows the two views K[I|0] and K[R|t] instead, and a hovered
        pixel adds its epipolar plane (triangle C1, C2, X) and the rays C1-X, C2-X.
        """
        cap = current()
        h, w = cap.frame.depth_m.shape
        if pair:
            K = (
                cap.frame.K
                if None in (fx, fy, cx, cy)
                else intrinsics(fx, fy, cx, cy, s)
            )
            view1 = frustum_segments(K, w, h, cap.vmax)
            frustums = [
                (view1, CAPTURE_FRUSTUM_RGB),
                (to_world(view1, ext), EDITED_FRUSTUM_RGB),
            ]
            triangles = []
            if hover_view is not None and ext is not None and np.any(ext[1]):
                R, t = ext
                tri = epipolar_plane(K, R, t, hover_view, hover_u, hover_v, cap.vmax)
                triangles.append((tri, (*EPIPOLAR_RGB, EPIPOLAR_PLANE_ALPHA)))
                rays = np.stack([[tri[0], tri[2]], [tri[1], tri[2]]])
                frustums.append((rays, EPIPOLAR_RGB))
            glb = to_glb(cap.points, cap.colors, frustums, triangles=triangles)
            return Response(glb, media_type="model/gltf-binary")
        # The image plane sits behind the optical centre like a physical sensor:
        # each pixel's ray continues through the centre to depth -d, so the image
        # lands inverted. Its camera's frustum gets the rear rays to the plane.
        K, image, ext_plane = cap.frame.K, cap.frame.rgb, None
        frustums = [(frustum_segments(K, w, h, cap.vmax), CAPTURE_FRUSTUM_RGB)]
        if None not in (fx, fy, cx, cy):
            K = intrinsics(fx, fy, cx, cy, s)
            image = cap.render(K, interp, ext)[0]
            edited = frustum_segments(K, w, h, cap.vmax)
            frustums.append((to_world(edited, ext), EDITED_FRUSTUM_RGB))
            ext_plane = ext
        # d = f = fx * pixel width, with the pixel width fixed by the capture K.
        d = IMAGE_PLANE_DEPTH * cap.vmax * K[0, 0] / cap.frame.K[0, 0]
        rear = to_world(frustum_segments(K, w, h, -d), ext_plane)
        segments, color = frustums[-1]
        frustums[-1] = (np.concatenate([segments, rear]), color)
        corners = rear[:4, 1]
        glb = to_glb(cap.points, cap.colors, frustums, image_plane=(corners, image))
        return Response(glb, media_type="model/gltf-binary")

    @app.get("/matrices.json")
    def matrices(
        K: np.ndarray = Depends(intrinsics), ext: Extrinsics = Depends(extrinsics)
    ) -> dict:
        """P1 = K[I|0], P2 = K[R|t], E, F and the epipoles, for the readouts."""
        R, t = ext if ext is not None else (np.eye(3), np.zeros(3))
        e1, e2 = epipoles(K, R, t)
        return {
            "P1": (K @ np.hstack([np.eye(3), np.zeros((3, 1))])).tolist(),
            "P2": (K @ np.hstack([R, t[:, None]])).tolist(),
            "E": essential_matrix(R, t).tolist(),
            "F": fundamental_matrix(K, R, t).tolist(),
            "e1": e1.tolist(),
            "e2": e2.tolist(),
        }

    @app.get("/render.png")
    def render(
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        s: float = 0.0,
        kind: Literal["rgb", "depth"] = "rgb",
        interp: bool = False,
        ext: Extrinsics = Depends(extrinsics),
    ) -> Response:
        cap = current()
        rgb, depth = cap.render(intrinsics(fx, fy, cx, cy, s), interp, ext)
        return _png(rgb if kind == "rgb" else cap.depth_rgb(depth))

    # Reconstruction demos (app/mvg.py): synthetic targets, no camera involved.
    def pose(ext: Extrinsics = Depends(extrinsics)) -> tuple[np.ndarray, np.ndarray]:
        return ext if ext is not None else (np.eye(3), np.zeros(3))

    def corner_ids(ids: str = Query("", pattern=r"^(\d+(,\d+)*)?$")) -> list[int]:
        return [int(i) for i in ids.split(",") if i]

    def glb(data: bytes) -> Response:
        return Response(data, media_type="model/gltf-binary")

    @app.get("/mvg/triangulation.json")
    def mvg_triangulation(
        p=Depends(pose), noise: float = Query(0.0, ge=0), point: int | None = None
    ) -> dict:
        return mvg.jsonable(mvg.triangulation(*p, noise, point))

    @app.get("/mvg/triangulation.glb")
    def mvg_triangulation_glb(
        p=Depends(pose), noise: float = Query(0.0, ge=0), point: int | None = None
    ) -> Response:
        return glb(mvg.triangulation_glb(*p, noise, point))

    @app.get("/mvg/plane.json")
    def mvg_plane(
        p=Depends(pose), noise: float = Query(0.0, ge=0), ids=Depends(corner_ids)
    ) -> dict:
        return mvg.jsonable(mvg.plane(*p, noise, ids))

    @app.get("/mvg/plane.glb")
    def mvg_plane_glb(
        p=Depends(pose), noise: float = Query(0.0, ge=0), ids=Depends(corner_ids)
    ) -> Response:
        return glb(mvg.plane_glb(*p, noise, ids))

    @app.get("/mvg/twoview.json")
    def mvg_twoview(
        p=Depends(pose), noise: float = Query(0.0, ge=0), ids=Depends(corner_ids)
    ) -> dict:
        return mvg.jsonable(mvg.twoview(*p, noise, ids))

    @app.get("/mvg/twoview.glb")
    def mvg_twoview_glb(
        p=Depends(pose), noise: float = Query(0.0, ge=0), ids=Depends(corner_ids)
    ) -> Response:
        return glb(mvg.twoview_glb(*p, noise, ids))

    @app.get("/mvg/scale.json")
    def mvg_scale(
        p=Depends(pose), noise: float = Query(0.0, ge=0), ids=Depends(corner_ids)
    ) -> dict:
        return mvg.jsonable(mvg.metric_scale(*p, noise, ids))

    @app.get("/mvg/scale.glb")
    def mvg_scale_glb(
        p=Depends(pose), noise: float = Query(0.0, ge=0), ids=Depends(corner_ids)
    ) -> Response:
        return glb(mvg.metric_scale_glb(*p, noise, ids))

    return app


def main() -> FastAPI:
    """App factory for uvicorn: the RealSense if available, else a synthetic scene."""
    from app import realsense
    from app.synthetic import SyntheticCamera

    try:
        camera: Camera = realsense.RealSenseCamera()
    except RuntimeError as err:
        logging.warning("RealSense unavailable (%s); using a synthetic scene", err)
        camera = SyntheticCamera()
    return create_app(camera)
