import threading

import numpy as np
import pyrealsense2 as rs

from app.camera import RGBDFrame


class RealSenseCamera:
    """Streams aligned colour + depth in a background thread; keeps the latest frame."""

    name = "realsense"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
        )
        self._align = rs.align(rs.stream.color)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._latest: RGBDFrame | None = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            frames = self._align.process(self._pipeline.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            frame = RGBDFrame(
                rgb=np.asanyarray(color.get_data()).copy(),
                depth_m=np.asanyarray(depth.get_data()).astype(np.float32)
                * self._depth_scale,
                K=self._K,
            )
            with self._lock:
                self._latest = frame
            self._ready.set()

    def _get(self) -> RGBDFrame:
        self._ready.wait(timeout=5.0)
        with self._lock:
            if self._latest is None:
                raise RuntimeError("No frames received from the RealSense camera")
            return self._latest

    def latest_rgb(self) -> np.ndarray:
        return self._get().rgb

    def capture(self) -> RGBDFrame:
        return self._get()
