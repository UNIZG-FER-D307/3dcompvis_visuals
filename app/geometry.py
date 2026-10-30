from typing import Sequence

import numpy as np
import trimesh
from PIL import Image
from trimesh.path.entities import Line
from matplotlib import colormaps


def backproject(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lift valid depth pixels to camera-frame 3D points.

    Returns points [N, 3] and their flat pixel indices [N].
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    idx = np.flatnonzero(depth > 0)
    z = depth.reshape(-1)[idx].astype(np.float64)
    pix = np.stack([u.reshape(-1)[idx], v.reshape(-1)[idx], np.ones_like(idx)])
    rays = np.linalg.inv(K) @ pix
    return (rays * z).T, idx


def project(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
    sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render points through K with a z-buffer. Returns rgb [h, w, 3], depth [h, w].

    Optional per-point `sizes` draw each point as a centred s×s square (splat).
    """
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    if sizes is None:
        sizes = np.ones(len(points), dtype=np.int64)
    z = points[:, 2]
    front = z > 0
    points, colors, z, sizes = points[front], colors[front], z[front], sizes[front]
    uvw = points @ K.T
    u = np.rint(uvw[:, 0] / z).astype(np.int64)
    v = np.rint(uvw[:, 1] / z).astype(np.int64)
    lo, hi = -((sizes - 1) // 2), sizes // 2  # square offsets covered per point
    smax = int(sizes.max(initial=1))
    us, vs, zs, cs = [], [], [], []
    for dy in range(-((smax - 1) // 2), smax // 2 + 1):
        for dx in range(-((smax - 1) // 2), smax // 2 + 1):
            m = (lo <= dx) & (dx <= hi) & (lo <= dy) & (dy <= hi)
            us.append(u[m] + dx)
            vs.append(v[m] + dy)
            zs.append(z[m])
            cs.append(colors[m])
    u, v, z, colors = (np.concatenate(a) for a in (us, vs, zs, cs))
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, colors = u[inside], v[inside], z[inside], colors[inside]
    # Far-to-near order: later (nearer) writes overwrite earlier ones.
    order = np.argsort(-z, kind="stable")
    rgb[v[order], u[order]] = colors[order]
    depth[v[order], u[order]] = z[order]
    return rgb, depth


MAX_SPLAT = 8


def splat_sizes(
    K_src: np.ndarray, K_dst: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray
) -> np.ndarray:
    """Splat size per point: how much its 1-pixel capture footprint is magnified."""
    scale = max(K_dst[0, 0] / K_src[0, 0], K_dst[1, 1] / K_src[1, 1])
    with np.errstate(divide="ignore"):
        mag = scale * z_src / z_dst
    return np.clip(np.ceil(mag), 1, MAX_SPLAT).astype(np.int64)


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """R = Rz @ Ry @ Rx for angles in degrees about the camera x, y, z axes."""
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = (
        np.cos(ax),
        np.sin(ax),
        np.cos(ay),
        np.sin(ay),
        np.cos(az),
        np.sin(az),
    )
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def warp(
    rgb: np.ndarray, depth: np.ndarray, K_src: np.ndarray, K_dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Re-render a capture through K_dst by inverse warping with bilinear sampling.

    With fixed extrinsics, reprojection is the homography K_dst @ inv(K_src), so
    each output pixel samples the source at K_src @ inv(K_dst) @ p. Only neighbours
    with valid depth contribute; pixels with none stay black / depth 0.
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    pix = np.stack([u.ravel(), v.ravel(), np.ones(h * w)])
    # Rounding snaps near-integer coordinates (float error in K @ inv(K)) onto the
    # pixel, so a ~1e-9 weight cannot pull in an invalid pixel's neighbour.
    x, y = np.round(K_src @ np.linalg.inv(K_dst) @ pix, 6)[:2]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    rgb_flat = rgb.reshape(-1, 3).astype(np.float32)
    depth_flat = depth.reshape(-1).astype(np.float32)
    acc_rgb = np.zeros((h * w, 3), dtype=np.float32)
    acc_depth = np.zeros(h * w, dtype=np.float32)
    acc_w = np.zeros(h * w, dtype=np.float32)
    for dx, dy, wgt in [
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]:
        xi, yi = x0 + dx, y0 + dy
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        flat = np.clip(yi, 0, h - 1) * w + np.clip(xi, 0, w - 1)
        d = depth_flat[flat]
        wgt *= inside & (d > 0)
        acc_rgb += wgt[:, None] * rgb_flat[flat]
        acc_depth += wgt * d
        acc_w += wgt
    hit = acc_w > 0
    acc_rgb[hit] /= acc_w[hit, None]
    acc_depth[hit] /= acc_w[hit]
    out_rgb = np.rint(acc_rgb).astype(np.uint8).reshape(h, w, 3)
    return out_rgb, acc_depth.reshape(h, w)


def skew(t: np.ndarray) -> np.ndarray:
    """Cross-product matrix [t]x, so that skew(t) @ x == t x x."""
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """E = [t]x R for views P1 = [I|0], P2 = [R|t]: x2n^T E x1n = 0."""
    return skew(t) @ R


def fundamental_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """F = K^-T E K^-1 for both views sharing K: x2^T F x1 = 0 in pixels."""
    Kinv = np.linalg.inv(K)
    return Kinv.T @ essential_matrix(R, t) @ Kinv


def epipoles(
    K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous epipoles: e1 = image of C2 in view 1, e2 = image of C1 in view 2."""
    return K @ (-R.T @ t), K @ t


def epipolar_plane(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    view: int,
    u: float,
    v: float,
    depth: float,
) -> np.ndarray:
    """Triangle [C1, C2, X] (world = view 1 frame) spanning the epipolar plane of
    pixel (u, v) in `view`; X is on that pixel's ray at `depth` in that camera."""
    X_cam = depth * (np.linalg.inv(K) @ [u, v, 1.0])
    X = X_cam if view == 1 else R.T @ (X_cam - t)
    return np.stack([np.zeros(3), -R.T @ t, X])


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Plasma-colormapped depth as uint8 RGB; invalid (0) depth is black."""
    t = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (colormaps["plasma"](t)[..., :3] * 255).round().astype(np.uint8)
    rgb[depth <= 0] = 0
    return rgb


def frustum_segments(K: np.ndarray, w: int, h: int, depth: float) -> np.ndarray:
    """Line segments [8, 2, 3] of the camera frustum out to `depth`, camera frame.

    4 rays from the camera centre through the image corners, then the far rectangle.
    """
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
    far = (np.linalg.inv(K) @ corners.T).T * depth
    rays = np.stack([np.zeros_like(far), far], axis=1)
    edges = np.stack([far, np.roll(far, -1, axis=0)], axis=1)
    return np.concatenate([rays, edges])


IMAGE_PLANE_OPACITY = 0.75


def to_glb(
    points: np.ndarray,
    colors: np.ndarray,
    frustums: Sequence[tuple[np.ndarray, tuple[int, int, int]]] = (),
    image_plane: tuple[np.ndarray, np.ndarray] | None = None,
    triangles: Sequence[tuple[np.ndarray, tuple[int, int, int, int]]] = (),
    max_points: int = 150_000,
) -> bytes:
    """Export a coloured point cloud, frustum line segments (with an RGB colour
    each), an optional image plane (corners TL, TR, BR, BL and its RGB image) and
    flat translucent triangles (vertices [3, 3] and an RGBA colour each) as GLB,
    converting camera axes to glTF axes."""
    flip = np.array([1, -1, -1])
    if len(points) > max_points:
        keep = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]
    geometry = [trimesh.PointCloud(points * flip, colors=colors)]
    for segments, color in frustums:
        entities = [
            Line([2 * i, 2 * i + 1], color=(*color, 255)) for i in range(len(segments))
        ]
        geometry.append(
            trimesh.path.Path3D(
                entities=entities, vertices=segments.reshape(-1, 3) * flip
            )
        )
    if image_plane is not None:
        geometry.append(_textured_quad(image_plane[0] * flip, image_plane[1]))
    for vertices, rgba in triangles:
        geometry.append(_flat_triangle(vertices * flip, rgba))
    return trimesh.Scene(geometry).export(file_type="glb")


def _flat_triangle(
    vertices: np.ndarray, rgba: tuple[int, int, int, int]
) -> trimesh.Trimesh:
    """Double-sided, translucent, unlit-looking (emissive on black) triangle."""
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0, 0, 0, rgba[3]],
        emissiveFactor=[c / 255 for c in rgba[:3]],
        alphaMode="BLEND",
        doubleSided=True,
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=[[0, 1, 2]],
        visual=trimesh.visual.TextureVisuals(material=material),
        process=False,
    )


def _textured_quad(corners: np.ndarray, rgb: np.ndarray) -> trimesh.Trimesh:
    """Semi-transparent, unlit-looking quad over corners (TL, TR, BR, BL) showing rgb.

    The image is emissive on a black base, so scene lighting does not shade it.
    """
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0, 0, 0, round(255 * IMAGE_PLANE_OPACITY)],
        emissiveTexture=Image.fromarray(rgb),
        emissiveFactor=[1.0, 1.0, 1.0],
        alphaMode="BLEND",
        doubleSided=True,
    )
    # trimesh UVs have a bottom-left origin; it flips them to glTF on export.
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
    return trimesh.Trimesh(
        vertices=corners,
        faces=[[0, 3, 2], [0, 2, 1]],
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
