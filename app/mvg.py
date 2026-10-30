"""Multi-view geometry demos on synthetic checkerboard targets, in the notation of
the lecture notes (mvg.tex): linear triangulation, the homography between a plane
and its image (and the camera pose from it), and two views of a plane.

All frames are right-handed OpenCV frames (x right, y down, z forward); a pose
(R, t) maps a point into camera coordinates: X_cam = R X + t.
"""

from typing import Any, NamedTuple

import numpy as np
from matplotlib import colormaps

from app.geometry import frustum_segments, rotation_matrix, to_glb
from app.synthetic import HEIGHT, WIDTH
from app.synthetic import K as _K

K = _K.copy()
FRUSTUM_DEPTH = 0.3  # m, how far the camera frustums are drawn in 3D
DARK_RGB, LIGHT_RGB = (60, 60, 60), (235, 235, 235)
CAMERA_1_RGB, CAMERA_2_RGB = (255, 255, 255), (0, 255, 255)
RAY_RGB = (255, 220, 0)
ESTIMATE_RGB = (0, 220, 80)
# Up to four candidates, each in its own colour: greens if valid, reds if not.
VALID_RGB = [(0, 200, 80), (170, 255, 60), (0, 150, 120), (110, 255, 180)]
INVALID_RGB = [(230, 30, 30), (255, 130, 200), (255, 140, 0), (150, 0, 40)]
PLANE_ALPHA = 70
_NOISE = np.random.default_rng(0).standard_normal((3, 256, 2))  # per view, corner


class Target(NamedTuple):
    corners: np.ndarray  # [N, 3]
    squares: np.ndarray  # [M, 4] corner indices, in order around the square
    dark: np.ndarray  # [M] bool


def _grid(
    origin: np.ndarray, du: np.ndarray, dv: np.ndarray, nu: int, nv: int
) -> Target:
    """nu x nv squares spanned by du, dv; corner (i, j) has index i * (nv + 1) + j."""
    i, j = np.mgrid[0 : nu + 1, 0 : nv + 1]
    corners = origin + i.reshape(-1, 1) * du + j.reshape(-1, 1) * dv
    i, j = np.mgrid[0:nu, 0:nv]
    i, j = i.ravel(), j.ravel()
    c = lambda a, b: a * (nv + 1) + b  # noqa: E731
    squares = np.stack([c(i, j), c(i + 1, j), c(i + 1, j + 1), c(i, j + 1)], 1)
    return Target(corners, squares, (i + j) % 2 == 1)


def checkerboard(nx: int, ny: int, square: float) -> Target:
    """Calibration target: Z = 0 in the target frame, centred on its origin."""
    origin = np.array([-nx * square / 2, -ny * square / 2, 0.0])
    return _grid(origin, np.array([square, 0, 0]), np.array([0, square, 0]), nx, ny)


BOOK_PAGE = (5, 6)  # squares across and down a page
BOOK_SQUARE = 0.06  # m
BOOK_SPINE_Z = 1.1  # m in front of camera A


def book_target() -> Target:
    """Two perpendicular boards joined along a vertical spine, opening towards
    camera A (world = camera A frame)."""
    nu, nv = BOOK_PAGE
    spine = np.array([0.0, -nv * BOOK_SQUARE / 2, BOOK_SPINE_Z])
    dv = np.array([0.0, BOOK_SQUARE, 0.0])
    s = BOOK_SQUARE / np.sqrt(2)
    pages = [_grid(spine, np.array([d * s, 0, -s]), dv, nu, nv) for d in (-1, 1)]
    corners = np.concatenate([p.corners for p in pages])
    squares = np.concatenate([pages[0].squares, pages[1].squares + len(corners) // 2])
    # Merge the spine corners the two pages share.
    unique, inverse = np.unique(
        np.round(corners, 9), axis=0, return_index=False, return_inverse=True
    )
    return Target(
        unique, inverse.ravel()[squares], np.concatenate([p.dark for p in pages])
    )


PLANE_TARGET = checkerboard(8, 6, 0.05)
BOOK_TARGET = book_target()
# Camera 1 of the two-view tabs, target -> camera 1: the board is parallel to its
# image plane, 1 m away (n = e3, d = 1 m).
CAMERA_1 = (np.eye(3), np.array([0.0, 0.0, 1.0]))


def target_samples(target: Target, per_edge: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Coloured points filling each square (per_edge^2 each), for the 3D view."""
    s = (np.arange(per_edge) + 0.5) / per_edge
    a, b = (x.ravel() for x in np.meshgrid(s, s))
    weights = np.stack([(1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b], 1)
    points = np.einsum("kj,mjc->mkc", weights, target.corners[target.squares])
    colors = np.where(target.dark[:, None], DARK_RGB, LIGHT_RGB).astype(np.uint8)
    return points.reshape(-1, 3), np.repeat(colors, len(a), axis=0)


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pixels [N, 2] of points X [N, 2 or 3] (homogeneous 1 appended) through P."""
    x = np.column_stack([X, np.ones(len(X))]) @ P.T
    return x[:, :2] / x[:, 2:]


def noisy(q: np.ndarray, view: int, ids: list[int], sigma: float) -> np.ndarray:
    """q plus sigma times a fixed standard-normal offset per (view, corner)."""
    return q + sigma * _NOISE[view, ids]


def triangulate(
    P_A: np.ndarray, P_B: np.ndarray, q_A: np.ndarray, q_B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear method: two rows of [q]x P Q = 0 per view give A_4x4 Q = 0.

    Returns A, its singular values and Q (homogeneous, unit norm) from the last
    right singular vector."""
    A = np.array(
        [
            q_A[1] * P_A[2] - P_A[1],
            q_A[0] * P_A[2] - P_A[0],
            q_B[1] * P_B[2] - P_B[1],
            q_B[0] * P_B[2] - P_B[0],
        ]
    )
    _, s, Vt = np.linalg.svd(A)
    return A, s, Vt[-1]


def homography_dlt(
    q_R: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H with q ~ H q_R from q x (H q_R) = 0: the two independent rows per pair
    give A_2nx9 h = 0, solved by SVD with |h| = 1. Returns A, its singular values
    and H."""
    rows = []
    for (X, Y), (x, y) in zip(q_R, q):
        r = np.array([X, Y, 1.0])
        rows += [np.concatenate([0 * r, -r, y * r]), np.concatenate([r, 0 * r, -x * r])]
    A = np.array(rows)
    _, s, Vt = np.linalg.svd(A)
    # With 4 points A is 8x9: its ninth singular value is 0.
    return A, np.pad(s, (0, 9 - len(s))), Vt[-1].reshape(3, 3)


def _nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def pose_from_homography(
    K: np.ndarray, H: np.ndarray, q_R: np.ndarray
) -> list[dict[str, Any]]:
    """Both candidates of K^-1 H = mu [r1 r2 t], mu = +-|K^-1 h1|.

    r3 = r1 x r2, and R is the nearest rotation to [r1 r2 r3]. A candidate is
    valid if every target point q_R lies in front of the camera."""
    M = np.linalg.inv(K) @ H
    candidates = []
    for sign in (1, -1):
        mu = sign * np.linalg.norm(M[:, 0])
        r1, r2, t = M.T / mu
        R_raw = np.column_stack([r1, r2, np.cross(r1, r2)])
        depths = q_R @ np.array([r1[2], r2[2]]) + t[2]
        candidates.append(
            {
                "mu": mu,
                "R_raw": R_raw,
                "R": _nearest_rotation(R_raw),
                "t": t,
                "r2_norm": np.linalg.norm(r2),
                "r1_dot_r2": r1 @ r2,
                "depths": depths,
                "valid": bool(np.all(depths > 0)),
            }
        )
    return candidates


def decompose_homography(
    H: np.ndarray, m1: np.ndarray, m2: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Candidates (R, t/d, n) of a calibrated H = R + t n^T / d (Ma et al., An
    Invitation to 3-D Vision, Alg. 5.2), from normalised points m1, m2 [N, 3].

    H is first scaled to sigma_2 = 1 with x2^T H x1 > 0. A candidate is valid if
    all points lie in front of camera 1: n^T m1 > 0 (since d > 0)."""
    H = H / np.linalg.svd(H, compute_uv=False)[1]
    if np.sum(m2 * (m1 @ H.T)) < 0:
        H = -H
    w, V = np.linalg.eigh(H.T @ H)  # ascending: sigma_3^2 <= 1 <= sigma_1^2
    if w[2] - w[0] < 1e-9:  # pure rotation: t = 0, n undefined
        return H, [{"R": H, "t": np.zeros(3), "n": None, "n_m1": None, "valid": True}]
    v1, v2, v3 = V[:, 2], V[:, 1], V[:, 0]
    a, b = np.sqrt(max(1 - w[0], 0)), np.sqrt(max(w[2] - 1, 0))
    c = np.sqrt(w[2] - w[0])
    candidates = []
    for u in ((a * v1 + b * v3) / c, (a * v1 - b * v3) / c):
        U = np.column_stack([v2, u, np.cross(v2, u)])
        W = np.column_stack([H @ v2, H @ u, np.cross(H @ v2, H @ u)])
        R = W @ U.T
        n = np.cross(v2, u)
        t = (H - R) @ n
        candidates.append((R, t, n))
    candidates += [(R, -t, -n) for R, t, n in candidates]
    return H, [
        {"R": R, "t": t, "n": n, "n_m1": m1 @ n, "valid": bool(np.all(m1 @ n > 0))}
        for R, t, n in candidates
    ]


def view_scene(R: np.ndarray, t: np.ndarray, target: Target) -> dict[str, Any]:
    """A target seen by K[R|t]: corner pixels, which corners are visible (in front
    and inside the image), and the squares in front, far to near, for drawing."""
    z = (target.corners @ R.T + t)[:, 2]
    q = project(K @ np.column_stack([R, t]), target.corners)
    q = np.where(z[:, None] > 0, q, 0.0)
    valid = (
        (z > 0)
        & (q[:, 0] >= 0)
        & (q[:, 0] <= WIDTH)
        & (q[:, 1] >= 0)
        & (q[:, 1] <= HEIGHT)
    )
    front = np.flatnonzero(np.all(z[target.squares] > 0, axis=1))
    order = front[np.argsort(-z[target.squares[front]].mean(1))]
    return {
        "q": q,
        "valid": valid,
        "squares": target.squares[order],
        "dark": target.dark[order],
    }


def _keep(ids: list[int], selectable: np.ndarray) -> list[int]:
    """Selected corner ids that are currently selectable, without duplicates."""
    return list(
        dict.fromkeys(i for i in ids if 0 <= i < len(selectable) and selectable[i])
    )


def _to_world(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points - t) @ R


def _frustum(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return _to_world(frustum_segments(K, WIDTH, HEIGHT, FRUSTUM_DEPTH), R, t)


def rainbow(n: int) -> np.ndarray:
    """n colours [n, 3] (uint8) evenly spaced over matplotlib's rainbow colormap."""
    rgba = colormaps["rainbow"](np.linspace(0, 1, n))
    return np.round(255 * rgba[:, :3]).astype(np.uint8)


def _projections(
    R: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    X: np.ndarray,
    rgb: tuple,
    colors: np.ndarray,
) -> tuple[list, list]:
    """Camera (R, t) with its image plane (the frustum's far rectangle, filled)
    and, per picked point in its colour: the ray from the centre to the world
    point X (through the true pixel), and on the image plane a + where that ray
    crosses it, an x at the noisy pixel q and a line between them."""
    s = 0.001  # m, half a marker (~2 px); 1 px is FRUSTUM_DEPTH / fx ~ 0.5 mm there
    diagonals = np.array([[1, 1, 0], [1, -1, 0]]) / np.sqrt(2)
    X_cam = X @ R.T + t
    noisy = _homogeneous(q) @ np.linalg.inv(K).T * FRUSTUM_DEPTH
    lines = []
    for Xc, n, c in zip(X_cam, noisy, colors):
        p = Xc / Xc[2] * FRUSTUM_DEPTH
        segments = [[np.zeros(3), Xc]]
        if not np.allclose(p, n):  # without noise the link has no length
            segments.append([p, n])
        segments += [[p - s * e, p + s * e] for e in np.eye(3)[:2]]
        segments += [[n - s * e, n + s * e] for e in diagonals]
        lines.append((_to_world(np.array(segments), R, t), tuple(int(x) for x in c)))
    far = frustum_segments(K, WIDTH, HEIGHT, FRUSTUM_DEPTH)[4:, 0]
    plane = [_to_world(far[[0, 1, 2]], R, t), _to_world(far[[0, 2, 3]], R, t)]
    return lines, [(tri, (*rgb, PLANE_ALPHA)) for tri in plane]


def _cross_marker(p: np.ndarray, size: float = 0.03) -> np.ndarray:
    return np.stack([[p - size * e, p + size * e] for e in np.eye(3)])


def _homogeneous(q: np.ndarray) -> np.ndarray:
    return np.column_stack([q, np.ones(len(q))])


def _degenerate(q_R: np.ndarray) -> bool:
    """The target points don't determine H (e.g. 3 of 4 on a line): the noise-free
    DLT of q_R onto itself has a null space of more than one dimension. Noise
    would hide this, so the picked pixels are not used."""
    s = homography_dlt(q_R, q_R)[1]
    return s[-2] < 1e-9 * s[0]


def triangulation(
    R: np.ndarray, t: np.ndarray, noise: float, point: int | None
) -> dict[str, Any]:
    """Book target seen by P_A = K[I|0] and P_B = K[R|t]; with a selected corner,
    its (noisy) pixels in both views and the linear triangulation."""
    views = [
        view_scene(np.eye(3), np.zeros(3), BOOK_TARGET),
        view_scene(R, t, BOOK_TARGET),
    ]
    selectable = views[0]["valid"] & views[1]["valid"]
    out: dict[str, Any] = {"views": views, "selectable": selectable, "point": None}
    if point is None or not _keep([point], selectable):
        return out
    P_A = K @ np.eye(3, 4)
    P_B = K @ np.column_stack([R, t])
    q_A = noisy(views[0]["q"][[point]], 1, [point], noise)[0]
    q_B = noisy(views[1]["q"][[point]], 2, [point], noise)[0]
    A, s, Q_h = triangulate(P_A, P_B, q_A, q_B)
    Q_h = Q_h * np.sign(Q_h[3])
    Q = Q_h[:3] / Q_h[3]
    out.update(
        point=point,
        P_A=P_A,
        P_B=P_B,
        q_A=q_A,
        q_B=q_B,
        A=A,
        s=s,
        Q_h=Q_h,
        Q=Q,
        Q_true=BOOK_TARGET.corners[point],
        reprojection=[
            np.linalg.norm(project(P, Q[None])[0] - q)
            for P, q in ((P_A, q_A), (P_B, q_B))
        ],
    )
    return out


def triangulation_glb(
    R: np.ndarray, t: np.ndarray, noise: float, point: int | None
) -> bytes:
    """Book target, both cameras and, for a selected corner, the rays through its
    (noisy) pixels, the triangulated point (green) and the true corner (white)."""
    d = triangulation(R, t, noise, point)
    frustums = [
        (_frustum(np.eye(3), np.zeros(3)), CAMERA_1_RGB),
        (_frustum(R, t), CAMERA_2_RGB),
    ]
    if d["point"] is not None:
        Q_true, Kinv = d["Q_true"], np.linalg.inv(K)
        rays = []
        for (Rv, tv), q in (((np.eye(3), np.zeros(3)), d["q_A"]), ((R, t), d["q_B"])):
            z = (Rv @ Q_true + tv)[2]
            end = Kinv @ np.array([*q, 1.0]) * 1.3 * z
            rays.append(_to_world(np.stack([np.zeros(3), end]), Rv, tv))
        frustums += [
            (np.stack(rays), RAY_RGB),
            (_cross_marker(d["Q"]), ESTIMATE_RGB),
            (_cross_marker(Q_true), CAMERA_1_RGB),
        ]
    return to_glb(*target_samples(BOOK_TARGET), frustums)


def _colour(candidates: list[dict[str, Any]]) -> None:
    """Each candidate its own colour: greens if valid, reds if not, in order."""
    counts = {True: 0, False: 0}
    for c in candidates:
        palette = VALID_RGB if c["valid"] else INVALID_RGB
        c["rgb"] = palette[counts[c["valid"]] % len(palette)]
        counts[c["valid"]] += 1


def plane(R: np.ndarray, t: np.ndarray, noise: float, ids: list[int]) -> dict[str, Any]:
    """Calibration target (Z = 0) seen by K[R|t]; from >= 4 selected corners, the
    DLT homography and both pose candidates."""
    view = view_scene(R, t, PLANE_TARGET)
    ids = _keep(ids, view["valid"])
    q = noisy(view["q"][ids], 1, ids, noise)
    q_R = PLANE_TARGET.corners[ids, :2]
    H_true = K @ np.column_stack([R[:, 0], R[:, 1], t])
    out: dict[str, Any] = {
        "views": [view],
        "selectable": view["valid"],
        "ids": ids,
        "colors": rainbow(len(ids)),
        "q": q,
        "q_R": q_R,
        "R_true": R,
        "t_true": t,
        "H_true": H_true / H_true[2, 2],
    }
    if len(ids) < 4:
        out["message"] = f"Pick at least 4 points ({len(ids)} so far)."
        return out
    A, s, H = homography_dlt(q_R, q)
    out.update(A=A, s=s)
    if _degenerate(q_R):
        out["message"] = "Degenerate points (3 or more on a line): H is not determined."
        return out
    out.update(h=H.ravel(), H=H / H[2, 2])
    out["candidates"] = pose_from_homography(K, H, q_R)
    _colour(out["candidates"])
    return out


def plane_glb(R: np.ndarray, t: np.ndarray, noise: float, ids: list[int]) -> bytes:
    """Target, the true camera (white) with its image plane and the rays through
    the picked pixels, and each pose candidate (green if valid, red if not), in
    the target frame."""
    d = plane(R, t, noise, ids)
    frustums = [(_frustum(R, t), CAMERA_1_RGB)]
    for c in d.get("candidates", []):
        frustums.append((_frustum(c["R"], c["t"]), c["rgb"]))
    X = PLANE_TARGET.corners[d["ids"]]
    rays, image_plane = _projections(R, t, d["q"], X, CAMERA_1_RGB, d["colors"])
    return to_glb(*target_samples(PLANE_TARGET), frustums + rays, triangles=image_plane)


def twoview(
    R: np.ndarray, t: np.ndarray, noise: float, ids: list[int]
) -> dict[str, Any]:
    """Calibration target seen by camera 1 (fixed) and camera 2, X2 = R X1 + t;
    from >= 4 corners selected in both views, H1, H2, H21 = H2 H1^-1 and the
    decomposition of the calibrated K^-1 H21 K."""
    R1, t1 = CAMERA_1
    views = [
        view_scene(R1, t1, PLANE_TARGET),
        view_scene(R @ R1, R @ t1 + t, PLANE_TARGET),
    ]
    selectable = views[0]["valid"] & views[1]["valid"]
    ids = _keep(ids, selectable)
    q1 = noisy(views[0]["q"][ids], 1, ids, noise)
    q2 = noisy(views[1]["q"][ids], 2, ids, noise)
    # The target plane in camera 1: n^T Q1 = d, d > 0.
    n, d = R1[:, 2], R1[:, 2] @ t1
    if d < 0:
        n, d = -n, -d
    H_true = R + np.outer(t, n) / d
    H21_true = K @ H_true @ np.linalg.inv(K)
    out: dict[str, Any] = {
        "views": views,
        "selectable": selectable,
        "ids": ids,
        "colors": rainbow(len(ids)),
        "q1": q1,
        "q2": q2,
        "q_R": PLANE_TARGET.corners[ids, :2],
        "R_true": R,
        "t_d_true": t / d,
        "n_true": n,
        "d_true": d,
        "H_true": H_true,
        "H21_true": H21_true / H21_true[2, 2],
    }
    if len(ids) < 4:
        out["message"] = f"Pick at least 4 points ({len(ids)} so far)."
        return out
    if _degenerate(out["q_R"]):
        out["message"] = "Degenerate points (3 or more on a line): H is not determined."
        return out
    H1 = homography_dlt(out["q_R"], q1)[2]
    H2 = homography_dlt(out["q_R"], q2)[2]
    H21 = H2 @ np.linalg.inv(H1)
    Kinv = np.linalg.inv(K)
    m1, m2 = _homogeneous(q1) @ Kinv.T, _homogeneous(q2) @ Kinv.T
    Hc, candidates = decompose_homography(Kinv @ H21 @ K, m1, m2)
    _colour(candidates)
    error = [
        np.linalg.norm(c["R"] - R)
        + np.linalg.norm(c["t"] - t / d)
        + (0 if c["n"] is None else np.linalg.norm(c["n"] - n))
        for c in candidates
    ]
    out.update(
        H1=H1 / H1[2, 2],
        H2=H2 / H2[2, 2],
        H21=H21 / H21[2, 2],
        Hc=Hc,
        candidates=candidates,
        truth=int(np.argmin(error)),
    )
    return out


def _plane_patch(
    n: np.ndarray, d: float, near: np.ndarray, half: float = 0.2
) -> list[np.ndarray]:
    """Two triangles of a square on n^T X = d, centred on the point nearest `near`."""
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    centre = near - (n @ near - d) * n
    c = [
        centre + half * (a * u + b * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return [np.stack([c[0], c[1], c[2]]), np.stack([c[0], c[2], c[3]])]


def twoview_glb(R: np.ndarray, t: np.ndarray, noise: float, ids: list[int]) -> bytes:
    """In camera 1's frame: the target, camera 1 (white) and the true camera 2
    (cyan), each with its image plane and the rays through the picked pixels,
    and, per candidate, camera 2 and the plane in its colour. H only gives t/d, so
    the candidates are drawn with the true d."""
    data = twoview(R, t, noise, ids)
    R1, t1 = CAMERA_1
    points, colors = target_samples(PLANE_TARGET)
    frustums = [
        (_frustum(np.eye(3), np.zeros(3)), CAMERA_1_RGB),
        (_frustum(R, t), CAMERA_2_RGB),
    ]
    triangles = []
    X1 = PLANE_TARGET.corners[data["ids"]] @ R1.T + t1
    for (Rc, tc), q, rgb in (
        ((np.eye(3), np.zeros(3)), data["q1"], CAMERA_1_RGB),
        ((R, t), data["q2"], CAMERA_2_RGB),
    ):
        rays, image_plane = _projections(Rc, tc, q, X1, rgb, data["colors"])
        frustums += rays
        triangles += image_plane
    for c in data.get("candidates", []):
        frustums.append((_frustum(c["R"], c["t"] * data["d_true"]), c["rgb"]))
        if c["n"] is not None:
            # Centred near the target's centre (t1 in camera 1).
            for tri in _plane_patch(c["n"], data["d_true"], t1):
                triangles.append((tri, (*c["rgb"], PLANE_ALPHA)))
    return to_glb(points @ R1.T + t1, colors, frustums, triangles=triangles)


METRIC_RGB, SCALED_RGB = (0, 220, 80), (255, 140, 0)
DISTANCE_RGB = (255, 220, 0)


def metric_scale(
    R: np.ndarray, t: np.ndarray, noise: float, ids: list[int]
) -> dict[str, Any]:
    """The two-view scene, plus the metric scale H21 alone cannot give.

    H1 and H2 map the metric target, so each gives its camera's metric pose (as in
    the plane tab). Then R = R2 R1^T, t = t2 - R t1, and the plane in camera 1 is
    n = R1 e3, d = n^T t1. The metric n also picks the decomposition candidate
    (the valid one with the nearest n), whose t/d times d gives t again."""
    out = twoview(R, t, noise, ids)
    out["t_true"] = t
    if "candidates" not in out:
        return out
    poses = [
        max(
            pose_from_homography(K, H, out["q_R"]),
            key=lambda c: c["depths"].min(),
        )
        for H in (out["H1"], out["H2"])
    ]
    (Ra, ta), (Rb, tb) = ((p["R"], p["t"]) for p in poses)
    R_metric = Rb @ Ra.T
    n, d = Ra[:, 2], Ra[:, 2] @ ta
    if d < 0:
        n, d = -n, -d
    candidates = out["candidates"]
    valid = [i for i, c in enumerate(candidates) if c["valid"]] or list(
        range(len(candidates))
    )
    chosen = min(
        valid,
        key=lambda i: (
            0 if candidates[i]["n"] is None else np.linalg.norm(candidates[i]["n"] - n)
        ),
    )
    out.update(
        poses=poses,
        R_metric=R_metric,
        t_metric=tb - R_metric @ ta,
        n_metric=n,
        d_metric=d,
        chosen=chosen,
        t_scaled=candidates[chosen]["t"] * d,
    )
    return out


def metric_scale_glb(
    R: np.ndarray, t: np.ndarray, noise: float, ids: list[int]
) -> bytes:
    """In camera 1's frame: the target, camera 1 (white), the true camera 2 (cyan),
    camera 2 from the metric poses (green) and from the chosen H21 candidate
    scaled by the recovered d (orange), the recovered d as a segment from camera
    1 to the plane."""
    data = metric_scale(R, t, noise, ids)
    R1, t1 = CAMERA_1
    points, colors = target_samples(PLANE_TARGET)
    frustums = [
        (_frustum(np.eye(3), np.zeros(3)), CAMERA_1_RGB),
        (_frustum(R, t), CAMERA_2_RGB),
    ]
    triangles = []
    if "d_metric" in data:
        c = data["candidates"][data["chosen"]]
        n, d = data["n_metric"], data["d_metric"]
        frustums += [
            (_frustum(data["R_metric"], data["t_metric"]), METRIC_RGB),
            (_frustum(c["R"], data["t_scaled"]), SCALED_RGB),
            (np.array([[np.zeros(3), n * d]]), DISTANCE_RGB),
        ]
        # Large enough to hold both the target and the foot n d of the d segment.
        patch = _plane_patch(n, d, (t1 + n * d) / 2, half=0.4)
        triangles += [(tri, (*SCALED_RGB, PLANE_ALPHA)) for tri in patch]
    return to_glb(points @ R1.T + t1, colors, frustums, triangles=triangles)


def jsonable(x: Any) -> Any:
    """numpy values (also inside dicts, lists and tuples) as plain JSON types."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.ndarray, np.generic)):
        return x.tolist()
    return x
