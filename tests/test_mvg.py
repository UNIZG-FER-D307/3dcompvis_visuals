import numpy as np
import pytest

from app import mvg
from app.geometry import rotation_matrix

K = mvg.K


def _P(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t[:, None]])


def _rotation_ok(R: np.ndarray) -> bool:
    return np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(
        np.linalg.det(R), 1.0
    )


def test_project_divides_by_third_coordinate() -> None:
    q = mvg.project(_P(np.eye(3), np.zeros(3)), np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_allclose(q, [[K[0, 2], K[1, 2]]])


def test_triangulate_recovers_point_exactly() -> None:
    R, t = rotation_matrix(5, 12, -3), np.array([-0.3, 0.05, 0.1])
    P_A, P_B = _P(np.eye(3), np.zeros(3)), _P(R, t)
    Q = np.array([0.1, -0.2, 1.5])
    q_A, q_B = mvg.project(P_A, Q[None])[0], mvg.project(P_B, Q[None])[0]
    A, s, Q_h = mvg.triangulate(P_A, P_B, q_A, q_B)
    np.testing.assert_allclose(Q_h[:3] / Q_h[3], Q, atol=1e-9)
    assert A.shape == (4, 4) and s[-1] < 1e-9 * s[0]


def test_triangulate_rows_follow_the_notes() -> None:
    rng = np.random.default_rng(1)
    P_A, P_B = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    (x_A, y_A), (x_B, y_B) = (3.0, 4.0), (5.0, 6.0)
    A, _, _ = mvg.triangulate(P_A, P_B, np.array([x_A, y_A]), np.array([x_B, y_B]))
    expected = [
        y_A * P_A[2] - P_A[1],
        x_A * P_A[2] - P_A[0],
        y_B * P_B[2] - P_B[1],
        x_B * P_B[2] - P_B[0],
    ]
    np.testing.assert_allclose(A, expected)


def _plane_homography(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


@pytest.mark.parametrize("n", [4, 10])
def test_homography_dlt_recovers_h_up_to_scale(n: int) -> None:
    H_true = _plane_homography(rotation_matrix(20, -10, 5), np.array([0.1, 0, 1.0]))
    q_R = np.random.default_rng(n).uniform(-0.2, 0.2, size=(n, 2))
    q = mvg.project(H_true, q_R)
    A, s, H = mvg.homography_dlt(q_R, q)
    assert A.shape == (2 * n, 9) and len(s) == 9 and s[-1] < 1e-9 * s[0]
    np.testing.assert_allclose(H / H[2, 2], H_true / H_true[2, 2], rtol=1e-7)


def test_homography_dlt_rows_follow_the_notes() -> None:
    q_R, q = np.array([[0.1, 0.2]] * 4), np.array([[30.0, 40.0]] * 4)
    A, _, _ = mvg.homography_dlt(q_R, q)
    r = np.array([0.1, 0.2, 1.0])
    np.testing.assert_allclose(A[0], np.concatenate([0 * r, -r, 40.0 * r]))
    np.testing.assert_allclose(A[1], np.concatenate([r, 0 * r, -30.0 * r]))


@pytest.mark.parametrize("scale", [2.5, -3.7])
def test_pose_from_homography_one_valid_candidate_is_truth(scale: float) -> None:
    R, t = rotation_matrix(25, -15, 5), np.array([0.05, -0.02, 1.0])
    q_R = mvg.PLANE_TARGET.corners[:, :2]
    candidates = mvg.pose_from_homography(K, scale * _plane_homography(R, t), q_R)
    assert len(candidates) == 2
    (valid,) = [c for c in candidates if c["valid"]]
    np.testing.assert_allclose(valid["R"], R, atol=1e-9)
    np.testing.assert_allclose(valid["t"], t, atol=1e-9)
    assert valid["r2_norm"] == pytest.approx(1) and valid["r1_dot_r2"] == pytest.approx(
        0, abs=1e-9
    )
    (invalid,) = [c for c in candidates if not c["valid"]]
    assert np.all(invalid["depths"] < 0)


def test_pose_from_noisy_homography_is_a_rotation() -> None:
    H = _plane_homography(rotation_matrix(25, -15, 5), np.array([0, 0, 1.0]))
    H = H + np.random.default_rng(0).normal(scale=1e-3 * np.abs(H).max(), size=(3, 3))
    for c in mvg.pose_from_homography(K, H, mvg.PLANE_TARGET.corners[:, :2]):
        assert _rotation_ok(c["R"])
        assert c["r2_norm"] != pytest.approx(1, abs=1e-6)


def _plane_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    R, t = rotation_matrix(3, 12, -2), np.array([-0.2, 0.03, 0.05])
    n, d = np.array([0.1, -0.3, 1.0]) / np.linalg.norm([0.1, -0.3, 1.0]), 1.2
    xy = np.random.default_rng(2).uniform(-0.3, 0.3, size=(6, 2))
    # Points on n^T X = d in camera 1.
    X1 = np.column_stack([xy, (d - xy @ n[:2]) / n[2]])
    return R, t, n, d, X1


@pytest.mark.parametrize("scale", [2.5, -0.4])
def test_decompose_homography_candidates(scale: float) -> None:
    R, t, n, d, X1 = _plane_scene()
    H = R + np.outer(t, n) / d
    X2 = X1 @ R.T + t
    m1, m2 = X1 / X1[:, 2:], X2 / X2[:, 2:]
    Hn, candidates = mvg.decompose_homography(scale * H, m1, m2)
    np.testing.assert_allclose(Hn, H, atol=1e-9)  # sigma_2(H) = 1, sign fixed
    assert len(candidates) == 4
    for c in candidates:
        assert _rotation_ok(c["R"])
        np.testing.assert_allclose(c["R"] + np.outer(c["t"], c["n"]), Hn, atol=1e-9)
        assert np.linalg.norm(c["n"]) == pytest.approx(1)
        np.testing.assert_allclose(c["n_m1"], m1 @ c["n"])
        assert c["valid"] == bool(np.all(c["n_m1"] > 0))
    # The point test can be stricter than Ma's n_3 > 0: one or two survive.
    valid = [c for c in candidates if c["valid"]]
    assert 1 <= len(valid) <= 2
    assert any(
        np.allclose(c["R"], R) and np.allclose(c["t"], t / d) and np.allclose(c["n"], n)
        for c in valid
    )


def test_decompose_pure_rotation_has_one_candidate() -> None:
    R = rotation_matrix(3, 12, -2)
    m1 = np.array([[0.1, 0.2, 1.0], [-0.1, 0.0, 1.0], [0.0, -0.2, 1.0]])
    m2 = m1 @ R.T
    Hn, candidates = mvg.decompose_homography(3.0 * R, m1, m2)
    (c,) = candidates
    np.testing.assert_allclose(c["R"], R, atol=1e-9)
    np.testing.assert_allclose(c["t"], 0, atol=1e-9)
    assert c["n"] is None and c["valid"]


def test_checkerboard_is_planar_and_alternates() -> None:
    board = mvg.checkerboard(3, 2, 0.1)
    assert board.corners.shape == (12, 3) and np.all(board.corners[:, 2] == 0)
    assert board.squares.shape == (6, 4) and board.dark.tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
    ]
    np.testing.assert_allclose(board.corners.mean(0), 0, atol=1e-12)  # centred


def test_book_target_pages_are_perpendicular_and_share_the_spine() -> None:
    book = mvg.BOOK_TARGET
    c = book.corners[book.squares]
    normals = np.cross(c[:, 1] - c[:, 0], c[:, 3] - c[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    unique = np.unique(np.round(normals, 6), axis=0)
    assert len(unique) == 2 and abs(unique[0] @ unique[1]) < 1e-9
    nu, nv = mvg.BOOK_PAGE
    assert len(book.corners) == 2 * (nu + 1) * (nv + 1) - (nv + 1)


def test_noise_is_fixed_per_corner_and_linear_in_sigma() -> None:
    q, ids = np.zeros((3, 2)), [4, 0, 7]
    np.testing.assert_array_equal(mvg.noisy(q, 1, ids, 0.0), q)
    one, two = mvg.noisy(q, 1, ids, 1.0), mvg.noisy(q, 1, ids, 2.0)
    np.testing.assert_allclose(two, 2 * one)
    np.testing.assert_array_equal(one, mvg.noisy(q, 1, ids, 1.0))
    assert not np.allclose(one, mvg.noisy(q, 2, ids, 1.0))
    np.testing.assert_array_equal(one[1], mvg.noisy(q[:1], 1, [0], 1.0)[0])


def test_target_samples_cover_squares_with_their_colours() -> None:
    board = mvg.checkerboard(2, 1, 0.1)
    points, colors = mvg.target_samples(board, per_edge=4)
    assert points.shape == (2 * 16, 3) and colors.shape == (2 * 16, 3)
    assert np.all(points[:16, 0] < 0) and np.all(points[16:, 0] > 0)
    assert len(np.unique(colors, axis=0)) == 2
