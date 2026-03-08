import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import tesseract_visualizer as tv


def test_generate_hypercube_vertices_shape_and_count():
    verts = tv.generate_hypercube_vertices()
    assert verts.shape == (16, 4)
    assert set(np.unique(verts)) == {-0.5, 0.5}


def test_generate_hypercube_edges_count():
    verts = tv.generate_hypercube_vertices()
    edges = tv.generate_hypercube_edges(verts)
    assert len(edges) == 32
    assert len(set(edges)) == 32


def test_rotation_matrix_is_orthonormal():
    mat = tv.rotation_matrix_4d(0, 3, np.pi / 4)
    ident = mat.T @ mat
    assert np.allclose(ident, np.eye(4), atol=1e-10)


@pytest.mark.parametrize(
    "axis1, axis2",
    [
        (-1, 2),
        (0, 4),
        (1, 1),
        ("0", 2),
    ],
)
def test_rotation_matrix_rejects_invalid_axes(axis1, axis2):
    with pytest.raises(ValueError):
        tv.rotation_matrix_4d(axis1, axis2, 0.2)


def test_project_4d_to_3d_preserves_sign_near_projection_plane():
    vd = 3.0
    eps = tv.EPSILON / 2
    points = np.array(
        [
            [1.0, 0.0, 0.0, vd - eps],
            [1.0, 0.0, 0.0, vd + eps],
        ]
    )
    projected = tv.project_4d_to_3d(points, viewer_distance=vd)
    assert projected[0, 0] > 0
    assert projected[1, 0] < 0


@pytest.mark.parametrize(
    "points4d, viewer_distance",
    [
        (np.array([1.0, 2.0, 3.0, 4.0]), 3.0),
        (np.array([[1.0, 2.0, 3.0]]), 3.0),
        (np.array([[1.0, 2.0, 3.0, 4.0]]), 0.0),
        (np.array([[1.0, 2.0, 3.0, 4.0]]), np.inf),
    ],
)
def test_project_4d_to_3d_rejects_invalid_inputs(points4d, viewer_distance):
    with pytest.raises(ValueError):
        tv.project_4d_to_3d(points4d, viewer_distance=viewer_distance)


@pytest.mark.parametrize("bad_angle", [np.inf, np.nan, "abc"])
def test_plot_tesseract_rejects_invalid_rotation_inputs(bad_angle):
    with pytest.raises(ValueError):
        tv.plot_tesseract(rot_xy=bad_angle, show_plot=False)


def test_plot_tesseract_rejects_invalid_viewer_distance():
    with pytest.raises(ValueError):
        tv.plot_tesseract(viewer_distance=0.0, show_plot=False)
