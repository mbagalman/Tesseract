"""
tesseract_visualizer.py

Visualizes a 4D hypercube (a tesseract) by projecting it into 3D and 2D,
and lets you rotate it on all six 4D axes like some kind of higher-dimensional deity.

Includes interactive sliders (in Jupyter) so you can see what happens
when you casually warp space-time over coffee.

Author: Michael Bagalman
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
from itertools import product
from typing import List, Tuple, Optional, Dict

# Try to import ipywidgets so we can run interactively in notebooks.
# If not, we fall back to boring-but-functional static mode.
try:
    from ipywidgets import interact, FloatSlider, fixed
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    interact, FloatSlider, fixed = None, None, None
    IPYWIDGETS_AVAILABLE = False

# --- Constants ---
EPSILON = 1e-6  # Prevents divide-by-zero meltdowns
AXIS_COLORS = ['orange', 'green', 'blue', 'red']  # Axis-to-color decoder ring
AXIS_LABELS = ['X', 'Y', 'Z', 'W']  # Because you deserve to know who's who

# --- Core Geometric Functions ---

def generate_hypercube_vertices() -> np.ndarray:
    return np.array(list(product([-0.5, 0.5], repeat=4)))

def generate_hypercube_edges(vertices: np.ndarray) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isclose(np.sum(np.abs(vertices[i] - vertices[j])), 1.0):
                edges.append((i, j))
    return edges

def rotation_matrix_4d(axis1: int, axis2: int, angle_rad: float) -> np.ndarray:
    M = np.eye(4)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    M[axis1, axis1] = c
    M[axis1, axis2] = -s
    M[axis2, axis1] = s
    M[axis2, axis2] = c
    return M

def project_4d_to_3d(points4d: np.ndarray, viewer_distance: float = 3.0) -> np.ndarray:
    w = points4d[:, 3]
    denom = viewer_distance - w
    denom = np.where(np.abs(denom) < EPSILON, EPSILON, denom)
    factor = viewer_distance / denom
    return points4d[:, :3] * factor[:, np.newaxis]

def get_edge_color(vertices: np.ndarray, edge: Tuple[int, int]) -> str:
    diff = vertices[edge[0]] - vertices[edge[1]]
    axis = np.flatnonzero(diff)[0]
    return AXIS_COLORS[axis]

# --- Plotting and Interaction ---

def plot_tesseract(
    rot_xy: float = 0,
    rot_xz: float = 0,
    rot_xw: float = 0,
    rot_yz: float = 0,
    rot_yw: float = 0,
    rot_zw: float = 0,
    viewer_distance: float = 3.0,
    show_plot: bool = True
) -> Optional[plt.Figure]:
    verts = generate_hypercube_vertices()
    edges = generate_hypercube_edges(verts)

    R = (
        rotation_matrix_4d(2, 3, rot_zw) @
        rotation_matrix_4d(1, 3, rot_yw) @
        rotation_matrix_4d(0, 3, rot_xw) @
        rotation_matrix_4d(1, 2, rot_yz) @
        rotation_matrix_4d(0, 2, rot_xz) @
        rotation_matrix_4d(0, 1, rot_xy)
    )

    rv = verts @ R.T
    p3 = project_4d_to_3d(rv, viewer_distance)

    ecols = [get_edge_color(verts, e) for e in edges]
    wvals = rv[:, 3]
    wmin, wmax = wvals.min(), wvals.max()
    wnorm = (wvals - wmin) / (wmax - wmin) if not np.isclose(wmax, wmin) else np.full_like(wvals, 0.5)
    sizes = 20 + 40 * wnorm

    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_box_aspect((1, 1, 1))
    M = np.max(np.abs(p3)) * 1.15
    ax1.set(
        xlim=(-M, M), ylim=(-M, M), zlim=(-M, M),
        xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis',
        title='3D Perspective of a Tesseract'
    )
    lines = [p3[list(e)] for e in edges]
    ax1.add_collection3d(Line3DCollection(lines, colors=ecols, linewidths=1.5, alpha=0.9))
    sc = ax1.scatter(
        p3[:, 0], p3[:, 1], p3[:, 2],
        c=wvals, s=sizes, cmap='viridis', depthshade=True,
        edgecolors='black', linewidths=0.5
    )
    plt.colorbar(sc, ax=ax1, label='W-coordinate', shrink=0.7)

    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')
    ax2.set(xlabel='X-axis', ylabel='Y-axis', title='2D Projection (X vs Y)')
    ax2.grid(True, linestyle='--', alpha=0.3)
    for idx, (i, j) in enumerate(edges):
        p1, p2 = p3[i], p3[j]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color=ecols[idx], alpha=0.8, linewidth=1.2)
    ax2.scatter(
        p3[:, 0], p3[:, 1],
        c=wvals, s=sizes, cmap='viridis', alpha=0.9,
        edgecolors='black', linewidths=0.5
    )

    legend_elems = [
        Line2D([0], [0], color=c, lw=3, label=f'{l}-aligned')
        for c, l in zip(AXIS_COLORS, AXIS_LABELS)
    ]
    fig.legend(
        handles=legend_elems, loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, 0.01), frameon=True, fancybox=True
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if show_plot:
        plt.show()
        return None
    return fig

def create_interactive_visualization():
    if not IPYWIDGETS_AVAILABLE:
        raise ImportError("Install ipywidgets for interactive mode.")
    pi = np.pi
    sliders: Dict[str, FloatSlider] = {
        'rot_xy': FloatSlider(min=-pi, max=pi, step=0.05, value=0, description='XY (3D)'),
        'rot_xz': FloatSlider(min=-pi, max=pi, step=0.05, value=0, description='XZ (3D)'),
        'rot_yz': FloatSlider(min=-pi, max=pi, step=0.05, value=0, description='YZ (3D)'),
        'rot_xw': FloatSlider(min=-pi, max=pi, step=0.05, value=0, description='XW (4D)'),
        'rot_yw': FloatSlider(min=-pi, max=pi, step=0.05, value=0, description='YW (4D)'),
        'rot_zw': FloatSlider(min=-pi, max=pi, step=0.05, value=0, description='ZW (4D)'),
        'viewer_distance': FloatSlider(min=2.0, max=10.0, step=0.1, value=3.0, description='Viewer Dist'),
    }
    interact(plot_tesseract, **sliders, show_plot=fixed(True))

def is_jupyter_environment() -> bool:
    try:
        get_ipython()
        return True
    except NameError:
        return False

def demo_rotation_sequence():
    print("\nStatic demo of tesseract rotation...")
    plot_tesseract(
        rot_xy=np.pi/7,
        rot_xz=np.pi/8,
        rot_xw=np.pi/4,
        rot_yz=0,
        rot_yw=0,
        rot_zw=np.pi/3,
        viewer_distance=3.5
    )

if __name__ == "__main__":
    header = "\n".join([
        "="*60,
        "  4D Hypercube (Tesseract) Visualization",
        "="*60,
        "Visualize a 4D hypercube projected into 3D/2D.",
        "Edge colors map to axis orientation:",
        "  Orange (X), Green (Y), Blue (Z), Red (W)",
        "Vertex size/color reflect W-coordinate (4th dimension)",
    ])
    print(header)
    if is_jupyter_environment() and IPYWIDGETS_AVAILABLE:
        print("Jupyter + ipywidgets detected → interactive mode")
        create_interactive_visualization()
    else:
        demo_rotation_sequence()
