"""4D+ geometric primitives and visualization helpers."""

from __future__ import annotations

import itertools
import logging
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import pyvista as pv  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    pv = None  # type: ignore
    LOGGER.warning("PyVista unavailable: %s", exc)


Point4D = Tuple[float, float, float, float]
Point3D = Tuple[float, float, float]
Edge = Tuple[int, int]


def generate_tesseract(scale: float = 1.0) -> Tuple[List[Point4D], List[Edge]]:
    """Return vertices and edges of a 4D hypercube (tesseract)."""

    coordinates = [-scale, scale]
    vertices = [tuple(p) for p in itertools.product(coordinates, repeat=4)]

    edges: List[Edge] = []
    for i, vi in enumerate(vertices):
        for j in range(i + 1, len(vertices)):
            vj = vertices[j]
            # Two vertices are connected if they differ in exactly one axis.
            if sum(1 for a, b in zip(vi, vj) if a != b) == 1:
                edges.append((i, j))
    return vertices, edges


def rotation_matrix_4d(angle_xy: float, angle_xw: float, angle_yw: float, angle_zw: float) -> np.ndarray:
    """Construct a 4D rotation matrix given planar angles."""

    rot = np.identity(4)

    def apply(axis_a: int, axis_b: int, theta: float) -> None:
        nonlocal rot
        sub = np.identity(4)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        sub[axis_a, axis_a] = cos_t
        sub[axis_b, axis_b] = cos_t
        sub[axis_a, axis_b] = -sin_t
        sub[axis_b, axis_a] = sin_t
        rot = sub @ rot

    apply(0, 1, angle_xy)
    apply(0, 3, angle_xw)
    apply(1, 3, angle_yw)
    apply(2, 3, angle_zw)
    return rot


def project_points(points: Sequence[Point4D], rotation: np.ndarray, perspective: float = 2.5) -> np.ndarray:
    """Rotate 4D points and project them into 3D space."""

    rotated = np.dot(np.array(points), rotation.T)
    w = rotated[:, 3]
    factor = perspective / (perspective - w)
    projected = rotated[:, :3] * factor[:, None]
    return projected


def build_polydata(points3d: np.ndarray, edges: Iterable[Edge]):
    """Create a PyVista mesh composed of line segments."""

    if pv is None:
        raise RuntimeError("PyVista is not available in this environment.")

    mesh = pv.PolyData(points3d)
    lines = []
    for start, end in edges:
        lines.extend([2, start, end])
    mesh.lines = np.array(lines)
    return mesh


class PolytopeRenderer:
    """Helper that renders and rotates 4D+ polytopes."""

    def __init__(self) -> None:
        if pv is None:
            LOGGER.warning("PolytopeRenderer initialized without PyVista support.")
        self._plotter = pv.Plotter() if pv is not None else None

    def display_tesseract(
        self,
        *,
        angle_xy: float = 0.0,
        angle_xw: float = 0.6,
        angle_yw: float = 0.3,
        angle_zw: float = 0.9,
        auto_close: bool = True,
    ) -> None:
        """Render a tesseract projection with configurable rotations."""

        vertices, edges = generate_tesseract()
        rotation = rotation_matrix_4d(angle_xy, angle_xw, angle_yw, angle_zw)
        projected = project_points(vertices, rotation)

        if self._plotter is None:
            LOGGER.info("Tesseract projected points:\n%s", projected)
            return

        self._plotter.clear()
        mesh = build_polydata(projected, edges)
        self._plotter.add_mesh(mesh, color="cyan", line_width=2)
        self._plotter.show_grid()
        self._plotter.add_title("4D Tesseract Projection")
        self._plotter.show(auto_close=auto_close)


__all__ = [
    "Point4D",
    "Point3D",
    "Edge",
    "generate_tesseract",
    "rotation_matrix_4d",
    "project_points",
    "build_polydata",
    "PolytopeRenderer",
]
