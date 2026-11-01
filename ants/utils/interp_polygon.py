########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Smooth curve fits on polygon meshes for nearby problems. A Legendre
# polynomial is fit to the cell averages of each material block by
# least squares, and integrated over the cells with Gauss quadrature
# on a triangle fan about each centroid.
#
########################################################################

import numpy as np
from numpy.polynomial import legendre


def polygon_quadrature(mesh, degree):
    """Gauss quadrature points over each polygon cell.

    Each cell is split into triangles about its centroid and each
    triangle uses a collapsed Gauss-Legendre rule.

    Parameters
    ----------
    mesh : PolygonMesh
        Polygon mesh.
    degree : int
        Polynomial degree to integrate exactly.

    Returns
    -------
    points : ndarray, shape (cells, Q, 2)
        Quadrature points [cm].
    weights : ndarray, shape (cells, Q)
        Quadrature weights [cm^2], summing to the cell area.
    """
    n = degree // 2 + 2
    xx, wx = np.polynomial.legendre.leggauss(n)
    xx = 0.5 * (xx + 1)
    wx = 0.5 * wx
    # Collapsed rule on the reference triangle (0, 0), (1, 0), (0, 1)
    uu, vv = np.meshgrid(xx, xx, indexing="ij")
    ref_x = (uu * (1 - vv)).flatten()
    ref_y = vv.flatten()
    ref_w = (wx[:, None] * wx[None, :] * (1 - vv)).flatten()

    max_faces = mesh.cell_vertices.shape[1]
    points = np.zeros((mesh.cells, max_faces * ref_w.size, 2))
    weights = np.zeros((mesh.cells, max_faces * ref_w.size))
    for cc in range(mesh.cells):
        cell = mesh.cell_vertices[cc][mesh.cell_vertices[cc] >= 0]
        v0 = mesh.cell_centroid[cc]
        for kk in range(cell.size):
            v1 = mesh.vertices[cell[kk]]
            v2 = mesh.vertices[cell[(kk + 1) % cell.size]]
            area = 0.5 * np.fabs(
                (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
            )
            idx = slice(kk * ref_w.size, (kk + 1) * ref_w.size)
            points[cc, idx] = (
                v0 + ref_x[:, None] * (v1 - v0) + ref_y[:, None] * (v2 - v0)
            )
            weights[cc, idx] = 2 * area * ref_w
    return points, weights


class PolygonInterpolation:
    """Least squares Legendre polynomial fit of cell averages.

    Parameters
    ----------
    psi : ndarray, shape (cells,)
        Cell average values.
    mesh : PolygonMesh
        Polygon mesh.
    medium_map : ndarray of int, shape (cells, 1)
        Material of each cell, used for block fits.
    block : bool, optional
        If True (default), fit each material separately.
    degree : int, optional
        Total polynomial degree. Default is 5. Reduced for blocks with
        too few cells.
    quadrature : tuple, optional
        Output of :func:`polygon_quadrature` to reuse between fits.
    """

    def __init__(self, psi, mesh, medium_map, block=True, degree=5, quadrature=None):
        self.mesh = mesh
        self.degree = degree
        if quadrature is None:
            quadrature = polygon_quadrature(mesh, degree)
        self.points, self.weights = quadrature
        self.cell_block = (
            np.asarray(medium_map).flatten()
            if block
            else np.zeros((mesh.cells,), dtype=np.int32)
        )
        psi = np.asarray(psi).flatten()

        self.blocks = {}
        for value in np.unique(self.cell_block):
            cells = np.argwhere(self.cell_block == value).flatten()
            points = self.points[cells].reshape(-1, 2)
            low = np.min(points, axis=0)
            high = np.max(points, axis=0)
            degree = self.degree
            while (degree + 1) * (degree + 2) // 2 > cells.size and degree > 0:
                degree -= 1
            terms = [
                (ii, jj) for ii in range(degree + 1) for jj in range(degree + 1 - ii)
            ]
            fit = {"low": low, "high": high, "terms": terms}
            # Cell average of each basis function
            basis = self._basis(fit, self.points[cells])
            matrix = np.sum(basis * self.weights[cells][..., None], axis=1)
            matrix /= mesh.cell_area[cells][:, None]
            fit["coefs"] = np.linalg.lstsq(matrix, psi[cells], rcond=None)[0]
            self.blocks[value] = fit

    def _scale(self, fit, points):
        center = 0.5 * (fit["high"] + fit["low"])
        half = 0.5 * (fit["high"] - fit["low"])
        half[half == 0.0] = 1.0
        return (points - center) / half, half

    def _basis(self, fit, points, derivative=None):
        # Legendre basis P_i(x) P_j(y) (or its x/y derivative) at points
        scaled, half = self._scale(fit, points)
        basis = np.zeros(points.shape[:-1] + (len(fit["terms"]),))
        for kk, (ii, jj) in enumerate(fit["terms"]):
            cx = np.zeros((ii + 1,))
            cx[ii] = 1.0
            cy = np.zeros((jj + 1,))
            cy[jj] = 1.0
            if derivative == "x":
                cx = legendre.legder(cx) / half[0]
            elif derivative == "y":
                cy = legendre.legder(cy) / half[1]
            basis[..., kk] = legendre.legval(scaled[..., 0], cx) * legendre.legval(
                scaled[..., 1], cy
            )
        return basis

    def _evaluate(self, points, cells, derivative=None):
        values = np.zeros(points.shape[:-1])
        for value, fit in self.blocks.items():
            idx = self.cell_block[cells] == value
            values[idx] = self._basis(fit, points[idx], derivative) @ fit["coefs"]
        return values

    def interpolate_centers(self):
        """Curve fit at the cell centroids, shape (cells,)."""
        cells = np.arange(self.mesh.cells)
        return self._evaluate(self.mesh.cell_centroid, cells)

    def interpolate_faces(self, faces):
        """Curve fit at face centers using the owner cell block, shape (faces,)."""
        faces = np.asarray(faces)
        cells = self.mesh.face_cells[faces, 0]
        return self._evaluate(self.mesh.face_center[faces], cells)

    def integrate_centers(self):
        """Cell integrals of the curve fit and its x and y derivatives.

        Returns
        -------
        int_psi, int_dx, int_dy : ndarray, shape (cells,)
        """
        cells = np.repeat(np.arange(self.mesh.cells), self.weights.shape[1])
        points = self.points.reshape(-1, 2)
        weights = self.weights.flatten()
        shape = (self.mesh.cells, -1)
        int_psi = np.sum(
            (self._evaluate(points, cells) * weights).reshape(shape), axis=1
        )
        int_dx = np.sum(
            (self._evaluate(points, cells, "x") * weights).reshape(shape), axis=1
        )
        int_dy = np.sum(
            (self._evaluate(points, cells, "y") * weights).reshape(shape), axis=1
        )
        return int_psi, int_dx, int_dy
