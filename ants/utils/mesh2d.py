########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Utility functions for generating 2D geometry weight matrices and
# medium maps for common neutron transport benchmark layouts.
#
# Layouts:
#   cylinder-1mat   - single fissile cylinder in void
#   cylinder-2mat   - fuel core + reflector annulus in void
#   double-chevron  - 4-triangle chevron pattern in moderator
#   c5g7            - 7-material C5G7 MOX benchmark (medium map only)
#
# Utilities:
#   resize_weight_matrix - expand or shrink a weight matrix
#
# CLI:
#   python -m ants.utils.mesh2d <layout> --cells-x N --cells-y N [opts]
#   python -m ants.utils.mesh2d --resize input.npy --cells-x N --cells-y N [opts]
#
########################################################################

import numpy as np
from scipy.ndimage import zoom as _ndimage_zoom

from ants.main import spatial2d, weight_matrix2d


def cylinder_1mat(
    cells_x,
    cells_y,
    length_x=None,
    length_y=None,
    radius=None,
    N_particles=None,
):
    """Weight matrix for a single fissile cylinder embedded in void.

    Parameters
    ----------
    cells_x : int
        Number of spatial cells in x.
    cells_y : int
        Number of spatial cells in y.
    length_x : float, optional
        Domain length in x [cm]. Default is ``2 * radius``.
    length_y : float, optional
        Domain length in y [cm]. Default is ``2 * radius``.
    radius : float, optional
        Cylinder radius [cm]. Default is 4.279960.
    N_particles : int, optional
        Monte Carlo samples. Default is ``cells_x * 50_000``.

    Returns
    -------
    weight_matrix : ndarray of float, shape (cells_x, cells_y, 2)
        Material weight fractions. Material 0 = fuel cylinder,
        material 1 = void background.
    """
    if radius is None:
        radius = 4.279960
    if length_x is None:
        length_x = 2.0 * radius
    if length_y is None:
        length_y = 2.0 * radius
    assert length_x == length_y, "cylinder layouts require a square domain"
    if N_particles is None:
        N_particles = cells_x * 50_000

    edges_x = np.linspace(0, length_x, cells_x + 1)
    edges_y = np.linspace(0, length_y, cells_y + 1)
    cx = length_x / 2.0
    cy = length_y / 2.0

    return weight_matrix2d(
        edges_x,
        edges_y,
        materials=2,
        N_particles=N_particles,
        circles=[[(cx, cy), (0.0, radius)]],
        circle_index=[0],
    )


def cylinder_2mat(
    cells_x,
    cells_y,
    length_x=None,
    length_y=None,
    r_inner=None,
    r_outer=None,
    N_particles=None,
):
    """Weight matrix for a fuel core + reflector annulus embedded in void.

    Parameters
    ----------
    cells_x : int
        Number of spatial cells in x.
    cells_y : int
        Number of spatial cells in y.
    length_x : float, optional
        Domain length in x [cm]. Default is ``2 * r_outer``.
    length_y : float, optional
        Domain length in y [cm]. Default is ``2 * r_outer``.
    r_inner : float, optional
        Inner (fuel) cylinder radius [cm]. Default is 3.0.
    r_outer : float, optional
        Outer (reflector) annulus radius [cm]. Default is 4.5.
    N_particles : int, optional
        Monte Carlo samples. Default is ``cells_x * 50_000``.

    Returns
    -------
    weight_matrix : ndarray of float, shape (cells_x, cells_y, 3)
        Material weight fractions. Material 0 = inner fuel, 1 = reflector annulus,
        2 = void background.
    """
    if r_inner is None:
        r_inner = 3.0
    if r_outer is None:
        r_outer = 4.5
    if length_x is None:
        length_x = 2.0 * r_outer
    if length_y is None:
        length_y = 2.0 * r_outer
    assert length_x == length_y, "cylinder layouts require a square domain"
    if N_particles is None:
        N_particles = cells_x * 50_000

    edges_x = np.linspace(0, length_x, cells_x + 1)
    edges_y = np.linspace(0, length_y, cells_y + 1)
    cx = length_x / 2.0
    cy = length_y / 2.0

    # circles[-1] must be the outer annulus so _quarter_symmetry uses r_outer
    return weight_matrix2d(
        edges_x,
        edges_y,
        materials=3,
        N_particles=N_particles,
        circles=[
            [(cx, cy), (0.0, r_inner)],
            [(cx, cy), (r_inner, r_outer)],
        ],
        circle_index=[0, 1],
    )


def double_chevron(cells_x, cells_y, N_particles=None):
    """Weight matrix for the 9 x 9 cm double-chevron benchmark geometry.

    Four triangular uranium fuel regions in an HDPE moderator background,
    matching the geometry encoded in examples/weight_matrix_2d_chevron.npy.

    Parameters
    ----------
    cells_x : int
        Number of spatial cells in x.
    cells_y : int
        Number of spatial cells in y.
    N_particles : int, optional
        Monte Carlo samples. Default is ``cells_x * cells_y * 40``.

    Returns
    -------
    weight_matrix : ndarray of float, shape (cells_x, cells_y, 2)
        Material weight fractions. Material 0 = uranium fuel,
        material 1 = HDPE background.
    """
    if N_particles is None:
        N_particles = cells_x * cells_y * 40

    length_x = 9.0
    length_y = 9.0

    edges_x = np.round(np.linspace(0, length_x, cells_x + 1), 10)
    edges_y = np.round(np.linspace(0, length_y, cells_y + 1), 10)

    triangles = np.array(
        [
            [(0.1, 1.0), (0.1, 3.9), (5.9, 1.0)],
            [(6.0, 1.0), (8.9, 1.0), (8.9, 3.9)],
            [(0.1, 4.9), (0.1, 7.8), (5.9, 4.9)],
            [(6.0, 4.9), (8.9, 4.9), (8.9, 7.8)],
        ]
    )
    rectangles = [
        [(0, 0), 0.1, 9.0],
        [(0, 8.9), 9.0, 0.1],
        [(8.9, 0), 0.1, 9.0],
    ]

    return weight_matrix2d(
        edges_x,
        edges_y,
        materials=2,
        N_particles=N_particles,
        triangles=triangles,
        triangle_index=[0, 0, 0, 0],
        rectangles=rectangles,
        rectangle_index=[0, 0, 0],
    )


def c5g7(cells_x=153, cells_y=153):
    """Medium map for the 2D C5G7 MOX fuel assembly benchmark.

    Based on M.A. Smith et al. NEA/NSC/DOC(2003)16, Appendix A.
    Returns an integer medium map (not a weight matrix) because the C5G7
    geometry consists entirely of rectangular regions with no curved boundaries.

    Parameters
    ----------
    cells_x : int, optional
        Number of spatial cells in x. Default is 153 (= 51 * 3).
    cells_y : int, optional
        Number of spatial cells in y. Default is 153 (= 51 * 3).

    Returns
    -------
    medium_map : ndarray of int, shape (cells_x, cells_y)
        Material index per cell. 0=moderator, 1=UO2, 2=MOX43, 3=MOX70, 4=MOX87,
        5=guide tube, 6=fission chamber.

    Raises
    ------
    ValueError
        If cells do not align with the 1.26 cm pin pitch.
    """
    pin = 1.26
    length_x = 64.26
    length_y = 64.26

    # Validate pin-pitch alignment
    pins_per_length = length_x / pin
    if abs(round(pins_per_length) - pins_per_length) > 1e-9:
        raise ValueError(
            f"length_x={length_x} is not a multiple of pin pitch {pin}. "
            f"Use cells_x that is a multiple of {round(pins_per_length)}."
        )
    cells_per_pin = cells_x / pins_per_length
    if abs(round(cells_per_pin) - cells_per_pin) > 1e-9:
        raise ValueError(
            f"cells_x={cells_x} does not divide evenly into"
            f" {pins_per_length:.0f} pins. "
            f"Try cells_x = 51*k (e.g. 51, 102, 153)."
        )

    edges_x = np.round(np.linspace(0, length_x, cells_x + 1), 12)
    edges_y = np.round(np.linspace(0, length_y, cells_y + 1), 12)

    medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)

    # UO2 assemblies (material 1)
    uo2_assemblies = [
        [(34 * pin, 0), 17 * pin, 17 * pin],
        [(17 * pin, 17 * pin), 17 * pin, 17 * pin],
    ]
    medium_map = spatial2d(medium_map, 1, uo2_assemblies, edges_x, edges_y)

    # MOX 4.3% assemblies (material 2)
    mox43_assemblies = [
        [(17 * pin, 0), 17 * pin, 17 * pin],
        [(34 * pin, 17 * pin), 17 * pin, 17 * pin],
    ]
    medium_map = spatial2d(medium_map, 2, mox43_assemblies, edges_x, edges_y)

    # MOX 7.0% assemblies (material 3)
    mox70_assemblies = [
        [(17 * pin + pin, pin), 15 * pin, 15 * pin],
        [(34 * pin + pin, 17 * pin + pin), 15 * pin, 15 * pin],
    ]
    medium_map = spatial2d(medium_map, 3, mox70_assemblies, edges_x, edges_y)

    # MOX 8.7% assemblies (material 4)
    mox87_assemblies = [
        [(17 * pin + 4 * pin, 4 * pin), 9 * pin, 9 * pin],
        [(17 * pin + 5 * pin, 3 * pin), 7 * pin, pin],
        [(17 * pin + 3 * pin, 5 * pin), pin, 7 * pin],
        [(17 * pin + 13 * pin, 5 * pin), pin, 7 * pin],
        [(17 * pin + 5 * pin, 13 * pin), 7 * pin, pin],
        [(34 * pin + 4 * pin, 17 * pin + 4 * pin), 9 * pin, 9 * pin],
        [(34 * pin + 5 * pin, 17 * pin + 3 * pin), 7 * pin, pin],
        [(34 * pin + 3 * pin, 17 * pin + 5 * pin), pin, 7 * pin],
        [(34 * pin + 13 * pin, 17 * pin + 5 * pin), pin, 7 * pin],
        [(34 * pin + 5 * pin, 17 * pin + 13 * pin), 7 * pin, pin],
    ]
    medium_map = spatial2d(medium_map, 4, mox87_assemblies, edges_x, edges_y)

    # Guide tubes (material 5)
    gt_idx = [
        (5, 2),
        (8, 2),
        (11, 2),
        (3, 3),
        (13, 3),
        (2, 5),
        (5, 5),
        (8, 5),
        (11, 5),
        (14, 5),
        (2, 8),
        (5, 8),
        (11, 8),
        (14, 8),
        (2, 11),
        (5, 11),
        (8, 11),
        (11, 11),
        (14, 11),
        (3, 13),
        (13, 13),
        (5, 14),
        (8, 14),
        (11, 14),
    ]
    gt_assemblies = []
    for x, y in gt_idx:
        gt_assemblies.append([(34 * pin + x * pin, y * pin), pin, pin])
        gt_assemblies.append([(17 * pin + x * pin, 17 * pin + y * pin), pin, pin])
        gt_assemblies.append([(17 * pin + x * pin, y * pin), pin, pin])
        gt_assemblies.append([(34 * pin + x * pin, 17 * pin + y * pin), pin, pin])
    medium_map = spatial2d(medium_map, 5, gt_assemblies, edges_x, edges_y)

    # Fission chambers (material 6)
    fc_assemblies = [
        [(34 * pin + 8 * pin, 8 * pin), pin, pin],
        [(17 * pin + 8 * pin, 8 * pin), pin, pin],
        [(34 * pin + 8 * pin, 17 * pin + 8 * pin), pin, pin],
        [(17 * pin + 8 * pin, 17 * pin + 8 * pin), pin, pin],
    ]
    medium_map = spatial2d(medium_map, 6, fc_assemblies, edges_x, edges_y)

    return medium_map


def resize_weight_matrix(weight_matrix, new_cells_x, new_cells_y):
    """Expand or shrink a weight matrix (or medium map) to a new grid size.

    Parameters
    ----------
    weight_matrix : ndarray
        Float array of shape (cells_x, cells_y, materials), or integer array
        of shape (cells_x, cells_y).
    new_cells_x : int
        Target number of cells in x.
    new_cells_y : int
        Target number of cells in y.

    Returns
    -------
    resized : ndarray
        Array with shape (new_cells_x, new_cells_y, materials) for float inputs,
        or (new_cells_x, new_cells_y) for integer medium maps. Float outputs are
        renormalized so weights sum to 1 along axis 2.
    """
    zoom_x = new_cells_x / weight_matrix.shape[0]
    zoom_y = new_cells_y / weight_matrix.shape[1]

    if weight_matrix.ndim == 2 and np.issubdtype(weight_matrix.dtype, np.integer):
        # Integer medium map: nearest-neighbour zoom
        resized = _ndimage_zoom(weight_matrix.astype(float), (zoom_x, zoom_y), order=0)
        return resized.astype(np.int32)

    # Float weight matrix: per-material bilinear zoom then renormalize
    n_materials = weight_matrix.shape[2]
    result = np.zeros((new_cells_x, new_cells_y, n_materials))
    for m in range(n_materials):
        result[..., m] = _ndimage_zoom(
            weight_matrix[..., m], (zoom_x, zoom_y), order=1, prefilter=False
        )
    # Clip negatives introduced by bilinear interpolation at boundaries
    result = np.clip(result, 0.0, None)
    sums = result.sum(axis=2, keepdims=True)
    sums[sums == 0.0] = 1.0
    return result / sums


def _plot_result(arr):
    import matplotlib.pyplot as plt

    if arr.ndim == 3:
        n_mat = arr.shape[2]
        fig, axes = plt.subplots(1, n_mat, figsize=(4 * n_mat, 4))
        if n_mat == 1:
            axes = [axes]
        for m, ax in enumerate(axes):
            im = ax.imshow(arr[:, :, m].T, origin="lower", vmin=0, vmax=1)
            ax.set_title(f"Material {m}")
            plt.colorbar(im, ax=ax)
    else:
        n_mat = int(arr.max()) + 1
        cmap = plt.get_cmap("tab10", n_mat)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=-0.5, vmax=n_mat - 0.5)
        cbar = plt.colorbar(im, ax=ax, ticks=range(n_mat))
        cbar.set_label("Material index")
        ax.set_title("Medium map")

    plt.tight_layout()
    plt.show()


_LAYOUTS = ("cylinder-1mat", "cylinder-2mat", "double-chevron", "c5g7")

_DISPATCH = {
    "cylinder-1mat": cylinder_1mat,
    "cylinder-2mat": cylinder_2mat,
    "double-chevron": double_chevron,
    "c5g7": c5g7,
}

_LAYOUT_KWARGS = {
    "cylinder-1mat": ("radius", "length_x", "length_y", "n_particles"),
    "cylinder-2mat": ("r_inner", "r_outer", "length_x", "length_y", "n_particles"),
    "double-chevron": ("n_particles"),
    "c5g7": (),
}


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ants.utils.mesh2d",
        description="Generate or resize 2D neutron transport geometry layouts.",
    )
    parser.add_argument(
        "layout",
        nargs="?",
        choices=_LAYOUTS,
        help="Geometry layout to generate",
    )
    parser.add_argument(
        "--resize",
        metavar="INPUT.NPY",
        help=(
            "Path to existing weight matrix .npy file to resize"
            " (mutually exclusive with layout)"
        ),
    )
    parser.add_argument(
        "--cells-x", type=int, required=True, help="Number of cells in x"
    )
    parser.add_argument(
        "--cells-y", type=int, required=True, help="Number of cells in y"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show matplotlib plot of result"
    )
    parser.add_argument("--save", metavar="OUTPUT.NPY", help="Save result to .npy file")
    # Layout-specific overrides
    parser.add_argument(
        "--radius", type=float, help="[cylinder_1mat] cylinder radius [cm]"
    )
    parser.add_argument(
        "--r-inner",
        type=float,
        dest="r_inner",
        help="[cylinder-2mat] inner radius [cm]",
    )
    parser.add_argument(
        "--r-outer",
        type=float,
        dest="r_outer",
        help="[cylinder-2mat] outer radius [cm]",
    )
    parser.add_argument(
        "--length-x", type=float, dest="length_x", help="Domain length in x [cm]"
    )
    parser.add_argument(
        "--length-y", type=float, dest="length_y", help="Domain length in y [cm]"
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        dest="n_particles",
        help="[MC layouts] number of MC samples",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.layout is None and args.resize is None:
        parser.error("specify a layout name or --resize INPUT.NPY")
    if args.layout is not None and args.resize is not None:
        parser.error("layout and --resize are mutually exclusive")

    if args.resize is not None:
        data = np.load(args.resize)
        result = resize_weight_matrix(data, args.cells_x, args.cells_y)
    else:
        fn = _DISPATCH[args.layout]
        allowed = _LAYOUT_KWARGS[args.layout]
        kwargs = {k: v for k in allowed if (v := getattr(args, k, None)) is not None}
        result = fn(args.cells_x, args.cells_y, **kwargs)

    print(f"Result shape: {result.shape}")

    if args.save:
        np.save(args.save, result)
        print(f"Saved to {args.save}")

    if args.plot:
        _plot_result(result)
