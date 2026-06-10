########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two dimensional, seven energy group criticality problem using multiple
# materials based off of M.A. Smith et al. "Benchmark on Deterministic
# Transport Calculations Without Spatial Homogenisation: A 2-D/3-D MOX
# Fuel Assembly Benchmark" OECD/NEA report, NEA/NSC/DOC(2003)16 (2003)
# Appendix A. Uses cross sections from "critical_2d_C5G7_xs.h5".
# Expected k-effective ≈ 1.19.
#
########################################################################

from pathlib import Path

import h5py
import numpy as np

import ants
from ants.critical2d import k_criticality
from ants.datatypes import GeometryData, MaterialData, SolverData

cells_x = 51 * 3
cells_y = 51 * 3
angles = 4
groups = 7
bc_x = [0, 1]  # vacuum at x=0, reflective at x=X
bc_y = [1, 0]  # reflective at y=0, vacuum at y=Y

length_x = 64.26
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.round(np.linspace(0, length_x, cells_x + 1), 12)

length_y = 64.26
delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.round(np.linspace(0, length_y, cells_y + 1), 12)

pin = 1.26
cylinder = np.pi * 0.54**2
mod_ratio = (pin**2 - cylinder) / (pin * pin)
pin_ratio = cylinder / (pin * pin)

# UO2 fuel assemblies - material index 1
uo2_assemblies = [
    [(34 * pin, 0), 17 * pin, 17 * pin],
    [(17 * pin, 17 * pin), 17 * pin, 17 * pin],
]

# 4.3% MOX fuel assemblies - material index 2
mox43_assemblies = [
    [(17 * pin, 0), 17 * pin, 17 * pin],
    [(34 * pin, 17 * pin), 17 * pin, 17 * pin],
]

# 7.0% MOX fuel assemblies - material index 3
mox70_assemblies = [
    [(17 * pin + pin, pin), 15 * pin, 15 * pin],
    [(34 * pin + pin, 17 * pin + pin), 15 * pin, 15 * pin],
]

# 8.7% MOX fuel assemblies - material index 4
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

# Guide tube assemblies - material index 5
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

# Fission chamber assemblies - material index 6
fc_assemblies = [
    [(34 * pin + 8 * pin, 8 * pin), pin, pin],
    [(17 * pin + 8 * pin, 8 * pin), pin, pin],
    [(34 * pin + 8 * pin, 17 * pin + 8 * pin), pin, pin],
    [(17 * pin + 8 * pin, 17 * pin + 8 * pin), pin, pin],
]

# Build medium map: moderator (0) is default, fuel/guides/chambers override
medium_map = np.zeros((cells_x, cells_y), dtype=np.int32)
medium_map = ants.spatial2d(medium_map, 1, uo2_assemblies, edges_x, edges_y)
medium_map = ants.spatial2d(medium_map, 2, mox43_assemblies, edges_x, edges_y)
medium_map = ants.spatial2d(medium_map, 3, mox70_assemblies, edges_x, edges_y)
medium_map = ants.spatial2d(medium_map, 4, mox87_assemblies, edges_x, edges_y)
medium_map = ants.spatial2d(medium_map, 5, gt_assemblies, edges_x, edges_y)
medium_map = ants.spatial2d(medium_map, 6, fc_assemblies, edges_x, edges_y)


def slurries(pin_ratio, pin_xs, mod_ratio, mod_xs):
    """Homogenize pin cell by volume-weighted cross sections."""
    keys = ["xs_total", "xs_scatter", "xs_fission"]
    return [pin_ratio * pin_xs[key][:] + mod_ratio * mod_xs[key][:] for key in keys]


data = h5py.File(Path(__file__).with_name("critical_2d_C5G7_xs.h5"), "r")

md_total = data["moderator"]["xs_total"][:]
md_scatter = data["moderator"]["xs_scatter"][:]
md_fission = data["moderator"]["xs_fission"][:]

uo2_total, uo2_scatter, uo2_fission = slurries(
    pin_ratio, data["UO2"], mod_ratio, data["moderator"]
)
mox43_total, mox43_scatter, mox43_fission = slurries(
    pin_ratio, data["MOX43"], mod_ratio, data["moderator"]
)
mox70_total, mox70_scatter, mox70_fission = slurries(
    pin_ratio, data["MOX70"], mod_ratio, data["moderator"]
)
mox87_total, mox87_scatter, mox87_fission = slurries(
    pin_ratio, data["MOX87"], mod_ratio, data["moderator"]
)
gt_total, gt_scatter, gt_fission = slurries(
    pin_ratio, data["guide_tubes"], mod_ratio, data["moderator"]
)
fc_total, fc_scatter, fc_fission = slurries(
    pin_ratio, data["fission_chamber"], mod_ratio, data["moderator"]
)

mat_data = MaterialData(
    total=np.array(
        [md_total, uo2_total, mox43_total, mox70_total, mox87_total, gt_total, fc_total]
    ),
    scatter=np.array(
        [
            md_scatter,
            uo2_scatter,
            mox43_scatter,
            mox70_scatter,
            mox87_scatter,
            gt_scatter,
            fc_scatter,
        ]
    ),
    fission=np.array(
        [
            md_fission,
            uo2_fission,
            mox43_fission,
            mox70_fission,
            mox87_fission,
            gt_fission,
            fc_fission,
        ]
    ),
)

# Angular quadrature
quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    delta_y=delta_y,
    bc_x=bc_x,
    bc_y=bc_y,
    geometry=3,  # 2D slab
)
solver = SolverData()

flux, keff = k_criticality(mat_data, geometry, quadrature, solver)
print(f"k-effective: {keff:.6f}")

np.savez("c5g7_flux", flux=flux, keff=keff)
np.savez("c5g7_flux", flux=flux, keff=keff)
