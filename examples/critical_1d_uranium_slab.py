########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional k-eigenvalue (criticality) problem for a uranium-
# hydride slab with 20% enrichment surrounded by HDPE moderator and
# depleted uranium reflector, using 87 energy groups.
#
########################################################################

import numpy as np

import ants
from ants.critical1d import k_criticality
from ants.datatypes import GeometryData, SolverData

# General conditions
cells_x = 1000
angles = 8
groups = 87
bc_x = [0, 1]  # vacuum at x=0, reflective at x=X
enrich = "20"

# Spatial
length = 100.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Angular quadrature
quadrature = ants.angular_x(angles, bc_x=bc_x)

# Medium map
layers = [
    [0, "high-density-polyethyene-087", "0-45"],
    [1, "uranium-hydride-%{}%".format(enrich), "45-80"],
    [2, "uranium-hydride-%0%", "80-100"],
]
medium_map = ants.spatial1d(layers, edges_x)

# Cross sections
mat_data = ants.materials(87, np.array(layers)[:, 1], datatype=True)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    bc_x=bc_x,
    geometry=1,
)
solver = SolverData()

flux, keff = k_criticality(mat_data, geometry, quadrature, solver)
print(f"k-effective: {keff:.6f}")

# np.save(f"critical_uranium_{enrich}_hdpe_slab_flux", flux)
# np.save(f"critical_uranium_{enrich}_hdpe_slab_keff", keff)
