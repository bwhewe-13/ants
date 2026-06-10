########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional k-eigenvalue (criticality) problem in spherical
# geometry. A uranium-20% enriched core surrounded by depleted uranium
# and stainless steel shielding, using 87 energy groups.
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
bc_x = [1, 0]  # reflective at origin, vacuum at surface

# Spatial
length = 10.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Angular quadrature
quadrature = ants.angular_x(angles, bc_x=bc_x)

# Medium map
layers = [
    [0, "uranium-%20%", "0-4"],
    [1, "uranium-%0%", "4-6"],
    [2, "stainless-steel-440", "6-10"],
]
medium_map = ants.spatial1d(layers, edges_x)

# Cross sections
mat_data = ants.materials(87, np.array(layers)[:, 1], datatype=True)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    bc_x=bc_x,
    geometry=2,  # spherical geometry
)
solver = SolverData()

flux, keff = k_criticality(mat_data, geometry, quadrature, solver)
print(f"k-effective: {keff:.6f}")

# np.save("critical_uranium_sphere_flux", flux)
# np.save("critical_uranium_sphere_keff", keff)
