########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional monoenergetic, multi-region problem. Taken from Reed's
# "New Difference Schemes for the Neutron Transport Equation" (1971).
#
########################################################################

import matplotlib.pyplot as plt
import numpy as np

import ants
from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
from ants.fixed1d import fixed_source

# General conditions
cells_x = 160
angles = 4
bc_x = [0, 0]

# Spatial
length = 16.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Medium map
layout = [
    [0, "scattering", "0-4, 12-16"],
    [1, "vacuum", "4-5, 11-12"],
    [2, "absorber", "5-6, 10-11"],
    [3, "source", "6-10"],
]
medium_map = ants.spatial1d(layout, edges_x)

# Angular quadrature
quadrature = ants.angular_x(angles, bc_x=bc_x)

# Cross sections
mat_data = MaterialData(
    total=np.array([[1.0], [0.0], [5.0], [50.0]]),
    scatter=np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]]),
    fission=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
)

# Sources
sources = SourceData(
    external=ants.external1d.reeds(edges_x, bc_x),
    boundary_x=np.zeros((2, 1, 1)),
)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    bc_x=bc_x,
    geometry=1,
)
solver = SolverData(angular=True)

flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
# Convert to scalar flux
flux = np.sum(flux[:, :, 0] * quadrature.angle_w[None, :], axis=1)

fig, ax = plt.subplots()
ax.plot(centers_x, flux.flatten(), color="r", label="Scalar Flux")
ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Scalar Flux")
ax.set_title("Reed's Solution")
plt.show()
