########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional method of manufactured solutions problem for a two-
# material slab with a quasi-scattering region (sigma_s = 0.3) and a
# scattering region (sigma_s = 0.9). Angle-independent boundary
# conditions, 2 cm domain.
#
########################################################################

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import ants
from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
from ants.fixed1d import fixed_source
from ants.utils import manufactured_1d as mms

cells_x = 100
angles = 4

length = 2.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

bc_x = [0, 0]

# Angular quadrature
quadrature = ants.angular_x(angles, bc_x=bc_x)

layers = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
medium_map = ants.spatial1d(layers, edges_x)

mat_data = MaterialData(
    total=np.array([[1.0], [1.0]]),
    scatter=np.array([[[0.3]], [[0.9]]]),
    fission=np.array([[[0.0]], [[0.0]]]),
)

sources = SourceData(
    external=ants.external1d.manufactured_ss_04(centers_x, quadrature.angle_x),
    boundary_x=ants.boundary1d.manufactured_ss_04(),
)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    bc_x=bc_x,
    geometry=1,
)
solver = SolverData(angular=True)

flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
exact = mms.solution_ss_04(centers_x, quadrature.angle_x)

colors = sns.color_palette("hls", angles)

fig, ax = plt.subplots()
for n in range(angles):
    ax.plot(
        centers_x, flux[:, n, 0], color=colors[n], alpha=0.6, label="Angle {}".format(n)
    )
    ax.plot(centers_x, exact[:, n], color=colors[n], ls=":")
ax.plot([], [], c="k", label="Approximate")
ax.plot([], [], c="k", ls=":", label="Analytical")
ax.legend(loc=0, framealpha=1)
ax.grid(which="both")
ax.set_xlabel("Location (cm)")
ax.set_ylabel("Angular Flux")
ax.set_title("Manufactured Solutions 04")
plt.show()
