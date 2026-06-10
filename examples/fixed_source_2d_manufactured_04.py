########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Two dimensional method of manufactured solutions problem with
# scattering (sigma_s = 0.5), angle-dependent external source, and
# angle-independent boundary conditions on a 2x2 cm domain.
# Solution 04: polynomial flux in space scaled by exponential in angle.
#
########################################################################

import matplotlib.pyplot as plt
import numpy as np

import ants
from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
from ants.fixed2d import fixed_source
from ants.utils import manufactured_2d as mms

cells_x = 50
cells_y = 50
angles = 4
bc_x = [0, 0]
bc_y = [0, 0]

length_x = 2.0
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0, length_x, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

length_y = 2.0
delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.linspace(0, length_y, cells_y + 1)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

# Angular quadrature
quadrature = ants.angular_xy(angles, bc_x=bc_x, bc_y=bc_y)

mat_data = MaterialData(
    total=np.array([[1.0]]),
    scatter=np.array([[[0.5]]]),
    fission=np.array([[[0.0]]]),
)

external = ants.external2d.manufactured_ss_04(
    centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
)
boundary_x, boundary_y = ants.boundary2d.manufactured_ss_04(
    centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
)

sources = SourceData(
    external=external,
    boundary_x=boundary_x,
    boundary_y=boundary_y,
)

geometry = GeometryData(
    medium_map=np.zeros((cells_x, cells_y), dtype=np.int32),
    delta_x=delta_x,
    delta_y=delta_y,
    bc_x=bc_x,
    bc_y=bc_y,
    geometry=3,  # 2D slab
)
solver = SolverData()

flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
flux = np.squeeze(flux)
exact = mms.solution_ss_04(centers_x, centers_y, quadrature.angle_x, quadrature.angle_y)
exact_scalar = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=(2, 3))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
img0 = axes[0].pcolormesh(centers_x, centers_y, flux.T, cmap="viridis")
axes[0].set_title("Numerical Flux")
fig.colorbar(img0, ax=axes[0])

img1 = axes[1].pcolormesh(centers_x, centers_y, exact_scalar.T, cmap="viridis")
axes[1].set_title("Exact Flux")
fig.colorbar(img1, ax=axes[1])

error = np.abs(flux - exact_scalar)
img2 = axes[2].pcolormesh(centers_x, centers_y, error.T, cmap="hot_r")
axes[2].set_title("Absolute Error")
fig.colorbar(img2, ax=axes[2])

for ax in axes:
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", "box")

fig.suptitle("2D Manufactured Solution 04")
plt.tight_layout()
plt.show()
