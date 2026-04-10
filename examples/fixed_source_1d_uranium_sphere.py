########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# One dimensional multigroup (87-group) fixed source problem in spherical
# geometry. A uranium-20% enriched core surrounded by depleted uranium and
# stainless steel shielding. A deuterium-tritium (14.1 MeV) source enters
# at the origin (reflective inner boundary).
#
########################################################################

import numpy as np

import ants
from ants.datatypes import GeometryData, SolverData, SourceData
from ants.fixed1d import fixed_source

# General conditions
cells_x = 1000
angles = 16
groups = 87
bc_x = [1, 0]  # reflective at origin, vacuum at surface

# Spatial
length = 10.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy grid
edges_g, edges_gidx = ants.energy_grid(87, groups)

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

# Sources: DT boundary source at inner surface (reflective boundary, location=1)
sources = SourceData(
    external=np.zeros((cells_x, 1, 1)),
    boundary_x=ants.boundary1d.deuterium_tritium(1, edges_g),
)

geometry = GeometryData(
    medium_map=medium_map,
    delta_x=delta_x,
    bc_x=bc_x,
    geometry=2,  # spherical geometry
)
solver = SolverData()

flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
# np.save("fixed_source_uranium_sphere", flux)
