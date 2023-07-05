########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
# 
# Default Problems for Testing
# 
########################################################################

import numpy as np

import ants

# Path for reference solutions
PATH = "data/references_multigroup/"

def reeds(boundary):
    # General conditions
    cells = 160
    angles = 4
    groups = 1

    # Different boundary conditions
    if boundary == [0, 0]:
        layout = [[0, "scattering", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"], \
                  [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]
    elif boundary == [0, 1]:
        layout = [[0, "scattering", "0-4"], [1, "vacuum", "4-5"], \
                  [2, "absorber", "5-6"], [3, "source", "6-8"]]
    elif boundary == [1, 0]:
        layout = [[0, "scattering", "4-8"], [1, "vacuum", "3-4"], \
                  [2, "absorber", "2-3"], [3, "source", "0-2"]]

    info = {
            "cells_x": cells,
            "angles": angles, 
            "groups": groups, 
            "materials": 4,
            "geometry": 1, 
            "spatial": 2, 
            "qdim": 3, 
            "bc_x": boundary,
            "bcdim_x": 1,
            "angular": False
            }

    # Spatial
    length = 8. if np.sum(boundary) > 0.0 else 16.
    delta_x = np.repeat(length / cells, cells)
    edges_x = np.linspace(0, length, cells+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Medium Map
    materials = np.array(layout)[:,1]
    medium_map = ants.spatial1d(layout, edges_x)

    # Angular
    angle_x, angle_w = ants.angular_x(info)

    # Cross Sections
    xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
    xs_scatter = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
    xs_fission = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])

    # External and boundary sources
    external = ants.externals1d("reeds", (cells, angles, groups), \
                              edges_x=edges_x, bc=info["bc_x"]).flatten()
    boundary_x = np.zeros((2,))
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info


def sphere_01(ptype):
    # ptype can be "timed", "fixed", or "critical"

    # General conditions
    cells = 100
    angles = 4
    groups = 87

    # Spatial
    length = 10.
    delta_x = np.repeat(length / cells, cells)
    edges_x = np.linspace(0, length, cells+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Energy Grid
    edges_g, edges_gidx = ants.energy_grid(groups, 87)
    velocity = ants.energy_velocity(groups, edges_g)

    # Medium Map
    materials = [[0, "uranium-%20%", "0-4"], [1, "uranium-%0%", "4-6"], \
                 [2, "stainless-steel-440", "6-10"]]
    medium_map = ants.spatial1d(materials, edges_x)

    # Cross Sections
    materials = np.array(materials)[:,1]
    xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

    # External and boundary sources
    external = ants.externals1d(0.0, (cells * angles * groups,))
    boundary_x = ants.boundaries1d("14.1-mev", (2, groups), [1], \
                                 energy_grid=edges_g).flatten()

    info = {
                "cells_x": cells,
                "angles": angles,
                "groups": groups,
                "materials": 3,
                "geometry": 2,
                "spatial": 2,
                "qdim": 3,
                "bc_x": [1, 0],
                "bcdim_x": 2
            }

    # Angular
    angle_x, angle_w = ants.angular_x(info)

    if ptype == "timed":
        info["steps"] = 5
        info["dt"] = 1e-8
        info["bcdecay_x"] = 2
        return xs_total, xs_scatter, xs_fission, velocity, external, \
            boundary_x, medium_map, delta_x, angle_x, angle_w, info

    elif ptype == "fixed":
        return xs_total, xs_scatter, xs_fission, external, \
            boundary_x, medium_map, delta_x, angle_x, angle_w, info

    elif ptype == "critical":
        info["bcdim_x"] = 1
        info["qdim"] = 2
        return xs_total, xs_scatter, xs_fission, medium_map, \
            delta_x, angle_x, angle_w, info


def manufactured_01(cells, angles):
    info = {"cells_x": cells, "angles": angles, "groups": 1, "materials": 1,
        "geometry": 1, "spatial": 2, "qdim": 3, "bc_x": [0, 0],
        "bcdim_x": 3, "angular": False, "edges": 0}
    length = 1.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals1d(1.0, (info["cells_x"] * info["angles"] * info["groups"],))
    boundary_x = ants.boundaries1d(1.0, (2, info["angles"], info["groups"]), [0]).flatten()
    angle_x, angle_w = ants.angular_x(info)
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_02(cells, angles):
    info = {"cells_x": cells, "angles": angles, "groups": 1, "materials": 1,
        "geometry": 1, "spatial": 2, "qdim": 3, "bc_x": [0, 0],
        "bcdim_x": 3, "angular": False, "edges": 0}
    length = 1.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals1d(0.5, (info["cells_x"] * info["angles"] * info["groups"],))
    boundary_x = ants.boundaries1d(1.0, (2, info["angles"], info["groups"]), [0]).flatten()
    angle_x, angle_w = ants.angular_x(info)
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_03(cells, angles):
    info = {"cells_x": cells, "angles": angles, "groups": 1, "materials": 1,
            "geometry": 1, "spatial": 2, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": False, "edges": 0}
    length = 1.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants.angular_x(info)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.9]]])
    xs_fission = np.array([[[0.0]]])
    external = ants.externals1d("mms-03", (info["cells_x"], info["angles"], \
                              info["groups"]), centers_x=centers_x, \
                              angle_x=angle_x).flatten()
    boundary_x = ants.boundaries1d("mms-03", (2, info["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_04(cells, angles):
    info = {"cells_x": cells, "angles": angles, "groups": 1, "materials": 2,
            "geometry": 1, "spatial": 2, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": False, "edges": 0}
    length = 2.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants.angular_x(info)
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])
    external = ants.externals1d("mms-04", (info["cells_x"], info["angles"]), \
                              centers_x=centers_x, angle_x=angle_x).flatten()
    boundary_x = ants.boundaries1d("mms-04", (2, info["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants.spatial1d(materials, edges_x)
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_05(cells, angles):
    info = {"cells_x": cells, "angles": angles, "groups": 1, "materials": 2,
            "geometry": 1, "spatial": 2, "qdim": 3, "bc_x": [0, 0],
            "bcdim_x": 3, "angular": False, "edges": 0}
    length = 2.
    delta_x = np.repeat(length / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    angle_x, angle_w = ants.angular_x(info)
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])
    external = ants.externals1d("mms-05", (info["cells_x"], info["angles"]), \
                              centers_x=centers_x, angle_x=angle_x).flatten()
    boundary_x = ants.boundaries1d("mms-05", (2, info["angles"]), [0, 1], \
                               angle_x=angle_x).flatten()
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants.spatial1d(materials, edges_x)
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x
