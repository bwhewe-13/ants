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
from ants.utils import manufactured_1d as mms

# Path for reference solutions
PATH = "data/references_multigroup/"


def reeds(bc_x):
    # General conditions
    cells_x = 160
    angles = 4
    groups = 1

    info = {
            "cells_x": cells_x,
            "angles": angles, 
            "groups": groups, 
            "materials": 4,
            "geometry": 1, 
            "spatial": 2, 
            "bc_x": bc_x,
            "angular": False
            }

    # Spatial
    length_x = 8. if np.sum(bc_x) > 0.0 else 16.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(info)

    # Layout and Materials
    if bc_x == [0, 0]:
        layout = [[0, "scattering", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"], \
                  [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]
    elif bc_x == [0, 1]:
        layout = [[0, "scattering", "0-4"], [1, "vacuum", "4-5"], \
                  [2, "absorber", "5-6"], [3, "source", "6-8"]]
    elif bc_x == [1, 0]:
        layout = [[0, "scattering", "4-8"], [1, "vacuum", "3-4"], \
                  [2, "absorber", "2-3"], [3, "source", "0-2"]]
    medium_map = ants.spatial1d(layout, edges_x)

    xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
    xs_scatter = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
    xs_fission = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])
    
    # Sources
    external = ants.external1d.reeds(edges_x, bc_x)
    boundary_x = np.zeros((2, 1, 1))

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info


def sphere_01(ptype):
    # ptype can be "timed", "fixed", or "critical"

    # General conditions
    cells_x = 100
    angles = 4
    groups = 87

    info = {
                "cells_x": cells_x,
                "angles": angles,
                "groups": groups,
                "materials": 3,
                "geometry": 2,
                "spatial": 2,
                "bc_x": [1, 0]
            }

    # Spatial
    length_x = 10.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(info)

    # Energy Grid
    edges_g, edges_gidx = ants.energy_grid(groups, 87)
    velocity = ants.energy_velocity(groups, edges_g)

    # Layout and Materials
    layout = [[0, "uranium-%20%", "0-4"], [1, "uranium-%0%", "4-6"], \
                 [2, "stainless-steel-440", "6-10"]]
    medium_map = ants.spatial1d(layout, edges_x)

    materials = np.array(layout)[:,1]
    xs_total, xs_scatter, xs_fission = ants.materials(groups, materials)

    # Sources
    external = np.zeros((cells_x, 1, 1))
    boundary_x = ants.boundary1d.deuterium_tritium(1, edges_g)

    if ptype == "timed":
        info["steps"] = 5
        info["dt"] = 1e-8
        initial_flux = np.zeros((info["cells_x"], info["angles"], info["groups"]))
        external = external[None,...].copy()
        boundary_x = boundary_x[None,...].copy()

        return initial_flux, xs_total, xs_scatter, xs_fission, velocity, \
            external, boundary_x, medium_map, delta_x, angle_x, angle_w, info

    elif ptype == "fixed":
        return xs_total, xs_scatter, xs_fission, external, \
            boundary_x, medium_map, delta_x, angle_x, angle_w, info

    elif ptype == "critical":
        info["bcdim_x"] = 1
        info["qdim"] = 2
        return xs_total, xs_scatter, xs_fission, medium_map, \
            delta_x, angle_x, angle_w, info


def manufactured_ss_01(cells_x, angles):
    info = {"cells_x": cells_x, "angles": angles, "groups": 1, \
            "materials": 1, "geometry": 1, "spatial": 2, "bc_x": [0, 0], \
            "angular": False, "edges": 0}

    # Spatial
    length_x = 1.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(info)
    
    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])

    # Externals
    external = np.ones((info["cells_x"], 1, 1))
    boundary_x = np.zeros((2, 1, 1))
    boundary_x[0] = 1.

    # Layout
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_ss_02(cells_x, angles):
    info = {"cells_x": cells_x, "angles": angles, "groups": 1, \
            "materials": 1, "geometry": 1, "spatial": 2, "bc_x": [0, 0],
            "angular": False, "edges": 0}

    # Spatial
    length_x = 1.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(info)
    
    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])

    # Sources
    external = 0.5 * np.ones((info["cells_x"], 1, 1))
    boundary_x = np.zeros((2, 1, 1))
    boundary_x[0] = 1.

    # Layout
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)
    
    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_ss_03(cells_x, angles):
    info = {"cells_x": cells_x, "angles": angles, "groups": 1, \
            "materials": 1, "geometry": 1, "spatial": 2, "bc_x": [0, 0],
            "angular": False, "edges": 0}

    # Spatial
    length_x = 1.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(info)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.9]]])
    xs_fission = np.array([[[0.0]]])

    # Sources
    external = ants.external1d.manufactured_ss_03(centers_x, angle_x)
    boundary_x = ants.boundary1d.manufactured_ss_03(angle_x)

    # Layout
    medium_map = np.zeros((info["cells_x"]), dtype=np.int32)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_ss_04(cells_x, angles):
    info = {"cells_x": cells_x, "angles": angles, "groups": 1, \
            "materials": 2, "geometry": 1, "spatial": 2, "bc_x": [0, 0], \
            "angular": False, "edges": 0}

    # Spatial
    length_x = 2.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(info)

    # Materials
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])

    # Sources
    external = ants.external1d.manufactured_ss_04(centers_x, angle_x)
    boundary_x = ants.boundary1d.manufactured_ss_04()

    # Layout
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants.spatial1d(materials, edges_x)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_ss_05(cells_x, angles):
    info = {"cells_x": cells_x, "angles": angles, "groups": 1, \
            "materials": 2, "geometry": 1, "spatial": 2, "bc_x": [0, 0],
            "angular": False, "edges": 0}
    
    # Spatial
    length_x = 2.
    delta_x = np.repeat(length_x / info["cells_x"], info["cells_x"])
    edges_x = np.linspace(0, length_x, info["cells_x"]+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(info)
    
    # Materials
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])

    # Sources
    external = ants.external1d.manufactured_ss_05(centers_x, angle_x)
    boundary_x = ants.boundary1d.manufactured_ss_05()

    # Layout
    layout = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = ants.spatial1d(layout, edges_x)

    return xs_total, xs_scatter, xs_fission, external, boundary_x, \
        medium_map, delta_x, angle_x, angle_w, info, edges_x, centers_x


def manufactured_td_01(cells_x, angles, edges_t, dt, temporal=1):
    info = {"cells_x": cells_x, "angles": angles, "groups": 1, \
            "materials": 1, "geometry": 1, "spatial": 2, "bc_x": [0, 0],
            "angular": False, "steps": edges_t.shape[0] - 1, "dt": dt}

    # Spatial
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = ants.angular_x(angles)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    velocity = np.ones((info["groups"],))

    # Sources
    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_01(centers_x, angle_x, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_01(centers_x, angle_x, edges_t)[1:]
        boundary_x = ants.boundary1d.manufactured_td_01(angle_x, edges_t)[1:]
    # Crank Nicolson
    elif temporal == 2:
        initial_flux = mms.solution_td_01(edges_x, angle_x, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_01(centers_x, angle_x, edges_t)
        boundary_x = ants.boundary1d.manufactured_td_01(angle_x, edges_t)[1:]
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_01(centers_x, angle_x, np.array([0.0]))[0]
        external = ants.external1d.manufactured_td_01(centers_x, angle_x, edges_t)[1:]
        boundary_x = ants.boundary1d.manufactured_td_01(angle_x, edges_t)[1:]
    # TR-BDF2
    elif temporal == 4:    
        initial_flux = mms.solution_td_01(edges_x, angle_x, np.array([0.0]))[0]
        gamma_steps = ants.gamma_time_steps(edges_t)
        external = ants.external1d.manufactured_td_01(centers_x, angle_x, gamma_steps)
        boundary_x = ants.boundary1d.manufactured_td_01(angle_x, edges_t)[1:]

    # Layout
    medium_map = np.zeros((cells_x), dtype=np.int32)

    return initial_flux, xs_total, xs_scatter, xs_fission, velocity, \
        external, boundary_x, medium_map, delta_x, angle_x, angle_w, info
