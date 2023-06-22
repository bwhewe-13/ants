########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for large multigroup problems (G >= 87) and focuses on fixed
# source, time dependent, and criticality problems.
#
########################################################################

import pytest
import numpy as np

import ants
from ants import timed1d, fixed1d, critical1d

# Path for reference solutions
PATH = "data/references_multigroup/"

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
    medium_map = ants.spatial_map(materials, edges_x)

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
        info["bcdecay"] = 2
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


@pytest.mark.sphere1d
@pytest.mark.source_iteration
@pytest.mark.multigroup1d
def test_sphere_01_source_iteration():
    flux = fixed1d.source_iteration(*sphere_01("fixed"))
    reference = np.load(PATH + "uranium_sphere_source_iteration_flux.npy")
    assert np.isclose(flux, reference).all()


@pytest.mark.sphere1d
@pytest.mark.backward_euler
@pytest.mark.multigroup1d
def test_sphere_01_backward_euler():
    info = sphere_01("timed")[-1]
    flux = timed1d.backward_euler(*sphere_01("timed"))
    reference = np.load(PATH + "uranium_sphere_backward_euler_flux.npy")
    for tt in range(info["steps"]):
        assert np.isclose(flux[tt], reference[tt]).all()


@pytest.mark.sphere1d
@pytest.mark.power_iteration
@pytest.mark.multigroup1d
def test_sphere_01_power_iteration():
    flux, keff = critical1d.power_iteration(*sphere_01("critical"))
    reference_flux = np.load(PATH + "uranium_sphere_power_iteration_flux.npy")
    reference_keff = np.load(PATH + "uranium_sphere_power_iteration_keff.npy")
    assert np.isclose(flux, reference_flux).all()
    assert np.fabs(keff - reference_keff) < 1e-05
