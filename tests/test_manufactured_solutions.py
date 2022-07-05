########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Method of Manufactured Solutions for One Dimensional Slabs
#
########################################################################

from ants.transport import Transport
from ants.cyants import multi_group
from ants.utils import manufactured_solutions as mms

import pytest
import numpy as np

@pytest.mark.smoke
@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_one_material01(angular):
    file = "../input-files/mms-one-material01.inp"
    problem = Transport(file)
    problem.change_param("spatial x cells", 400)
    angular = True if angular == "True" else False
    flux = multi_group.source_iteration(problem.medium_map, problem.xs_total, \
                            problem.xs_scatter, problem.xs_fission, \
                            problem.external_source, problem.point_source, \
                            problem.mu, problem.angle_weight, problem.params, \
                            problem.cell_width, angular=angular)
    xspace = np.linspace(0, problem.cell_width * problem.cells, problem.cells+1)
    xspace = 0.5 * (xspace[1:] + xspace[:-1])
    ref_flux = mms.solution_one_material_01(xspace, problem.mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * problem.angle_weight, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)


@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_one_material02(angular):
    file = "../input-files/mms-one-material02.inp"
    problem = Transport(file)
    problem.change_param("spatial x cells", 400)
    angular = True if angular == "True" else False
    flux = multi_group.source_iteration(problem.medium_map, problem.xs_total, \
                            problem.xs_scatter, problem.xs_fission, \
                            problem.external_source, problem.point_source, \
                            problem.mu, problem.angle_weight, problem.params, \
                            problem.cell_width, angular=angular)
    xspace = np.linspace(0, problem.cell_width * problem.cells, problem.cells+1)
    xspace = 0.5 * (xspace[1:] + xspace[:-1])
    ref_flux = mms.solution_one_material_02(xspace, problem.mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * problem.angle_weight, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)


@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_two_material01(angular):
    file = "../input-files/mms-two-material01.inp"
    problem = Transport(file)
    problem.change_param("spatial x cells", 400)
    angular = True if angular == "True" else False
    flux = multi_group.source_iteration(problem.medium_map, problem.xs_total, \
                            problem.xs_scatter, problem.xs_fission, \
                            problem.external_source, problem.point_source, \
                            problem.mu, problem.angle_weight, problem.params, \
                            problem.cell_width, angular=angular)
    xspace = np.linspace(0, problem.cell_width * problem.cells, problem.cells+1)
    xspace = 0.5 * (xspace[1:] + xspace[:-1])
    ref_flux = mms.solution_two_material_01(xspace, problem.mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * problem.angle_weight, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)


@pytest.mark.source_iteration
@pytest.mark.parametrize("angular", ("True", "False"))
def test_mms_two_material02(angular):
    file = "../input-files/mms-two-material02.inp"
    problem = Transport(file)
    problem.change_param("spatial x cells", 400)
    angular = True if angular == "True" else False
    flux = multi_group.source_iteration(problem.medium_map, problem.xs_total, \
                            problem.xs_scatter, problem.xs_fission, \
                            problem.external_source, problem.point_source, \
                            problem.mu, problem.angle_weight, problem.params, \
                            problem.cell_width, angular=angular)
    xspace = np.linspace(0, problem.cell_width * problem.cells, problem.cells+1)
    xspace = 0.5 * (xspace[1:] + xspace[:-1])
    ref_flux = mms.solution_two_material_02(xspace, problem.mu)
    if angular:
        assert np.all(np.fabs(flux[:,:,0] - ref_flux) < 1e-4)
    else:
        ref_flux = np.sum(ref_flux * problem.angle_weight, axis=1)
        assert np.all(np.fabs(flux[:,0] - ref_flux) < 1e-4)
