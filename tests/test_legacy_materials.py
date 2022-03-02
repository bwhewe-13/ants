########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Test Problems for constucting material cross sections for materials 
# listed below. The cross sections are compared to the original one-
# dimenionsal 'discrete1' package.
# 
# Materials: "uranium", "uranium-hydride", "plutonium" 
#            "stainless-steel-440", "high-density-polyethyene-618", 
#            "high-density-polyethyene-087", "carbon", "uranium-235", 
#            "uranium-238", "water-uranium-dioxide", "plutonium-239", 
#            "plutonium-240"
#
########################################################################

from ants.materials import Materials

from discrete1 import generate, fixed 

import pytest
import numpy as np
import pkg_resources

ENR_PATH = pkg_resources.resource_filename('ants','data/energy/')
_ENERGY_BOUNDS = np.load(ENR_PATH + 'energy_bounds.npz')

@pytest.mark.enrichment
def test_uranium_enrichment(enrich):    
    legacy_xs = generate.XSGenerate087('U', \
                                enrich=int(enrich)*0.01).cross_section()
    material_list = ["uranium-%{}%".format(enrich)]
    ants_problem = Materials(material_list, g087r, _ENERGY_BOUNDS['87'])
    ants_xs = ants_problem.data["material-" + material_list[0]]
    assert [np.all(np.allclose(ants_xs[xs], legacy_xs[xs], atol=1e-12)) \
                                                    for xs in range(3)]

@pytest.mark.enrichment
@pytest.mark.energy087r
def test_uranium_energy_coarsen(enrich, g087r):
    legacy_xs = fixed.UraniumStainless.steady(g087r, 2, \
                                                enrich=int(enrich)*0.01)
    legacy_xs = [legacy_xs[4][500], legacy_xs[5][500], legacy_xs[6][500]]
    material_list = ["uranium-%{}%".format(enrich)]
    ants_problem = Materials(material_list, g087r, _ENERGY_BOUNDS['87'])
    ants_xs = ants_problem.data["material-" + material_list[0]]
    assert [np.all(np.allclose(ants_xs[xs], legacy_xs[xs], atol=1e-12)) \
                                                    for xs in range(3)]

@pytest.mark.enrichment
def test_uranium_hydride_enrichment(enrich):    
    legacy_xs = generate.XSGenerate087('UH3', \
                                enrich=int(enrich)*0.01).cross_section()
    material_list = ["uranium-hydride-%{}%".format(enrich)]
    ants_problem = Materials(material_list, 87, _ENERGY_BOUNDS['87'])
    ants_xs = ants_problem.data["material-" + material_list[0]]
    assert [np.all(np.allclose(ants_xs[xs], legacy_xs[xs], atol=1e-12)) \
                                                    for xs in range(3)]

@pytest.mark.enrichment
@pytest.mark.energy087r
def test_uranium_hydride_energy_coarsen(enrich, g087r):
    legacy_xs = []
    temp_xs = generate.XSGenerate087('UH3', \
                                enrich=int(enrich)*0.01).cross_section()
    idx = generate.ReduceTools.index_generator(87, g087r)
    legacy_xs.append(generate.ReduceTools.vector_reduction(temp_xs[0], idx))
    legacy_xs.append(generate.ReduceTools.matrix_reduction(temp_xs[1], idx))
    legacy_xs.append(generate.ReduceTools.matrix_reduction(temp_xs[2], idx))
    material_list = ["uranium-hydride-%{}%".format(enrich)]
    ants_problem = Materials(material_list, g087r, _ENERGY_BOUNDS['87'])
    ants_xs = ants_problem.data["material-" + material_list[0]]
    assert [np.all(np.allclose(ants_xs[xs], legacy_xs[xs], atol=1e-12)) \
                                                    for xs in range(3)]

@pytest.mark.enrichment
def test_plutonium_enrichment(enrich):
    legacy_xs = generate.XSGenerate618.cross_section(1 - int(enrich)*0.01)
    material_list = ["plutonium-%{}%".format(enrich)]
    ants_problem = Materials(material_list, 618, _ENERGY_BOUNDS['618'])
    ants_xs = ants_problem.data["material-" + material_list[0]]
    assert [np.all(np.allclose(ants_xs[xs], legacy_xs[xs], atol=1e-12)) \
                                                    for xs in range(3)]

@pytest.mark.enrichment
@pytest.mark.energy618r
def test_plutonium_energy_coarsen(enrich, g618r):
    legacy_xs = generate.XSGenerate618.cross_section_reduce(g618r, \
                                                   1 - int(enrich)*0.01)
    material_list = ["plutonium-%{}%".format(enrich)]
    ants_problem = Materials(material_list, g618r, _ENERGY_BOUNDS['618'])
    ants_xs = ants_problem.data["material-" + material_list[0]]
    assert [np.all(np.allclose(ants_xs[xs], legacy_xs[xs], atol=1e-12)) \
                                                    for xs in range(3)]    

#     _, _, _, _, orig_total, orig_scatter, orig_fission, _, _, _, _, 
#     orig_total = orig_total[500].copy()
#     orig_scatter = orig_scatter[500].copy()
#     orig_fission = orig_fission[500].copy()    

# from discrete1.generate import generate.XSGenerate087, XSGenerate618
# from ants.materials import Materials
# from discrete1.fixed import SHEM, UraniumStainless

# import numpy as np
# import matplotlib.pyplot as plt

# """
# [x] uranium
# [x] uranium-hydride
# [.] plutonium
# [x] stainless-steel-440
# [.] high-density-polyethyene-618
# [x] high-density-polyethyene-087
# [x] carbon
# [x] uranium-235
# [x] uranium-238
# [ ] water-uranium-dioxide
# [.] plutonium-239
# [.] plutonium-240
# """

# _ENERGY_BOUNDS = np.load('ants/ants/materials_sources/energy_bounds.npz')['87']
# new_problem = Materials([("uranium",0.20), ("high-density-polyethyene-087",)], 43, energy_bounds)
# new_problem.add_source('14.1-mev',0)
# new_source = new_problem.data['source-14.1-mev']
# _, _, _, _, _, _, _, _, _, _, orig_source, = UraniumStainless.steady(43, 8)

# print(np.array_equal(new_source[0], orig_source[1]))


# energy_bounds = np.load('ants/ants/materials_sources/energy_bounds.npz')['361']
# energy_idx = np.load('ants/ants/materials_sources/group_indices_361G.npz')['240']
# new_problem = Materials([("water-uranium-dioxide",0.20)], 240, energy_bounds, energy_idx)
# new_problem.add_source('ambe-point',0)
# new_source = new_problem.data['source-ambe-point']
# _, _, _, _, _, _, _, _, _, _, orig_source, = SHEM.steady(240, 8)

# print(np.array_equal(new_source[0], orig_source[1]))


# new_problem.compile_cross_section()
# print(new_problem.data.keys())

# stainless steel
# for ii in [87, 80, 60, 43, 21, 10]:
# # for ii in [87]:
#     # orig_total, orig_scatter, orig_fission = XSGenerate087('SS440').cross_section()
#     _, _, _, _, orig_total, orig_scatter, orig_fission, _, _, _, _, = UraniumStainless.steady(ii, 8)
#     orig_total = orig_total[500].copy()
#     orig_scatter = orig_scatter[500].copy()
#     orig_fission = orig_fission[500].copy()
#     # None is energy bounds
#     energy_bounds = np.load('ants/ants/materials_sources/energy_bounds.npz')['87']
#     new_problem = Materials([("uranium",0.20)], ii, energy_bounds)
#     new_total, new_scatter, new_fission = new_problem.material_cross_section()

#     print('Groups ',ii)
#     print('Total:', np.array_equal(orig_total, new_total), np.all(np.isclose(orig_total, new_total, atol=1e-12)))
#     print('Scatter:', np.array_equal(orig_scatter, new_scatter), np.all(np.isclose(orig_scatter, new_scatter, atol=1e-12)))
#     print('Fission:', np.array_equal(orig_fission, new_fission), np.all(np.isclose(orig_fission, new_fission, atol=1e-12)))

#     print(np.amax(orig_total - new_total))
#     print(np.amax(orig_scatter - new_scatter))
#     print(np.amax(orig_fission - new_fission))
#     print()


# # SHEM
# _, _, _, _, orig_total, orig_scatter, orig_fission, _, _, _, _, = SHEM.steady(240, 8)
# orig_total = orig_total[0].copy()
# orig_scatter = orig_scatter[0].copy()
# orig_fission = orig_fission[0].copy()

# # None is energy bounds
# energy_bounds = np.load('ants/ants/materials_sources/energy_bounds.npz')['361']
# energy_idx = np.load('ants/ants/materials_sources/group_indices_361G.npz')['240']
# new_problem = Materials([("water-uranium-dioxide",)], 240, energy_bounds, energy_idx)
# new_total, new_scatter, new_fission = new_problem.material_cross_section()

# print('SHEM')
# print('Total:', np.array_equal(orig_total, new_total))
# print('Scatter:', np.array_equal(orig_scatter, new_scatter))
# print('Fission:', np.array_equal(orig_fission, new_fission))
# print(np.amax(orig_total - new_total))
# print(np.amax(orig_scatter - new_scatter))
# print(np.amax(orig_fission - new_fission))
# print()

# for enrichment in [0.0, 0.25, 0.5, 0.75, 1.0]:
# # for enrichment in [1.0]:
#     # uranium
#     orig_total, orig_scatter, orig_fission = XSGenerate087('UH3',enrich=enrichment).cross_section()
#     # None is energy bounds
#     new_problem = Materials([("uranium-hydride",enrichment)], 87, None)
#     new_total, new_scatter, new_fission = new_problem.material_cross_section()


# for enrichment in [0.0, 0.25, 0.5, 0.75, 1.0]:
# # for enrichment in [0.25]:
#     # uranium
#     orig_total, orig_scatter, orig_fission = XSGenerate618.cross_section(enrichment)
#     # None is energy bounds
#     # new_problem = Materials([("high-density-polyethyene-618",1-enrichment)], 618, None)
#     new_problem = Materials([("plutonium-240",1-enrichment)], 618, None)
#     new_total, new_scatter, new_fission = new_problem.material_cross_section()

#     print('Uranium Hydride - {}'.format(enrichment))
#     print('Total:', np.array_equal(orig_total[2], new_total))
#     print('Scatter:', np.array_equal(orig_scatter[2], new_scatter))
#     print('Fission:', np.array_equal(orig_fission[2], new_fission))
#     print(np.amax(orig_total[2] - new_total))
#     print(np.amax(orig_scatter[2] - new_scatter))
#     print(np.amax(orig_fission[2] - new_fission))
#     print()
