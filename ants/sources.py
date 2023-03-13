########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Different Attributes of the Problem
# 
########################################################################

from ants.constants import *
from ants.utils import dimensions

import numpy as np
import pkg_resources
import warnings

DATA_PATH = pkg_resources.resource_filename("ants","sources/")

########################################################################
# Material Cross Sections
########################################################################

__enrichment_materials = ("uranium", "uranium-hydride", "plutonium")

__nonenrichment_materials = ("stainless-steel-440", \
        "high-density-polyethyene-618", "high-density-polyethyene-087", \
        "carbon", "uranium-235", "uranium-238", "water-uranium-dioxide", \
        "plutonium-239", "plutonium-240", "vacuum")

__materials = __enrichment_materials + __nonenrichment_materials

def materials(groups, materials, key=False):
    """ Creating cross sections for different materials
    Args:
        groups (int): Number of energy groups
        materials (list): [material1, ..., materialN]
    """
    material_key = {}
    xs_total = []
    xs_scatter = []
    xs_fission = []
    for idx, material in enumerate(materials):
        # Verify it is possible
        assert (material.split("-%")[0] in __materials),\
            "Material not recognized, use:\n{}".format(__materials)
        # Calculate cross section
        total, scatter, fission = _generate_cross_section(groups, material)
        xs_total.append(total)
        xs_scatter.append(scatter)
        xs_fission.append(fission)
        material_key[idx] = material
    xs_total = np.array(xs_total)
    xs_scatter = np.array(xs_scatter)
    xs_fission = np.array(xs_fission)
    if key:
        return xs_total, xs_scatter, xs_fission, material_key
    return xs_total, xs_scatter, xs_fission

def _generate_cross_section(groups, material):
    data = {}
    if "%" in material:
        material, enrichment = material.split("-%")
        enrichment = float(enrichment.strip("%")) * 0.01

    if material == "vacuum":
        return np.zeros((groups)), np.zeros((groups, groups)), np.zeros((groups, groups))
    elif material in __nonenrichment_materials:
        data = np.load(DATA_PATH + "materials/" + material + ".npz")
    elif material == "uranium":
        u235 = np.load(DATA_PATH + "materials/uranium-235.npz")
        u238 = np.load(DATA_PATH + "materials/uranium-238.npz")
        for xs in u235.files:
            data[xs] = u235[xs] * enrichment + u238[xs] * (1 - enrichment)
    elif material == "plutonium":
        pu239 = np.load(DATA_PATH + "materials/plutonium-239.npz")
        pu240 = np.load(DATA_PATH + "materials/plutonium-240.npz")
        for xs in pu239.files:
            data[xs] = pu239[xs] * enrichment + pu240[xs] * (1 - enrichment)
    elif material == "uranium-hydride":
        return _generate_uranium_hydride(enrichment)

    return data["total"], data["scatter"], data["fission"]

def _generate_uranium_hydride(enrichment):
    molar = enrichment * URANIUM_235_MM + (1 - enrichment) * URANIUM_238_MM
    rho = URANIUM_HYDRIDE_RHO / URANIUM_RHO
    
    n235 = (enrichment * rho * molar) / (molar + 3 * HYDROGEN_MM) 
    n238 = ((1 - enrichment) * rho * molar) / (molar + 3 * HYDROGEN_MM) 
    n1 = URANIUM_HYDRIDE_RHO * AVAGADRO / (molar + 3 * HYDROGEN_MM) * CM_TO_BARNS * 3
    
    u235 = np.load(DATA_PATH + "materials/uranium-235.npz")
    u238 = np.load(DATA_PATH + "materials/uranium-238.npz")
    h1 = np.load(DATA_PATH + "materials/hydrogen.npz")

    total = n235 * u235["total"] + n238 * u238["total"] + n1 * h1["total"]
    scatter = n235 * u235["scatter"] + n238 * u238["scatter"] + n1 * h1["scatter"]
    fission = n235 * u235["fission"] + n238 * u238["fission"] + n1 * h1["fission"]
    return total, scatter, fission

########################################################################
# External Sources
########################################################################

__externals = ("reeds", "mms-03", "mms-04", "mms-05", "ambe")

def externals(name, shape, **kw):
    external = np.zeros(shape)
    if isinstance(name, float):
        external[(...)] = name
        return external
    if name == "reeds":
        assert "edges_x" in kw, "Need edges_x for external source"
        return _external_reeds(external, kw["edges_x"])
    elif name in ["mms-03", "mms-04", "mms-05"]:
        assert "centers_x" in kw, "Need centers_x for external source"
        assert "angle_x" in kw, "Need angle_x for external source"
        if name == "mms-03":
            return _external_mms_03(external, kw["centers_x"], kw["angle_x"])
        elif name == "mms-04":
            return _external_mms_04(external, kw["centers_x"], kw["angle_x"])
        elif name == "mms-05":
            return _external_mms_05(external, kw["centers_x"], kw["angle_x"])
    elif name == "ambe":
        assert "edges_x" in kw, "Need edges_x for external source"
        assert "groups" in kw, "Need groups for external source"
        return _external_ambe(external, kw["groups"], kw["edges_x"])
    warnings.warn("External Source not populated, use {}".format(__externals))
    return external

def _external_reeds(external, edges_x):
    source_values = [0.0, 1.0, 0.0, 50.0, 0.0, 1.0, 0.0]
    lhs = [0., 2., 4., 6., 10., 12., 14.]
    rhs = [2., 4., 6., 10., 12., 14., 16.]
    loc = lambda x: int(np.argwhere(edges_x == x))
    bounds = [slice(loc(ii), loc(jj)) for ii, jj in zip(lhs, rhs)]
    for ii in range(len(bounds)):
        external[bounds[ii]] = source_values[ii]
    return external

def _external_mms_03(external, centers_x, angle_x):
    cc1 = 0.5
    cc2 = 0.25
    def dependence(mu):
        return cc2 * mu * np.exp(mu) * 2 * centers_x + cc1 + cc2 \
            * centers_x**2 * np.exp(mu) - 0.9 / 2 * (2 * cc1 + cc2 \
            * centers_x**2 * (np.exp(1) - np.exp(-1)))
    for n, mu in enumerate(angle_x):
        external[:,n] = dependence(mu)[:,None]
    return external

def _external_mms_04(external, centers_x, angle_x):
    width = 2.
    def quasi(x, mu):
        c = 0.3
        return 2 * width * mu - 4 * x * mu - 2 * x**2 \
               + 2 * width * x - c * (-2 * x**2 + 2 * width * x)
    def scatter(x, mu):
        c = 0.9
        const = -0.125 * width + 0.5 * width**2
        return 0.25 * (mu + x) + const - c * ((0.25 * x + const))
    for n, mu in enumerate(angle_x):
        idx = (centers_x < (0.5 * width))
        external[idx,n] = quasi(centers_x[idx], mu)
        idx = (centers_x > (0.5 * width))
        external[idx,n] = scatter(centers_x[idx], mu)
    return external

def _external_mms_05(external, centers_x, angle_x):
    width = 2.
    def quasi(x, mu):
        c = 0.3
        return mu * (2 * width**2 - 4 * np.exp(mu) * x) - 2 * np.exp(mu) \
                * x**2 + 2 * width**2 * x - c / 2 * (-2 * x**2 \
                * (np.exp(1) - np.exp(-1)) + 4 * width**2 * x)
    def scatter(x, mu):
        c = 0.9
        const = width**3 - width**2 * np.exp(mu)
        return width * mu * np.exp(mu) + width * x * np.exp(mu) + const \
                - c/2 * (2 * width**3 + (np.exp(1) - np.exp(-1)) \
                * (x * width - width**2))
    for n, mu in enumerate(angle_x):
        idx = (centers_x < (0.5 * width))
        external[idx,n] = quasi(centers_x[idx], mu)
        idx = (centers_x > (0.5 * width))
        external[idx,n] = scatter(centers_x[idx], mu)
    return external

def _external_ambe(external, groups, edges_x):
    AmBe = np.load(DATA_PATH + "external/AmBe_source_050G.npz")
    edges_gg = "energy/G{}_energy_grid.npy".format(str(groups).zfill(3))
    edges_gg = np.load(DATA_PATH + edges_gg)
    # Convert to MeV
    if np.max(edges_gg) > 20.0:
        edges_gg *= 1E-6
    centers_gg = 0.5 * (edges_gg[1:] + edges_gg[:-1])
    locs = lambda x1, x2: np.argwhere((centers_gg > x1) & (centers_gg <= x2)).flatten()
    center = 0.5 * max(edges_x)
    center_idx = np.where(abs(edges_x - center) == abs(edges_x - center).min())[0]
    for ii in range(len(AmBe["magnitude"])):
        idx = locs(AmBe["edges"][ii], AmBe["edges"][ii+1])
        for jj in center_idx:
            external[(jj, ..., idx)] = AmBe["magnitude"][ii]
    return external

########################################################################
# Boundary Conditions
########################################################################

__boundaries = ("14.1-mev", "mms-03", "mms-04", "mms-05")

def boundaries(name, shape, location, **kw):
    # location is list, 0: x = 0, 1: x = X
    boundary = np.zeros(shape)
    if isinstance(name, float):
        boundary[location] = name
        return boundary
    if name == "14.1-mev":
        assert "energy_grid" in kw, "Need energy_grid for boundary condition"
        group = np.argmin(abs(kw["energy_grid"] - 14.1E6))
        boundary[(location, ..., group)] = 1.0
        return boundary
    elif name == "mms-03":
        assert "angle_x" in kw, "Need angle_x for boundary condition"
        return _boundary_mms_03(boundary, kw["angle_x"])
    elif name in ["mms-04", "mms-05"]:
        return _boundary_mms_04_05(boundary, name)
    warnings.warn("Boundary condition not populated, use {}".format(__boundaries))
    return boundary

def _boundary_mms_03(boundary, angle_x):
    const1 = 0.5
    const2 = 0.25
    boundary[0] = const1
    boundary[1] = const1 + const2 * np.exp(angle_x)
    return boundary

def _boundary_mms_04_05(boundary, name):
    width = 2.
    if name == "mms-04":
        boundary[1] = 0.5 * width**2 + 0.125 * width
    elif name == "mms-05":
        boundary[1] = width**3
    return boundary