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

import numpy as np
import pkg_resources

import ants
from ants.constants import *

DATA_PATH = pkg_resources.resource_filename("ants","sources/")

########################################################################
# Material Cross Sections
########################################################################

__enrichment_materials = ("uranium", "uranium-hydride", "plutonium")

__nonenrichment_materials = ("stainless-steel-440", "hydrogen", \
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
