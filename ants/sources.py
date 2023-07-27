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
import warnings

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

########################################################################
# External Sources - 1D
########################################################################

__externals1d = ("reeds", "mms-03", "mms-04", "mms-05", "ambe")

def externals1d(name, shape, **kw):
    external = np.zeros(shape)
    if isinstance(name, float):
        external[(...)] = name
        return external
    if name == "reeds":
        assert "edges_x" in kw, "Need edges_x for external source"
        assert "bc" in kw, "Need bc for boundary conditions"
        return _external_reeds(external, kw["edges_x"], kw["bc"])
    elif name in ["mms-03", "mms-04", "mms-05"]:
        assert "centers_x" in kw, "Need centers_x for external source"
        assert "angle_x" in kw, "Need angle_x for external source"
        if name == "mms-03":
            return _external_1d_mms_03(external, kw["centers_x"], kw["angle_x"])
        elif name == "mms-04":
            return _external_1d_mms_04(external, kw["centers_x"], kw["angle_x"])
        elif name == "mms-05":
            return _external_1d_mms_05(external, kw["centers_x"], kw["angle_x"])
    elif name == "ambe":
        assert "edges_x" in kw, "Need edges_x for external source"
        assert "groups" in kw, "Need groups for external source"
        return _external_1d_ambe(external, kw["groups"], kw["edges_x"])
    warnings.warn("External Source not populated, use {}".format(__externals1d))
    return external

def _external_reeds(external, edges_x, bc):
    source_values = np.array([0.0, 1.0, 0.0, 50.0, 50.0, 0.0, 1.0, 0.0])
    lhs = np.array([0., 2., 4., 6., 8., 10., 12., 14.])
    rhs = np.array([2., 4., 6., 8., 10., 12., 14., 16.])
    if np.sum(bc) > 0.0:
        idx = slice(0, 4) if bc == [0, 1] else slice(4, 8)
        corrector = 0.0 if bc == [0, 1] else 8.0
        source_values = source_values[idx].copy()
        lhs = lhs[idx].copy() - corrector
        rhs = rhs[idx].copy() - corrector
    loc = lambda x: int(np.argwhere(edges_x == x))
    bounds = [slice(loc(ii), loc(jj)) for ii, jj in zip(lhs, rhs)]
    for ii in range(len(bounds)):
        external[bounds[ii]] = source_values[ii]
    return external

def _external_1d_mms_03(external, centers_x, angle_x):
    cc1 = 0.5
    cc2 = 0.25
    def dependence(mu):
        return cc2 * mu * np.exp(mu) * 2 * centers_x + cc1 + cc2 \
            * centers_x**2 * np.exp(mu) - 0.9 / 2 * (2 * cc1 + cc2 \
            * centers_x**2 * (np.exp(1) - np.exp(-1)))
    for n, mu in enumerate(angle_x):
        external[:,n] = dependence(mu)[:,None]
    return external

def _external_1d_mms_04(external, centers_x, angle_x):
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

def _external_1d_mms_05(external, centers_x, angle_x):
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

def _external_1d_ambe(external, groups, edges_x):
    AmBe = np.load(DATA_PATH + "external/AmBe_source_050G.npz")
    edges_g = "energy/G{}_energy_grid.npy".format(str(groups).zfill(3))
    edges_g = np.load(DATA_PATH + edges_g)
    # Convert to MeV
    if np.max(edges_g) > 20.0:
        edges_g *= 1E-6
    centers_g = 0.5 * (edges_g[1:] + edges_g[:-1])
    locs = lambda x1, x2: np.argwhere((centers_g > x1) & (centers_g <= x2)).flatten()
    center = 0.5 * max(edges_x)
    center_idx = np.where(abs(edges_x - center) == abs(edges_x - center).min())[0]
    for ii in range(len(AmBe["magnitude"])):
        idx = locs(AmBe["edges"][ii], AmBe["edges"][ii+1])
        for jj in center_idx:
            external[(jj, ..., idx)] = AmBe["magnitude"][ii]
    return external

########################################################################
# External Sources - 2D
########################################################################

__externals2d = ("mms-03", "mms-04", "ambe")

def externals2d(name, shape, **kw):
    external = np.zeros(shape)
    if isinstance(name, float):
        external[(...)] = name
        return external

    if name == "mms-03":
        variables = ["angle_x", "angle_y"]
        assert (np.sort(list(kw.keys())) == np.sort(variables)).all(), \
            "Need {} for external source".format(variables)
        return _external_2d_mms_03(external, kw["angle_x"], kw["angle_y"])

    elif name == "mms-04":
        variables = ["angle_x", "angle_y", "centers_x", "centers_y"]
        assert (np.sort(list(kw.keys())) == np.sort(variables)).all(), \
            "Need {} for MMS - 04 source".format(variables)
        return _external_2d_mms_04(external, kw["angle_x"], kw["angle_y"], \
                                    kw["centers_x"], kw["centers_y"])

    elif name == "ambe":
        variables = ["edges_g", "coordinates", "edges_x", "edges_y"]
        assert (np.sort(list(kw.keys())) == np.sort(variables)).all(), \
            "Need {} for AmBe source".format(variables)
        return _external_2d_ambe(external, kw["edges_g"], kw["coordinates"], \
                                 kw["edges_x"], kw["edges_y"])
    warnings.warn("External Source not populated, use float")
    # warnings.warn("External Source not populated, use {}".format(__externals2d))
    return external

def _external_2d_mms_03(external, angle_x, angle_y):
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        external[:,:,nn,0] = 0.5 * (np.exp(-1) - np.exp(1)) + np.exp(mu) + np.exp(eta)
    return external

def _external_2d_mms_04(external, angle_x, angle_y, centers_x, centers_y):
    x, y = np.meshgrid(centers_x, centers_y, indexing="ij")
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        external[:,:,nn,0] = 1 + 0.1 * np.exp(mu) * x**2 \
                     + 0.1 * np.exp(eta) * y**2 + 0.025 * np.exp(-1) \
                     * (-20 * np.exp(1) + x**2 + y**2 - np.exp(2) \
                     * (x**2 + y**2)) + 0.2 * (mu * x * np.exp(mu) \
                     + eta * y * np.exp(eta))
    external = np.transpose(external, (1, 0, 2, 3))
    return external

def _external_2d_ambe(external, edges_g, coordinates, edges_x, edges_y):
    AmBe = np.load(DATA_PATH + "external/AmBe_source_050G.npz")
    # Convert to MeV
    if np.max(edges_g) > 20.0:
        edges_g *= 1E-6
    # Get energy spectra of AmBe source
    value = resize_array_1d(edges_g, AmBe["edges"], AmBe["magnitude"])
    # Put in location
    external = ants.spatial2d(external, value, coordinates, edges_x, edges_y)
    return external



########################################################################
# Boundary Conditions - 1D
########################################################################

__boundaries1d = ("14.1-mev", "mms-03", "mms-04", "mms-05")

def boundaries1d(name, shape, location, **kw):
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
        return _boundary_1d_mms_03(boundary, kw["angle_x"])
    elif name in ["mms-04", "mms-05"]:
        return _boundary_1d_mms_04_05(boundary, name)
    warnings.warn("Boundary condition not populated, use {}".format(__boundaries1d))
    return boundary

def _boundary_1d_mms_03(boundary, angle_x):
    const1 = 0.5
    const2 = 0.25
    boundary[0] = const1
    boundary[1] = const1 + const2 * np.exp(angle_x)
    return boundary

def _boundary_1d_mms_04_05(boundary, name):
    width = 2.
    if name == "mms-04":
        boundary[1] = 0.5 * width**2 + 0.125 * width
    elif name == "mms-05":
        boundary[1] = width**3
    return boundary

########################################################################
# Boundary Conditions - 2D
########################################################################

__boundaries2d = ("mms-01", "mms-02", "mms-03", "mms-04")

def boundaries2d(name, shape_x, shape_y, **kw):
    # location is list, 0: x = 0, 1: x = X
    boundary_x = np.zeros(shape_x)
    boundary_y = np.zeros(shape_y)
    if isinstance(name, float):
        boundary_x = name
        boundary_y = name
        return boundary_x, boundary_y
    assert "angle_x" in kw, "Need angle_x for boundary condition"
    assert "angle_y" in kw, "Need angle_y for boundary condition"
    if name == "mms-01":
        assert "centers_x" in kw, "Need centers_x for boundary condition"
        return _boundary_2d_mms_01(boundary_x, boundary_y, kw["angle_x"], \
                                   kw["angle_y"], kw["centers_x"])
    elif name == "mms-02":
        assert "centers_x" in kw, "Need centers_x for boundary condition"
        assert "centers_y" in kw, "Need centers_y for boundary condition"
        return _boundary_2d_mms_02(boundary_x, boundary_y, kw["angle_x"], \
                        kw["angle_y"], kw["centers_x"], kw["centers_y"])
    elif name == "mms-03":
        return _boundary_2d_mms_03(boundary_x, boundary_y, kw["angle_x"], kw["angle_y"])
    elif name == "mms-04":
        assert "centers_x" in kw, "Need centers_x for boundary condition"
        assert "centers_y" in kw, "Need centers_y for boundary condition"
        return _boundary_2d_mms_04(boundary_x, boundary_y, kw["angle_x"], \
                        kw["angle_y"], kw["centers_x"], kw["centers_y"])
    warnings.warn("Boundary condition not populated, use {}".format(__boundaries2d))
    return boundary_x, boundary_y

def _boundary_2d_mms_01(boundary_x, boundary_y, angle_x, angle_y, centers_x):
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        if mu > 0.0 and eta > 0.0:
            boundary_x[0,:,nn,0] = 1.5
            boundary_y[0,:,nn,0] = 0.5 + np.exp(-centers_x / mu)
        elif mu > 0.0 and eta < 0.0:
            boundary_x[0,:,nn,0] = 1.5
            boundary_y[1,:,nn,0] = 0.5 + np.exp(-centers_x / mu)
        elif mu < 0.0 and eta > 0.0:
            boundary_x[1,:,nn,0] = 1.5
            boundary_y[0,:,nn,0] = 0.5 + np.exp((1 - centers_x) / mu)
        elif mu < 0.0 and eta < 0.0:
            boundary_x[1,:,nn,0] = 1.5
            boundary_y[1,:,nn,0] = 0.5 + np.exp((1 - centers_x) / mu)
    return boundary_x, boundary_y

def _boundary_2d_mms_02(boundary_x, boundary_y, angle_x, angle_y, centers_x, \
        centers_y):
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        if mu > 0.0 and eta > 0.0:
            boundary_x[0,:,nn,0] = 1.5 + 0.5 * np.exp(-centers_y / eta)
            boundary_y[0,:,nn,0] = 1.5 + 0.5 * np.exp(-centers_x / mu)
        elif mu > 0.0 and eta < 0.0:
            boundary_x[0,:,nn,0] = 1.5 + 0.5 * np.exp((1 - centers_y) / eta)
            boundary_y[1,:,nn,0] = 1.5 + 0.5 * np.exp(-centers_x / mu)
        elif mu < 0.0 and eta > 0.0:
            boundary_x[1,:,nn,0] = 1.5 + 0.5 * np.exp(-centers_y / eta)
            boundary_y[0,:,nn,0] = 1.5 + 0.5 * np.exp((1 - centers_x) / mu)
        elif mu < 0.0 and eta < 0.0:
            boundary_x[1,:,nn,0] = 1.5 + 0.5 * np.exp((1 - centers_y) / eta)
            boundary_y[1,:,nn,0] = 1.5 + 0.5 * np.exp((1 - centers_x) / mu)
    return boundary_x, boundary_y

def _boundary_2d_mms_03(boundary_x, boundary_y, angle_x, angle_y):
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        if mu > 0.0 and eta > 0.0:
            boundary_x[0,:,nn,0] = np.exp(mu) + np.exp(eta)
            boundary_y[0,:,nn,0] = np.exp(mu) + np.exp(eta)
        elif mu > 0.0 and eta < 0.0:
            boundary_x[0,:,nn,0] = np.exp(mu) + np.exp(eta)
            boundary_y[1,:,nn,0] = np.exp(mu) + np.exp(eta)
        elif mu < 0.0 and eta > 0.0:
            boundary_x[1,:,nn,0] = np.exp(mu) + np.exp(eta)
            boundary_y[0,:,nn,0] = np.exp(mu) + np.exp(eta)
        elif mu < 0.0 and eta < 0.0:
            boundary_x[1,:,nn,0] = np.exp(mu) + np.exp(eta)
            boundary_y[1,:,nn,0] = np.exp(mu) + np.exp(eta)
    return boundary_x, boundary_y

def _boundary_2d_mms_04(boundary_x, boundary_y, angle_x, angle_y, centers_x, \
        centers_y):
    for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        if mu > 0.0 and eta > 0.0:
            boundary_x[0,:,nn,0] = 1 + 0.1 * np.exp(eta) * centers_y**2
            boundary_y[0,:,nn,0] = 1 + 0.1 * np.exp(mu) * centers_x**2
        elif mu > 0.0 and eta < 0.0:
            boundary_x[0,:,nn,0] = 1 + 0.1 * np.exp(eta) * centers_y**2
            boundary_y[1,:,nn,0] = 1 + 0.4 * np.exp(eta) + 0.1 * np.exp(mu) * centers_x**2
        elif mu < 0.0 and eta > 0.0:
            boundary_x[1,:,nn,0] = 1 + 0.4 * np.exp(mu) + 0.1 * np.exp(eta) * centers_y**2
            boundary_y[0,:,nn,0] = 1 + 0.1 * np.exp(mu) * centers_x**2
        elif mu < 0.0 and eta < 0.0:
            boundary_x[1,:,nn,0] = 1 + 0.4 * np.exp(mu) + 0.1 * np.exp(eta) * centers_y**2
            boundary_y[1,:,nn,0] = 1 + 0.4 * np.exp(eta) + 0.1 * np.exp(mu) * centers_x**2
    return boundary_x, boundary_y

########################################################################
# Meshing Different Energy Grids
########################################################################

def _concatenate_edges_1d(fine, coarse, value):
    # Combine the edges for both the coarse and fine grid
    new_edges = np.sort(np.unique(np.concatenate((coarse, fine))))
    # Create new array for values
    new_value = np.zeros((new_edges.shape[0] - 1))
    # Iterate over fine edges
    for cc, (gg1, gg2) in enumerate(zip(fine[:-1], fine[1:])):
        idx1 = np.argmin(np.fabs(gg1 - new_edges))
        idx2 = np.argmin(np.fabs(gg2 - new_edges))
        for gg in range(idx1, idx2):
            new_value[gg] = value[cc]
    return new_edges, new_value

def resize_array_1d(fine, coarse, value):
    """ Coarsen array for difference energy grids where (G hat) < (G)
    Arguments:
        fine (float [G + 1]): fine energy edges
        coarse (float [G hat + 1]): coarse energy edges
        value (float [G] or [G hat]): values of grid values
    Returns:
        resized (float [G hat] or [G]): values of resized grid
    """
    # Combine edges
    if (value.shape[0] + 1 == coarse.shape[0]):
        fine, coarse = coarse.copy(), fine.copy()
    fine, value = _concatenate_edges_1d(fine, coarse, value)
    # Create coarse array
    shrink = np.zeros((coarse.shape[0] - 1))
    # Iterate over all coarse bins
    for cc, (gg1, gg2) in enumerate(zip(coarse[:-1], coarse[1:])):
        # Find indices for edge locations
        idx1 = np.argmin(np.fabs(gg1 - fine))
        idx2 = np.argmin(np.fabs(gg2 - fine))
        # Estimate magnitude
        magnitude = np.sum(value[idx1:idx2] * np.diff(fine[idx1:idx2+1]))
        magnitude /= (gg2 - gg1)
        # Populate coarsened array
        shrink[cc] = magnitude
    return shrink


def _concatenate_edges_2d(fine, coarse, value):
    # Combine the edges for both the coarse and fine grid
    new_edges = np.sort(np.unique(np.concatenate((coarse, fine))))
    # Create new array for values
    new_value = np.zeros((new_edges.shape[0] - 1, new_edges.shape[0] - 1))
    # Iterate over fine edges
    for cc1, (gg1, gg2) in enumerate(zip(fine[:-1], fine[1:])):
        idx1 = np.argmin(np.fabs(gg1 - new_edges))
        idx2 = np.argmin(np.fabs(gg2 - new_edges))
        for cc2, (gg3, gg4) in enumerate(zip(fine[:-1], fine[1:])):
            idx3 = np.argmin(np.fabs(gg3 - new_edges))
            idx4 = np.argmin(np.fabs(gg4 - new_edges))
            for gg1 in range(idx1, idx2):
                for gg2 in range(idx3, idx4):
                    new_value[gg1,gg2] = value[cc1,cc2]
    return new_edges, new_value


def resize_array_2d(fine, coarse, value):
    """ Coarsen array for difference energy grids where (G hat) < (G)
    Arguments:
        fine (float [G + 1]): fine energy edges
        coarse (float [G hat + 1]): coarse energy edges
        value (float [G x G] or [G hat x G hat]): values of grid values
    Returns:
        resized (float [G hat x G hat] or [G x G]): values of resized grid
    """
    # Combine edges
    if (value.shape[0] + 1 == coarse.shape[0]):
        fine, coarse = coarse.copy(), fine.copy()
    fine, value = _concatenate_edges_2d(fine, coarse, value)
    # return value
    # Create coarse array
    shrink = np.zeros((coarse.shape[0] - 1, coarse.shape[0] - 1))
    # Iterate over all coarse bins
    for cc1, (gg1, gg2) in enumerate(zip(coarse[:-1], coarse[1:])):
        # Find indices for edge locations
        idx1 = np.argmin(np.fabs(gg1 - fine))
        idx2 = np.argmin(np.fabs(gg2 - fine))
        for cc2, (gg3, gg4) in enumerate(zip(coarse[:-1], coarse[1:])):
            # Find indices for edge locations
            idx3 = np.argmin(np.fabs(gg3 - fine))
            idx4 = np.argmin(np.fabs(gg4 - fine))
            # Estimate magnitude
            magnitude = np.sum(value[idx1:idx2,idx3:idx4] \
                                * (np.diff(fine[idx1:idx2+1])[:,None] \
                                @ np.diff(fine[idx3:idx4+1])[None,:]))
            magnitude /= ((gg2 - gg1) * (gg4 - gg3))
            # Populate coarsened array
            shrink[cc1, cc2] = magnitude
    return shrink