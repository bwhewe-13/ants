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

MAT_PATH = pkg_resources.resource_filename("ants","data/materials/")
SOR_PATH = pkg_resources.resource_filename("ants","data/sources/")
ENR_PATH = pkg_resources.resource_filename("ants","data/energy/")

class Materials:
    __enrichment_materials = ("uranium", "uranium-hydride", "plutonium")

    __nonenrichment_materials = ("stainless-steel-440",
                             "high-density-polyethyene-618", \
                             "high-density-polyethyene-087", \
                             "carbon", "uranium-235", "uranium-238", \
                             "water-uranium-dioxide", "plutonium-239", \
                             "plutonium-240")

    __nonphysical_materials = ("reed-scatter", "reed-absorber", \
                               "reed-vacuum", "reed-strong-source", \
                               "mms-absorber", "mms-scatter", "mms-quasi", \
                               "diffusion-01", "diffusion-02")

    __materials = __enrichment_materials + __nonenrichment_materials \
                    + __nonphysical_materials

    def __init__(self, materials, groups, energy_grid, grid_index):
        """ Creating cross sections for different materials

        Args:
            materials (list): [material1, ..., materialN]
            groups (int):
            energy_grid (list):
            grid_index (list):

        """
        assert (isinstance(materials, list)), "Materials must be list"
        for mater in materials:
            assert (mater.split("-%")[0] in self.__class__.__materials),\
                    "Material not recognized, use:\n{}".format(\
                        self.__class__.__materials)
        self.materials = materials
        self.groups = groups
        self.energy_grid = energy_grid
        self.grid_index = grid_index
        self.material_key = {}
        self.compile_cross_sections()
        # self.compile_velocity()

    def __str__(self):
        msg = "Energy Groups: {}\nMaterial: ".format(self.groups)
        msg += "\nMaterial: ".join(self.materials)
        return msg

    def __len__(self):
        return len(self.materials)

    def __iter__(self):
        return iter((self.velocity, self.material_key.keys(), \
                     self.material_key.values()))

    def compile_cross_sections(self):
        for material in self.materials:
            # material_name = "material-" + material
            xs = Materials._generate_cross_section(self, material)
            if len(xs[0]) != self.groups:
                xs = Materials._generate_reduced_cross_section(self, xs)
            self.material_key[material] = xs

    def _generate_cross_section(self, material):
        data = {}
        if "%" in material:
            material, enrichment = Materials._generate_enrich(material)
        if material in self.__class__.__nonenrichment_materials:
            data = np.load(MAT_PATH + material + ".npz")
        elif material in ["uranium", "plutonium"]:
            iso_one = "235" if material == "uranium" else "239"
            iso_two = "238" if material == "uranium" else "240"
            data1 = np.load(MAT_PATH + f"{material}-{iso_one}.npz")
            data2 = np.load(MAT_PATH + f"{material}-{iso_two}.npz")
            for xs_type in data1.files:
                data[xs_type] = data1[xs_type] * enrichment \
                                    + data2[xs_type] * (1 - enrichment)
        elif material in ["uranium-hydride"]:
            return Materials._generate_uranium_hydride(enrichment)
        elif material in self.__class__.__nonphysical_materials:
            func = getattr(NonPhysical, material.replace("-", "_"))
            return func(self.groups)
        return [data["total"], data["scatter"], data["fission"]]

    def _generate_enrich(material):
        material = material.split("-%")
        material[1] = float(material[1].strip("%")) * 0.01
        return material

    def _generate_reduced_cross_section(self, cross_sections):
        total, scatter, fission = cross_sections
        if self.grid_index is None:
            self.grid_index = dimensions.index_generator(len(self.energy_grid)-1, self.groups)
        bin_width = np.diff(self.energy_grid)
        coarse_bin_width = np.array([self.energy_grid[self.grid_index[ii+1]] \
                            -self.energy_grid[self.grid_index[ii]] \
                            for ii in range(self.groups)])
        total = (dimensions.vector_reduction(total * bin_width, \
                 self.grid_index)) / coarse_bin_width
        scatter = (dimensions.matrix_reduction(scatter * bin_width, \
                 self.grid_index)) / coarse_bin_width
        fission = (dimensions.matrix_reduction(fission * bin_width, \
                 self.grid_index)) / coarse_bin_width
        return [total, scatter, fission]

    def _generate_uranium_hydride(enrichment):
        molar = enrichment * URANIUM_235_MM + (1 - enrichment) * URANIUM_238_MM
        rho = URANIUM_HYDRIDE_RHO / URANIUM_RHO
        n235 = (enrichment * rho * molar) / (molar + 3 * HYDROGEN_MM) 
        n238 = ((1 - enrichment) * rho * molar) / (molar + 3 * HYDROGEN_MM) 
        n1 = URANIUM_HYDRIDE_RHO * AVAGADRO / (molar + 3 * HYDROGEN_MM) * CM_TO_BARNS * 3
        u235 = np.load(MAT_PATH + "uranium-235.npz")
        u238 = np.load(MAT_PATH + "uranium-238.npz")
        h1 = np.load(MAT_PATH + "hydrogen.npz")
        total = n235 * u235["total"] + n238 * u238["total"] + n1 * h1["total"]
        scatter = n235 * u235["scatter"] + n238 * u238["scatter"] + n1 * h1["scatter"]
        fission = n235 * u235["fission"] + n238 * u238["fission"] + n1 * h1["fission"]
        return total, scatter, fission

    # def compile_velocity(self):
    #     if self.materials[0] in self.__class__.__nonphysical_materials:
    #         self.velocity = np.ones((self.groups))
    #     else:
    #         energy_centers = 0.5 * (self.energy_grid[1:] + self.energy_grid[:-1])
    #         gamma = (EV_TO_JOULES * energy_centers) \
    #                 / (MASS_NEUTRON * LIGHT_SPEED**2) + 1
    #         self.velocity = LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100
            # self.velocity = np.array([np.mean(velocity[self.grid_index[gg]: \
            #             self.grid_index[gg+1]]) for gg in range(self.groups)])

class NonPhysical:
    def reed_scatter(groups):
        if groups == 1:
            # return [np.array([10.]), np.array([[9.9]]), np.array([[0.]])]
            return [np.array([1.]), np.array([[0.9]]), np.array([[0.]])]

    def reed_absorber(groups):
        if groups == 1:
            return [np.array([5.]), np.array([[0.]]), np.array([[0.]])]

    def reed_vacuum(groups):
        if groups == 1:
            return [np.array([0.]), np.array([[0.]]), np.array([[0.]])]

    def reed_strong_source(groups):
        if groups == 1:
            return [np.array([50.]), np.array([[0.]]), np.array([[0.]])]

    def mms_absorber(groups):
        if groups == 1:
            return [np.array([1.]), np.array([[0.]]), np.array([[0.]])]

    def mms_scatter(groups):
        if groups == 1:
            return [np.array([1.]), np.array([[0.9]]), np.array([[0.]])]
            # return [np.array([1.]), np.array([[0.0]]), np.array([[0.]])]

    def mms_quasi(groups):
        if groups == 1:
            return [np.array([1.]), np.array([[0.3]]), np.array([[0.]])]
            # return [np.array([1.]), np.array([[0.0]]), np.array([[0.]])]

    def diffusion_01(groups):
        if groups == 1:
            return [np.array([100.]), np.array([[100.]]), np.array([[0.]])]

    def diffusion_02(groups):
        if groups == 1:
            return [np.array([2.]), np.array([[0.]]), np.array([[0.]])]

class Source:
    __sources = ("none", "unity", "half-unity", "small", \
                "reed", "mms-03", "mms-04", "mms-05", "ambe")

    def __init__(self, name, cells, cell_edges, angle_x, groups, \
                        energy_grid, qdim):
        assert (name in self.__class__.__sources), "Source name not \
                recognized, use:\n{}".format(self.__class__.__sources)
        self.name = name
        self.cells = cells
        self.cell_edges = cell_edges
        self.angle_x = angle_x
        self.groups = groups
        self.energy_grid = energy_grid
        self.qdim = qdim

    def _create_source(self):
        if self.qdim == 0:
            self.source = np.zeros((1))
        elif self.qdim == 1:
            self.source = np.zeros((self.cells))
        elif self.qdim == 2:
            self.source = np.zeros((self.cells, self.groups))
        elif self.qdim == 3:
            self.source = np.zeros((self.cells, len(self.angle_x), self.groups))

    def _source(self):
        self._create_source()
        if self.name in ["unity", "half-unity", "small"]:
            self._uniform()
        elif self.name == "reed":
            self._reed()
        elif self.name == "mms-03":
            self._mms_03()
        elif self.name == "mms-04":
            self._mms_04()
        elif self.name == "mms-05":
            self._mms_05()
        elif self.name == "ambe":
            self._ambe()
        return self.source

    def _uniform(self):
        if self.name == "unity":
            self.source[(...)] = 1.0
        elif self.name == "half-unity":
            self.source[(...)] = 0.5
        elif self.name == "small":
            self.source[(...)] = 0.01

    def _reed(self):
        source_values = [0, 1, 0, 50, 0, 1, 0]
        lhs = [0., 2., 4., 6., 10., 12., 14.]
        rhs = [2., 4., 6., 10., 12., 14., 16.]
        loc = lambda x: int(np.argwhere(self.cell_edges == x))
        bounds = [slice(loc(ii), loc(jj)) for ii, jj in zip(lhs, rhs)]
        for ii in range(len(bounds)):
            self.source[bounds[ii]] = source_values[ii]

    def _mms_03(self):
        const1 = 0.5
        const2 = 0.25
        cell_centers = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
        def dependence(mu):
            return const2 * mu * np.exp(mu) * 2 * cell_centers + const1 \
                + const2 * cell_centers**2 * np.exp(mu) - 0.9 \
                / 2 * (2 * const1 + const2 * cell_centers**2 * (np.exp(1) - np.exp(-1)))
        for n, mu in enumerate(self.angle_x):
            self.source[:,n] = dependence(mu)[:,None]

    def _mms_04(self):
        width = 2
        cell_centers = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
        def quasi(x, mu):
            c = 0.3
            return 2 * width * mu - 4 * x * mu - 2 * x**2 \
                   + 2 * width * x - c * (-2 * x**2 + 2 * width * x)
        def scatter(x, mu):
            c = 0.9
            const = -0.125 * width + 0.5 * width**2
            return 0.25 * (mu + x) + const - c * ((0.25 * x + const))
        for n, mu in enumerate(self.angle_x):
            idx = [cell_centers < (0.5 * width)]
            self.source[idx,n] = quasi(cell_centers[idx], mu)[:,None]
            idx = [cell_centers > (0.5 * width)]
            self.source[idx,n] = scatter(cell_centers[idx], mu)[:,None]

    def _mms_05(self):
        width = 2
        cell_centers = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
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
        for n, mu in enumerate(self.angle_x):
            idx = [cell_centers < (0.5 * width)]
            self.source[idx,n] = quasi(cell_centers[idx], mu)[:,None]
            idx = [cell_centers > (0.5 * width)]
            self.source[idx,n] = scatter(cell_centers[idx], mu)[:,None]

    def _ambe(self):
        AmBe = np.load(SOR_PATH + "AmBe_source_050G.npz")
        if np.max(self.energy_grid) > 20:
            self.energy_grid *= 1E-6
        g_centers = 0.5 * (self.energy_grid[1:] + self.energy_grid[:-1])
        locs = lambda x1, x2: np.argwhere((g_centers > x1) & (g_centers <= x2)).flatten()
        medium_center = 0.5 * max(self.cell_edges)
        center_idx = np.where(abs(self.cell_edges - medium_center) == \
                                abs(self.cell_edges - medium_center).min())[0]
        for ii in range(len(AmBe["magnitude"])):
            idx = locs(AmBe["edges"][ii], AmBe["edges"][ii+1])
            for jj in center_idx:
                self.source[(jj, ..., idx)] = AmBe["magnitude"][ii]

class BoundaryCondition:
    __available_sources = ("14.1-mev", "zero", "unit", "mms-03", \
                            "mms-04", "mms-05")

    def __init__(self, name, location, angle_x, groups, bcdim, energy_grid):
        assert (name in self.__class__.__available_sources), \
                "Boundary name not recognized, use:\n{}".format( \
                self.__class__.__available_sources)
        self.name = name
        self.location = location
        self.angle_x = angle_x
        self.groups = groups
        self.bcdim = bcdim
        self.energy_grid = energy_grid

    def _create_boundary(self):
        if self.bcdim == 0:
            self.boundary = np.zeros((2))
        elif self.bcdim == 1:
            self.boundary = np.zeros((2, self.groups))
        elif self.bcdim == 2:
            self.boundary = np.zeros((2, len(self.angle_x), self.groups))

    def _run(self):
        self._create_boundary()
        if self.name == "unit":
            self._unit()
        elif self.name == "14.1-mev":
            self._mev14()
        elif self.name == "ambe":
            self._ambe()
        elif self.name in ["mms-03", "mms-04", "mms-05"]:
            self._mms()
        return self.boundary

    def _mev14(self):
        group = np.argmin(abs(self.energy_grid - 14.1E6))
        self.boundary[(self.location, ..., group)] = 1.0

    def _unit(self):
        self.boundary[self.location] = 1.0

    def _mms(self):
        const1 = 0.5
        const2 = 0.25
        width = 2
        if self.name == "mms-03":
            self.boundary[0] = const1
            self.boundary[1] = const1 + const2 * np.exp(self.angle_x)
        elif self.name == "mms-04":
            self.boundary[1] = 0.5 * width**2 + 0.125 * width
        elif self.name == "mms-05":
            self.boundary[1] = width**3
