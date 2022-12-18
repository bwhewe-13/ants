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

import ants.constants as const
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
        self.compile_velocity()

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
        partial_molar_mass = enrichment * const.URANIUM_235_MM \
                    + (1 - enrichment) * const.URANIUM_238_MM
        rho = const.URANIUM_HYDRIDE_RHO / const.URANIUM_RHO
        n235 = (enrichment * rho * partial_molar_mass) \
                / (partial_molar_mass + 3 * const.HYDROGEN_MM) 
        n238 = ((1 - enrichment) * rho * partial_molar_mass) \
                / (partial_molar_mass + 3 * const.HYDROGEN_MM) 
        n1 = const.URANIUM_HYDRIDE_RHO * const.AVAGADRO_NUMBER \
                / (partial_molar_mass + 3 * const.HYDROGEN_MM) \
                * const.CM_TO_BARNS * 3
        u235 = np.load(MAT_PATH + "uranium-235.npz")
        u238 = np.load(MAT_PATH + "uranium-238.npz")
        h1 = np.load(MAT_PATH + "hydrogen.npz")
        total = n235 * u235["total"] + n238 * u238["total"] \
                + n1 * h1["total"]
        scatter = n235 * u235["scatter"] + n238 * u238["scatter"] \
                + n1 * h1["scatter"]
        fission = n235 * u235["fission"] + n238 * u238["fission"] \
                + n1 * h1["fission"]
        return total, scatter, fission

    def compile_velocity(self):
        if self.materials[0] in self.__class__.__nonphysical_materials:
            self.velocity = np.ones((self.groups))
        else:            
            centers = 0.5 * (self.energy_grid[1:] + self.energy_grid[:-1])
            gamma = (const.EV_TO_JOULES * energy_centers) \
                    / (const.MASS_NEUTRON * const.LIGHT_SPEED**2) + 1
            velocity = [const.LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100]
            self.velocity = np.array([np.mean(velocity[self.grid_index[gg]: \
                        self.grid_index[gg+1]]) for gg in range(self.groups)])


class BoundaryCondition:
    __available_sources = ("14.1-mev", "zero", "unit", "mms-01", \
                            "mms-02", "mms-03", "mms-04")

    def __init__(self, name, location, angle_x, groups, energy_grid, \
                    grid_index, dimension):
        assert (name in self.__class__.__available_sources), \
                "Boundary name not recognized, use:\n{}".format( \
                self.__class__.__available_sources)
        self.name = name
        self.location = location
        self.angles = len(angle_x)
        self.angle_x = angle_x
        self.groups = groups
        self.energy_grid = energy_grid
        self.grid_index = grid_index
        if dimension == 1:
            self.boundary = np.zeros((2, self.groups))
        elif dimension == 2:
            self.boundary = np.zeros((2, self.angles, self.groups))
        # self.energy_grid = energy_grid
        # self.grid_index = grid_index
        # self.energy_grid = np.load(ENR_PATH + "energy_grid.npz")
        # self.energy_grid = self.energy_grid[str(self.groups)]

    def _boundary(self):
        if self.name == "unit":
            self._unit()
        elif self.name == "14.1-mev":
            self._mev14()
        elif self.name == "ambe":
            self._ambe()
        elif self.name in ["mms-01", "mms-02", "mms-03", "mms-04"]:
            self._mms()
        # elif self.name in ["mms-left"]:
        #     source = self._mms_boundary_left()
        # elif self.name in ["mms-right"]:
        #     source = self._mms_boundary_right()
        # elif self.name in ["mms-two-material"]:
        #     source = self._mms_two_material_right()
        # elif self.name in ["mms-two-material-angle"]:
        #     source = self._mms_two_material_angle_right()
        if self.boundary.shape[-1] != self.groups:
            return dimensions.vector_reduction(self.boundary, self.grid_index)
        return self.boundary

    def _mev14(self):
        group = np.argmin(abs(self.energy_grid - 14.1E6))
        self.boundary[(self.location, ..., group)] = 1.0

    def _unit(self):
        self.boundary[self.location] = 1.0

    def _mms(self):
        psi_c1 = 0.5
        psi_c2 = 0.25
        XX = 2
        if self.name == "mms-01":
            self.boundary[0] = psi_c1
        elif self.name == "mms-02":
            self.boundary[1] = psi_c1 + psi_c2 * np.exp(self.angle_x)
        elif self.name == "mms-03":
            self.boundary[1] = 0.5 * XX**2 + 0.125 * XX
        elif self.name == "mms-04":
            self.boundary[1] = XX**3

    # def _mms_boundary_left(self):
    #     # at x = 0
    #     psi_constant_01 = 0.5
    #     psi_constant_02 = 0.25
    #     source = np.zeros((len(self.mu), self.groups))
    #     source[self.mu > 0] = psi_constant_01
    #     return source

    # def _mms_boundary_right(self):
    #     # at x = X = 1
    #     psi_constant_01 = 0.5
    #     psi_constant_02 = 0.25
    #     source = np.zeros((len(self.mu), self.groups))
    #     source[self.mu < 0] = psi_constant_01 + psi_constant_02 \
    #                       * np.exp(self.mu[self.mu < 0])[:,None]
    #     return source

    # def _mms_two_material_right(self):
    #     # at x = X = 2
    #     X = 2
    #     source = np.zeros((len(self.mu), self.groups))
    #     source[self.mu < 0] = 0.5 * X**2 + 0.125 * X
    #     return source
    
    # def _mms_two_material_angle_right(self):
    #     # at x = X = 2
    #     X = 2
    #     source = np.zeros((len(self.mu), self.groups))
    #     source[self.mu < 0] = X**3
    #     # source[self.mu < 0] = 0.25 * X * np.exp(self.mu[self.mu < 0])[:,None] \
    #     #     + X**2 - 0.125 * np.exp(self.mu[self.mu < 0])[:,None] * X *(4*X + 1)
    #     return source


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


class ExternalSource:
    __available_sources = ("none", "unity", "half-unity", "reed", "mms-source", \
                           "mms-two-material", "mms-two-material-angle", \
                           "diffusion-01", "ambe")

    def __init__(self, name, cells, cell_edges, angle_x, groups, \
                        energy_grid, dimension):

        assert (name in self.__class__.__available_sources), \
            "External Source name not recognized, use:\n{}".format(\
            self.__class__.__available_sources)
        self.name = name
        self.cells = cells
        self.cell_edges = cell_edges
        self.cell_width = np.diff(cell_edges)
        self.angles = len(angle_x)
        self.angle_x = angle_x
        self.groups = groups
        self.energy_grid = energy_grid
        if dimension == 1:
            self.source = np.zeros((self.cells))
        elif dimension == 2:
            self.source = np.zeros((self.cells, self.groups))
        elif dimension == 3:
            self.source = np.zeros((self.cells, self.angles, self.groups))

        # self.medium_radius = np.round(np.sum(cell_width), 8)

    def _source(self):
        if self.name in ["none", "unity", "half-unity", "small"]:
            self._uniform()
        elif self.name == "reed":
            self._reed()
        elif self.name in ["mms-01", "mms-02", "mms-03"]:
            # self._mms()
            ...
        elif self.name == "ambe":
            self._ambe()

        # elif self.name in ["mms-source"]:
        #     return self._mms_source()
        # elif self.name in ["mms-two-material"]:
        #     return self._mms_two_material()
        # elif self.name in ["mms-two-material-angle"]:
        #     return self._mms_two_material_angle()
        return self.source

    def _uniform(self):
        self.source = 1.0
        if self.name == "none":
            self.source *= 0.0
        elif self.name == "half-unity":
            self.source *= 0.5
        elif self.name == "small":
            self.source *= 0.01

    def _reed(self):
        source_values = [0, 1, 0, 50, 0, 1, 0]
        lhs = [0., 2., 4., 6., 10., 12., 14.]
        rhs = [2., 4., 6., 10., 12., 14., 16.]
        cell_edges = np.insert(np.round(np.cumsum(self.cell_width), 8), 0, 0)
        loc = lambda x: int(np.argwhere(cell_edges == x))
        bounds = [slice(loc(ii), loc(jj)) for ii, jj in zip(lhs, rhs)]
        for ii in range(len(bounds)):
            self.source[bounds[ii]] = source_values[ii]

    # def _mms_source(self):
    #     psi_c1 = 0.5
    #     psi_c2 = 0.25
    #     xspace = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
    #     def angle_dependent(angle):
    #         return psi_c2 * angle * np.exp(angle) * 2 * xspace \
    #                 + psi_c1 + psi_c2 * xspace**2 \
    #                 * np.exp(angle) - 0.9 / 2 * (2 * psi_c1 \
    #                 + psi_c2 * xspace**2 * (np.exp(1) - np.exp(-1)))
    #     source = np.array([angle_dependent(angle) for angle in self.mu]).T
    #     return source[:,:,None]

    # def _mms_two_material(self):
    #     X = max(self.cell_edges)
    #     xspace = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
    #     def angle_dependent(angle, x):
    #         def quasi(x):
    #             c = 0.3
    #             return 2 * X * angle - 4 * x * angle - 2 * x**2 \
    #                    + 2 * X * x - c * (-2 * x**2 + 2 * X * x)
    #         def scatter(x):
    #             c = 0.9
    #             const = -0.125 * X + 0.5 * X**2
    #             return 0.25 * angle + 0.25 * x + const \
    #                    - c * ((0.25 * x + const))
    #         return np.concatenate([quasi(xspace[xspace < 0.5 * X]), \
    #                                scatter(xspace[xspace > 0.5 * X])])
    #     source = np.array([angle_dependent(angle, xspace) for angle \
    #                                                     in self.angle_x]).T
    #     return source[:,:,None]

    # def _mms_two_material_angle(self):
    #     X = max(self.cell_edges)
    #     xspace = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
    #     def angle_dependent(angle, x):
    #         def quasi(x, angle):
    #             c = 0.3
    #             return angle * (2 * X**2 - 4 * np.exp(angle) * x) - 2 \
    #                     * np.exp(angle) * x**2 + 2 * X**2 * x - c / 2 \
    #                     * (-2 * x**2 * (np.exp(1) - np.exp(-1)) + 4 * X**2 * x)
    #         def scatter(x, angle):
    #             c = 0.9
    #             const = X**3 - X**2 * np.exp(angle)
    #             return X * angle * np.exp(angle) + X * x * np.exp(angle) + const \
    #                    - c/2 * (2 * X**3 + (np.exp(1) - np.exp(-1)) * (x * X - X**2))
    #         return np.concatenate([quasi(xspace[xspace < 0.5 * X], angle), \
    #                                scatter(xspace[xspace > 0.5 * X], angle)])
    #     source = np.array([angle_dependent(angle, xspace) for angle \
    #                                                     in self.mu]).T
    #     return source[:,:,None]
    

    # def quasi(x, angle=None):
    #     c = 0.3
    #     return angle * (2 * X**2 - 4 * np.exp(angle) * x) - 2 \
    #             * np.exp(angle) * x**2 + 2 * X**2 * x - c / 2 \
    #             * (-2 * x**2 * (np.exp(1) - np.exp(-1)) + 4 * X**2 * x)


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

    # def _mms_source(self):
    #     psi_constant_01 = 0.5
    #     psi_constant_02 = 0.25
    #     xspace = np.linspace(0, self.medium_radius, self.cells+1)
    #     xspace = 0.5 * (xspace[1:] + xspace[:-1])
    #     def angle_dependent(angle):
    #         return psi_constant_02 * angle * np.exp(angle) * 2 * xspace \
    #                 + psi_constant_01 + psi_constant_02 * xspace**2 \
    #                 * np.exp(angle) - 0.9 / 2 * (2 * psi_constant_01 \
    #                 + psi_constant_02 * xspace**2 * (np.exp(1) - np.exp(-1)))
    #     source = np.array([angle_dependent(angle) for angle in self.mu]).T
    #     return source[:,:,None]

    # def _mms_two_material(self):
    #     X = self.medium_radius
    #     xspace = np.linspace(0, self.medium_radius, self.cells+1)
    #     xspace = 0.5 * (xspace[1:] + xspace[:-1])
    #     def angle_dependent(angle, x):
    #         def quasi(x):
    #             c = 0.3
    #             return 2 * X * angle - 4 * x * angle - 2 * x**2 \
    #                    + 2 * X * x - c * (-2 * x**2 + 2 * X * x)
    #         def scatter(x):
    #             c = 0.9
    #             const = -0.125 * X + 0.5 * X**2
    #             return 0.25 * angle + 0.25 * x + const \
    #                    - c * ((0.25 * x + const))
    #         return np.concatenate([quasi(xspace[xspace < 0.5 * X]), \
    #                                scatter(xspace[xspace > 0.5 * X])])
    #     source = np.array([angle_dependent(angle, xspace) for angle \
    #                                                     in self.mu]).T
    #     return source[:,:,None]

    # def _mms_two_material_angle(self):
    #     X = self.medium_radius
    #     xspace = np.linspace(0, self.medium_radius, self.cells+1)
    #     xspace = 0.5 * (xspace[1:] + xspace[:-1])
    #     def angle_dependent(angle, x):
    #         def quasi(x, angle):
    #             c = 0.3
    #             return angle * (2 * X**2 - 4 * np.exp(angle) * x) - 2 \
    #                     * np.exp(angle) * x**2 + 2 * X**2 * x - c / 2 \
    #                     * (-2 * x**2 * (np.exp(1) - np.exp(-1)) + 4 * X**2 * x)
    #         def scatter(x, angle):
    #             c = 0.9
    #             const = X**3 - X**2 * np.exp(angle)
    #             return X * angle * np.exp(angle) + X * x * np.exp(angle) + const \
    #                    - c/2 * (2 * X**3 + (np.exp(1) - np.exp(-1)) * (x * X - X**2))
    #         return np.concatenate([quasi(xspace[xspace < 0.5 * X], angle), \
    #                                scatter(xspace[xspace > 0.5 * X], angle)])
    #     source = np.array([angle_dependent(angle, xspace) for angle \
    #                                                     in self.mu]).T
    #     return source[:,:,None]

