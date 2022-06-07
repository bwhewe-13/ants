########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Module to create material cross sections and point sources for varying
# enrichments and energy groups (full and collapsed). Also calculates 
# neutron velocity. External sources are found in ants.medium.
#
########################################################################

import ants.constants as const
import ants.mapper as mapper

import numpy as np
import pkg_resources


MAT_PATH = pkg_resources.resource_filename("ants","data/materials/")
SOR_PATH = pkg_resources.resource_filename("ants","data/sources/")


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
                               "mms-absorber", "mms-scatter", "mms-quasi")

    __materials = __enrichment_materials + __nonenrichment_materials \
                    + __nonphysical_materials

    def __init__(self, materials, energy_groups, energy_bounds, \
                 energy_idx=None):
        """ Creating cross sections for different materials

        Args:
            materials (list): [material1, ..., materialN]
            energy_groups (int):
            energy_bounds (list):
            energy_idx (list):

        """
        assert (isinstance(materials, list)), "Materials must be list"
        for mater in materials:
            assert (mater.split("-%")[0] in self.__class__.__materials),\
                    "Material not recognized, use:\n{}".format(\
                        self.__class__.__materials)
        self.materials = materials
        self.energy_groups = energy_groups
        self.energy_bounds = energy_bounds
        self.energy_idx = energy_idx
        self.data = {}
        self.p_sources = {}
        self.compile_cross_sections()
        self.compile_velocity()

    def __str__(self):
        msg = "Energy Groups: {}\nMaterial: ".format(self.energy_groups)
        msg += "\nMaterial: ".join(self.materials)
        return msg

    def __len__(self):
        return len(self.materials)

    def __iter__(self):
        return iter((self.velocity, self.data.keys(), self.data.values()))

    def compile_cross_sections(self):
        for material in self.materials:
            material_name = "material-" + material
            cross_sections = Materials._generate_cross_section(self, \
                                                               material)
            if len(cross_sections[0]) != self.energy_groups:
                cross_sections = Materials._generate_reduced_cross_section(\
                                            self, cross_sections)            
            self.data[material_name] = cross_sections

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
            return func(self.energy_groups)
        return [data["total"], data["scatter"], data["fission"]]

    def _generate_enrich(material):
        material = material.split("-%")
        material[1] = float(material[1].strip("%")) * 0.01
        return material

    def _generate_reduced_cross_section(self, cross_sections):
        total, scatter, fission = cross_sections
        if self.energy_idx is None:
            self.energy_idx = Materials._generate_index(self)
        bin_width = np.diff(self.energy_bounds)
        coarse_bin_width = np.array([self.energy_bounds[self.energy_idx[ii+1]] \
                            -self.energy_bounds[self.energy_idx[ii]] \
                            for ii in range(self.energy_groups)])
        total = (Materials._vector_reduction(total * bin_width, \
                 self.energy_idx)) / coarse_bin_width
        scatter = (Materials._matrix_reduction(scatter * bin_width, \
                 self.energy_idx)) / coarse_bin_width
        fission = (Materials._matrix_reduction(fission * bin_width, \
                 self.energy_idx)) / coarse_bin_width
        return [total, scatter, fission]

    def _generate_index(self):
        energy_grid = np.ones((self.energy_groups)) \
            * int((len(self.energy_bounds) - 1) / self.energy_groups)
        temp_idx = np.linspace(0, self.energy_groups - 1, \
            (len(self.energy_bounds) - 1) % self.energy_groups, dtype=int)
        energy_grid[temp_idx] += 1
        return np.cumsum(np.insert(energy_grid, 0, 0), dtype=int)

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

    def _matrix_reduction(matrix, idx):
        energy_groups = len(idx) - 1
        return [[np.sum(matrix[idx[ii]:idx[ii+1],idx[jj]:idx[jj+1]]) \
            for jj in range(energy_groups)] for ii in range(energy_groups)]

    def _vector_reduction(vector, idx):
        energy_groups = len(idx) - 1
        return [sum(vector[idx[ii]:idx[ii+1]]) \
                    for ii in range(energy_groups)]

    def compile_velocity(self):
        if self.materials[0] in self.__class__.__nonphysical_materials:
            self.velocity = np.ones((self.energy_groups))
        else:            
            energy_centers = 0.5 * (self.energy_bounds[1:] + \
                                    self.energy_bounds[:-1])
            gamma = (const.EV_TO_JOULES * energy_centers) \
                    / (const.MASS_NEUTRON * const.LIGHT_SPEED**2) + 1
            velocity = const.LIGHT_SPEED / gamma \
                        * np.sqrt(gamma**2 - 1) * 100
            if self.energy_idx is None:
                self.energy_idx = Materials._generate_index(self)
            velocity = [np.mean(velocity[self.energy_idx[group]: \
                        self.energy_idx[group+1]]) for group \
                        in range(self.energy_groups)]
            self.velocity = np.array(velocity)

    def add_point_source(self, name, location, angles):
        if self.energy_idx is None and self.energy_groups > 1:
            self.energy_idx = Materials._generate_index(self)
        source = PointSources(name, angles, self.energy_groups, \
                    self.energy_bounds, self.energy_idx)
        self.p_sources[name] = [location, source._generate_source()]


class PointSources:
    __available_sources = ("14.1-mev", "ambe", "single-left", \
                           "mms-left", "mms-right", "mms-two-material", \
                           "mms-two-material-angle")

    def __init__(self, name, angles, energy_groups, energy_bounds, \
                 energy_idx):
        assert (name in self.__class__.__available_sources), \
        "Source not recognized, use:\n{}".format(\
            self.__class__.__available_sources)
        self.name = name
        self.angles = angles
        self.energy_groups = energy_groups
        self.energy_bounds = energy_bounds
        self.energy_idx = energy_idx

    def _generate_source(self):
        if self.name in ["14.1-mev"]:
            source = self._mev14_source()
        elif self.name in ["ambe"]:
            source = self._ambe_source()
        elif self.name in ["single-left"]:
            source = self._single_source_left()
        elif self.name in ["mms-left"]:
            source = self._mms_boundary_left()
        elif self.name in ["mms-right"]:
            source = self._mms_boundary_right()
        elif self.name in ["mms-two-material"]:
            source = self._mms_two_material_right()
        elif self.name in ["mms-two-material-angle"]:
            source = self._mms_two_material_angle_right()
        if source.shape[1] != self.energy_groups:
            return Materials._vector_reduction(source, self.energy_idx) 
        return source

    def _mev14_source(self):
        source = np.zeros((len(self.energy_bounds) - 1))
        group = np.argmin(abs(self.energy_bounds - 14.1E6))
        source[group] = 1
        return source

    def _ambe_source(self):
        AmBe = np.load(SOR_PATH + "AmBe_source_050G.npz")
        if np.max(self.energy_bounds) > 20:
            self.energy_bounds *= 1E-6
        energy_centers = 0.5 * (self.energy_bounds[1:] + \
                                self.energy_bounds[:-1])
        locs = lambda xmin, xmax: np.argwhere((energy_centers > xmin) & \
                            (energy_centers <= xmax)).flatten()
        source = np.zeros((len(self.energy_bounds) - 1))
        for center in range(len(AmBe["magnitude"])):
            idx = locs(AmBe["edges"][center], AmBe["edges"][center+1])
            source[idx] = AmBe["magnitude"][center]
        return source

    def _single_source_left(self):
        source = np.ones((len(self.angles), self.energy_groups))
        source[self.angles < 0] = 0
        return source

    def _mms_boundary_left(self):
        # at x = 0
        psi_constant_01 = 0.5
        psi_constant_02 = 0.25
        source = np.zeros((len(self.angles), self.energy_groups))
        source[self.angles > 0] = psi_constant_01
        return source

    def _mms_boundary_right(self):
        # at x = X = 1
        psi_constant_01 = 0.5
        psi_constant_02 = 0.25
        source = np.zeros((len(self.angles), self.energy_groups))
        source[self.angles < 0] = psi_constant_01 + psi_constant_02 \
                          * np.exp(self.angles[self.angles < 0])[:,None]
        return source

    def _mms_two_material_right(self):
        # at x = X = 2
        X = 2
        source = np.zeros((len(self.angles), self.energy_groups))
        source[self.angles < 0] = 0.5 * X**2 + 0.125 * X
        return source
    
    def _mms_two_material_angle_right(self):
        # at x = X = 2
        X = 2
        source = np.zeros((len(self.angles), self.energy_groups))
        source[self.angles < 0] = X**3
        # source[self.angles < 0] = 0.25 * X * np.exp(self.angles[self.angles < 0])[:,None] \
        #     + X**2 - 0.125 * np.exp(self.angles[self.angles < 0])[:,None] * X *(4*X + 1)
        return source

class NonPhysical:
    def reed_scatter(energy_groups):
        if energy_groups == 1:
            return [np.array([10.]), np.array([[9.9]]), np.array([[0.]])]
            # return [np.array([1.]), np.array([[0.9]]), np.array([[0.]])]

    def reed_absorber(energy_groups):
        if energy_groups == 1:
            return [np.array([5.]), np.array([[0.]]), np.array([[0.]])]

    def reed_vacuum(energy_groups):
        if energy_groups == 1:
            return [np.array([0.]), np.array([[0.]]), np.array([[0.]])]

    def reed_strong_source(energy_groups):
        if energy_groups == 1:
            return [np.array([50.]), np.array([[0.]]), np.array([[0.]])]

    def mms_absorber(energy_groups):
        if energy_groups == 1:
            return [np.array([1.]), np.array([[0.]]), np.array([[0.]])]

    def mms_scatter(energy_groups):
        if energy_groups == 1:
            return [np.array([1.]), np.array([[0.9]]), np.array([[0.]])]
            # return [np.array([1.]), np.array([[0.0]]), np.array([[0.]])]

    def mms_quasi(energy_groups):
        if energy_groups == 1:
            return [np.array([1.]), np.array([[0.3]]), np.array([[0.]])]
            # return [np.array([1.]), np.array([[0.0]]), np.array([[0.]])]

if __name__ == "__main__":
    materials = ["reed-absorber","reed-strong-source",\
                 "reed-vacuum", "reed-scatter"]
    problem = Materials(materials, 1, None)
    print(problem.data.keys())

# [total, scatter, fission]