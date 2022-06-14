########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np

class MediumX:

    def __init__(self, cells, cell_width, angles, boundary_x):
        """ Constructing the spatial grid

        Args:
            cells (int): 
            cell_width (float):
            angles (int, even):
            boundary_x (list of length 2): [0: Vacuum, 1: Reflective]
        """
        assert (sum(boundary_x) != 2), "Both boundaries cannot be \
         reflective, one must be a vacuum"
        self.cells = cells
        self.cell_width = cell_width
        self.angles = angles
        self.boundary_x = boundary_x
        self.compile_medium()
        self.spatial_coef = self.mu / self.cell_width

    def __iter__(self):
        return iter((self.cells, self.cell_width, self.mu, self.weight))

    def compile_medium(self):
        mu, w = np.polynomial.legendre.leggauss(self.angles)
        w /= np.sum(w)
        # left hand boundary at cell_x = 0 is reflective - negative
        if self.boundary_x[0] == 1:
            mu = mu[:int(0.5 * self.angles)]
            w = w[:int(0.5 * self.angles)]
        elif self.boundary_x[1] == 1:
            mu = mu[int(0.5 * self.angles):]
            w = w[int(0.5 * self.angles):]
        self.mu = mu
        self.weight = w

    def add_external_source(self, name):
        source = ExternalSources(name, self.cells, self.cell_width, \
                                 self.mu)
        # source_name = 'external-source-' + name
        # self.ex_sources[source_name] = source._generate_source()
        self.ex_source = source._generate_source()


class ExternalSources:
    __available_sources = ("unity", "half-unity", "reed", "mms-source", \
                           "mms-two-material", "mms-two-material-angle")

    def __init__(self, name, cells, cell_width, mu):
        """ Constructing external sources - for

        Args:
            name (string):
            cells (int): 
            cell_width (float):
        """
        assert (name in self.__class__.__available_sources), \
        "Source not recognized, use:\n{}".format(\
            self.__class__.__available_sources)
        self.name = name
        self.cells = cells
        self.cell_width = cell_width
        self.mu = mu
        self.medium_radius = int(cells * cell_width)

    def _generate_source(self):
        if self.name in ["unity"]:
            source = np.ones((self.cells, len(self.mu), 1)) 
        elif self.name in ["half-unity"]:
            source = 0.5 * np.ones((self.cells, len(self.mu), 1))
        elif self.name in ["reed"]:
            return self._reed_source()
        elif self.name in ["mms-source"]:
            return self._mms_source()
        elif self.name in ["mms-two-material"]:
            return self._mms_two_material()
        elif self.name in ["mms-two-material-angle"]:
            return self._mms_two_material_angle()
        return source

    def _reed_source(self):
        source_values = [0, 1, 0, 50, 0, 1, 0]
        boundaries = [slice(0, int(2/self.cell_width)),\
                slice(int(2/self.cell_width), int(4/self.cell_width)), \
                slice(int(4/self.cell_width), int(6/self.cell_width)), \
                slice(int(6/self.cell_width), int(10/self.cell_width)), \
                slice(int(10/self.cell_width), int(12/self.cell_width)), \
                slice(int(12/self.cell_width), int(14/self.cell_width)), \
                slice(int(14/self.cell_width), int(16/self.cell_width))]
        source = np.zeros((self.cells, 1))
        for boundary in range(len(boundaries)):
            source[boundaries[boundary]] = source_values[boundary]
        return source

    def _mms_source(self):
        psi_constant_01 = 0.5
        psi_constant_02 = 0.25
        xspace = np.linspace(0, self.medium_radius, self.cells+1)
        xspace = 0.5 * (xspace[1:] + xspace[:-1])
        def angle_dependent(angle):
            return psi_constant_02 * angle * np.exp(angle) * 2 * xspace \
                    + psi_constant_01 + psi_constant_02 * xspace**2 \
                    * np.exp(angle) - 0.9 / 2 * (2 * psi_constant_01 \
                    + psi_constant_02 * xspace**2 * (np.exp(1) - np.exp(-1)))
        source = np.array([angle_dependent(angle) for angle in self.mu]).T
        return source[:,:,None]

    def _mms_two_material(self):
        X = self.medium_radius
        xspace = np.linspace(0, self.medium_radius, self.cells+1)
        xspace = 0.5 * (xspace[1:] + xspace[:-1])
        def angle_dependent(angle, x):
            def quasi(x):
                c = 0.3
                return 2 * X * angle - 4 * x * angle - 2 * x**2 \
                       + 2 * X * x - c * (-2 * x**2 + 2 * X * x)
            def scatter(x):
                c = 0.9
                const = -0.125 * X + 0.5 * X**2
                return 0.25 * angle + 0.25 * x + const \
                       - c * ((0.25 * x + const))
            return np.concatenate([quasi(xspace[xspace < 0.5 * X]), \
                                   scatter(xspace[xspace > 0.5 * X])])
        source = np.array([angle_dependent(angle, xspace) for angle \
                                                        in self.mu]).T
        return source[:,:,None]

    def _mms_two_material_angle(self):
        X = self.medium_radius
        xspace = np.linspace(0, self.medium_radius, self.cells+1)
        xspace = 0.5 * (xspace[1:] + xspace[:-1])
        def angle_dependent(angle, x):
            def quasi(x, angle):
                c = 0.3
                return angle * (2 * X**2 - 4 * np.exp(angle) * x) - 2 \
                        * np.exp(angle) * x**2 + 2 * X**2 * x - c / 2 \
                        * (-2 * x**2 * (np.exp(1) - np.exp(-1)) + 4 * X**2 * x)
            def scatter(x, angle):
                c = 0.9
                const = X**3 - X**2 * np.exp(angle)
                return X * angle * np.exp(angle) + X * x * np.exp(angle) + const \
                       - c/2 * (2 * X**3 + (np.exp(1) - np.exp(-1)) * (x * X - X**2))
            return np.concatenate([quasi(xspace[xspace < 0.5 * X], angle), \
                                   scatter(xspace[xspace > 0.5 * X], angle)])
        source = np.array([angle_dependent(angle, xspace) for angle \
                                                        in self.mu]).T
        return source[:,:,None]
