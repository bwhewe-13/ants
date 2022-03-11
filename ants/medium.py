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

    def __init__(self, cells_x, cell_width_x, angles_x, boundary_x):
        """ Constructing the spatial grid

        Args:
            cells_x (int): 
            cell_width_x (float):
            angles_x (int, even):
            boundary_x (list of length 2): [0: Vacuum, 1: Reflective]
        """
        assert (sum(boundary_x) != 2), "Both boundaries cannot be \
         reflective, one must be a vacuum"
        self.cells_x = cells_x
        self.cell_width_x = cell_width_x
        self.angles_x = angles_x
        self.boundary_x = boundary_x
        self.compile_medium()
        self.spatial_coef_x = self.mu_x / self.cell_width_x

    def __iter__(self):
        return iter((self.cells_x, self.cell_width_x, self.mu_x, self.weight))

    def compile_medium(self):
        mu, w = np.polynomial.legendre.leggauss(self.angles_x)
        w /= np.sum(w)
        # left hand boundary at cell_x = 0 is reflective - negative
        if self.boundary_x[0] == 1:
            mu = mu[:int(0.5 * self.angles_x)]
            w = w[:int(0.5 * self.angles_x)]
        elif self.boundary_x[1] == 1:
            mu = mu[int(0.5 * self.angles_x):]
            w = w[int(0.5 * self.angles_x):]
        self.mu_x = mu
        self.weight = w

    def add_external_source(self, name):
        source = ExternalSources(name, self.cells_x, self.cell_width_x)
        # source_name = 'external-source-' + name
        # self.ex_sources[source_name] = source._generate_source()
        self.ex_source = source._generate_source()


class ExternalSources:
    __available_sources = ("unity", "half-unity", "reed")

    def __init__(self, name, cells, cell_width):
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
        self.medium_radius = int(cells * cell_width)

    def _generate_source(self):
        if self.name in ["unity"]:
            source = np.ones((self.cells, 1)) 
        elif self.name in ["half-unity"]:
            source = 0.5 * np.ones((self.cells, 1))
        elif self.name in ["reed"]:
            return self._reed_source()
        return source

    def _reed_source(self):
        source_values = [0,1,0,0,50,0,0,1,0]
        boundaries = [slice(0, int(2/self.cell_width)),\
                slice(int(2/self.cell_width), int(3/self.cell_width)),\
                slice(int(3/self.cell_width), int(5/self.cell_width)),\
                slice(int(5/self.cell_width), int(6/self.cell_width)),\
                slice(int(6/self.cell_width), int(10/self.cell_width)),\
                slice(int(10/self.cell_width), int(11/self.cell_width)),\
                slice(int(11/self.cell_width), int(13/self.cell_width)),\
                slice(int(13/self.cell_width), int(14/self.cell_width)),\
                slice(int(14/self.cell_width), int(16/self.cell_width))]
        source = np.zeros((self.cells, 1))
        for boundary in range(len(boundaries)):
            source[boundaries[boundary]] = source_values[boundary]
        return source
