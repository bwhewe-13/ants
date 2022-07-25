########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from ants.transport import Transport
from ants.cyants import multi_group
from . import splines, dimensions
# from .math import root_mean_squared_error as rmse

import numpy as np
import os


class NearbyProblem:

    def __init__(self, input_file):
        self.input_file = input_file
        self.problem = Transport(self.input_file)

    def adjust_parameters(self):
        cell_width = self.problem.cell_width
        cells = self.problem.cells
        self.xspace = self.problem.cell_edges
        self.xspace = 0.5*(self.xspace[1:] + self.xspace[:-1])
        self.splits = dimensions.create_slices(self.problem.medium_map)        

    def create_nearby(self):
        self.adjust_parameters()
        # Step 1: Run the numerical problem
        self.numerical = self.run_problem()
        # Step 2: Calculate an analytical solution
        self.curve_fit()
        # Step 3: Calculate an analytical source
        self.residual()
        # Step 4: Add the analytical source to the original problem
        self.run_nearby()

    def curve_fit(self):
        self.cf_scalar = np.zeros((self.numerical.shape[0], \
                                   self.numerical.shape[-1]))
        self.cf_angular = np.zeros(self.numerical.shape)
        self.cf_deriv = np.zeros(self.numerical.shape)
        weight = self.problem.angle_weight
        for angle in range(self.problem.angles):
            for split in self.splits:
                yspline, deriv = splines.hermite(self.xspace[split], \
                                    self.numerical[:,angle,0][split], \
                                    knots=None, stype="quintic")
                self.cf_angular[:,angle,0][split] = yspline.copy()
                self.cf_deriv[:,angle,0][split] = deriv.copy()
                self.cf_scalar[:,0][split] += weight[angle] * yspline

    def residual(self):
        # Used to calculate the analytical source
        mu = self.problem.mu
        xs_total = self.problem.xs_total[:,0]
        xs_scatter = self.problem.xs_scatter
        xs_fission = self.problem.xs_fission
        external_source = self.problem.external_source.copy()
        external_source = external_source.reshape(self.problem.cells, \
                                self.problem.angles, self.problem.groups)
        medium_map = np.array(self.problem.medium_map, dtype=np.int32)
        # Analytical source
        self.cf_source = np.zeros((external_source.shape))
        for angle in range(self.problem.angles):
            psi = self.cf_angular[:, angle, 0].copy()
            phi = self.cf_scalar[:, 0].copy()
            dpsi = self.cf_deriv[:, angle, 0].copy()
            for cell, mat in enumerate(self.problem.medium_map):
                self.cf_source[cell, angle, 0] = (mu[angle] * dpsi[cell] \
                            + psi[cell] * xs_total[mat]) - ((xs_scatter[mat] \
                            + xs_fission[mat]) * phi[cell] \
                            + external_source[cell, angle, 0])
        address = os.path.join(self.problem.info.get("FILE LOCATION", "."), \
                                                    "analytical_source")
        # address = self.problem.info.get("FILE LOCATION", ".")
        np.save(address, self.cf_source)

    def run_nearby(self):
        self.problem.change_param("external source file", "analytical_source.npy")
        self.nearby = self.run_problem()

    def run_problem(self):
        angular = multi_group.source_iteration(self.problem.medium_map, \
                        self.problem.xs_total, self.problem.xs_scatter, \
                        self.problem.xs_fission, self.problem.external_source, \
                        self.problem.point_source, self.problem.mu, \
                        self.problem.angle_weight, self.problem.params, \
                        self.problem.cell_width, angular=True)
        return angular

    def save_nearby(self, file_name):
        mnb_data = {}
        try:
            mnb_data["nearby"] = self.nearby.copy()
        except NameError:
            self.create_nearby()
            mnb_data["nearby"] = self.nearby.copy()
        mnb_data["numerical"] = self.numerical.copy()
        mnb_data["curve_fit"] = self.cf_angular.copy()
        mnb_data["analytical_source"] = self.cf_source.copy()
        np.savez(file_name, **mnb_data)
        print("Nearby Problem Saved!")
