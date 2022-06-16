########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from ants.transport import Transport
from . import splines, dimensions
from .math import root_mean_squared_error as rmse

import numpy as np
import matplotlib.pyplot as plt

class NearbyProblem:

    def __init__(self, input_file, knots=None):
        self.input_file = input_file
        self.knots = knots
        self.problem = Transport(self.input_file)
        cell_width = self.problem.medium_obj.cell_width
        cells = self.problem.medium_obj.cells
        self.xspace = np.linspace(0, cells*cell_width, cells+1)
        self.xspace = 0.5*(self.xspace[1:] + self.xspace[:-1])
        self.splits = dimensions.create_slices(self.problem.medium_map)

    def run_nearby(self, display=False):
        # Step 1: Run the numerical problem
        ################################################################
        self.numerical_scalar, self.numerical_angular = self.problem.run()
        # Step 2: Calculate an analytical solution
        ################################################################
        self.analytical_solution()
        # Step 3: Calculate an analytical source
        ################################################################
        self.residual(save=True)
        # Step 4: Add the analytical source to the original problem
        ################################################################
        self.problem.change_param("external source file", "analytical_source.npy")
        self.nearby_scalar, self.nearby_angular = self.problem.run()
        # Step 5: Evaluate the discretization error
        ################################################################
        self.discretization_error()
        if display is True:
            self.graph()

    def analytical_solution(self, spline_type="cubic"):
        self.analytical_scalar = np.zeros((self.numerical_angular.shape[0], \
                                      self.numerical_angular.shape[-1]))
        self.analytical_angular = np.zeros(self.numerical_angular.shape)
        weight = self.problem.medium_obj.weight
        for angle in range(self.problem.medium_obj.angles):
            for split in self.splits:
                yspline, idx = splines.hermite(self.xspace[split], \
                                self.numerical_angular[:,angle,0][split], \
                                self.knots, spline_type)
                self.analytical_angular[:,angle,0][split] = yspline.copy()
                self.analytical_scalar[:,0][split] += weight[angle] * yspline

    def residual(self, scalar_flux=None, angular_flux=None, save=False):
        # Used to calculate the analytical source
        # Problem parameters
        mu = self.problem.medium_obj.mu
        xs_total = self.problem.xs_total[:,0]
        xs_scatter = self.problem.xs_scatter
        xs_fission = self.problem.xs_fission
        external_source = self.problem.medium_obj.ex_source
        # Analytical source
        self.source = np.zeros((external_source.shape))
        for angle in range(self.problem.medium_obj.angles):
            for split in self.splits:
                if scalar_flux is None:
                    phi = self.analytical_scalar[:,0][split].copy()
                    psi = self.analytical_angular[:,angle,0][split].copy()
                else:
                    phi = scalar_flux[:,0][split].copy()
                    psi = angular_flux[:,0][split].copy()
                dpsi = splines.first_derivative(self.xspace[split], psi)
                for cell, mat in enumerate(self.problem.medium_map[split]):
                    self.source[split][cell, angle, 0] = (mu[angle] * dpsi[cell] \
                        + psi[cell] * xs_total[mat]) \
                        - ((xs_scatter[mat] + xs_fission[mat]) * phi[cell] \
                        + external_source[split][cell, angle, 0])
        if save:
            address = self.problem.info.get("FILE LOCATION", ".")
            np.save(address + "/analytical_source", self.source)

    def graph(self):
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        for angle in range(self.problem.medium_obj.angles):
            ax[0].plot(self.xspace, self.analytical_angular[:,angle,0], c="k", ls="--")
            ax[0].plot(self.xspace, self.numerical_angular[:,angle,0], c="r", alpha=0.5)
            ax[0].plot(self.xspace, self.nearby_angular[:,angle,0], c="b", alpha=0.5)
        ax[0].plot([], [], label="Analytical", c="k", ls="--")
        ax[0].plot([], [], label="Numerical", c="r", alpha=0.5)
        ax[0].plot([], [], label="Nearby Problem", c="b", alpha=0.5)
        ax[0].grid(which="both")
        ax[0].legend(loc=0, framealpha=1)
        if self.knots is not None:
            ax[0].set_title("Nearby Problem vs {} Hermite Splines".format(knots-1))
        else:
            ax[0].set_title("Nearby Problem vs Hermite Splines")

        for angle in range(self.problem.medium_obj.angles):
            ax[1].plot(self.xspace, self.source[:,angle,0], label="Angle "+str(angle))
        ax[1].grid(which="both")
        ax[1].legend(loc=0, framealpha=1)
        ax[1].set_title("Analytical Source (Residual)")
        plt.show()

    def discretization_error(self):
        print("Method of Nearby Problems Discretization Error")
        print("="*46)
        for angle in range(self.problem.medium_obj.angles):
            error = rmse(self.analytical_angular[:,angle,0], \
                         self.nearby_angular[:,angle,0])
            print("Angle {} RMSE {}".format(angle, error))
        print("="*46)
