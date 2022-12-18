########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Running the Method of Nearby Problems for both fixed source and
# criticality problems.
#
########################################################################

from ants.utils import splines, dimensions, interp1d
from ants import x_multi_group

import numpy as np

# params = {"cells": 50, "angles": 4, "groups": 87, "materials": 2, \
#                 "geometry": 1, "spatial": 2, "qdim": 3, "bc": [0, 0], \
#                 "bcdim": 2}

def refine_interface_grid(medium_map, cell_width, dtype="half"):
    material_splits = dimensions.create_slices(medium_map)
    updated_cell_width = []
    for count, split in enumerate(material_splits):
        temp = cell_width[split].copy()
        if dtype == "half":
            if count != (len(material_splits) - 1):
                temp = np.append(temp[:-1], 2 * [temp[-1] * 0.5])
            if count != 0:
                temp = np.append(2 * [temp[0] * 0.5], temp[1:])
        elif dtype == "step":
            if count != (len(material_splits) - 1):
                step_down = np.concatenate((2 * [temp[-2] * 0.75], 3 * [temp[-1] * 0.5]))
                temp = np.append(temp[:-3], np.round(step_down, 10))
            if count != 0:
                step_down = np.concatenate((3 * [temp[0] * 0.5], 2 * [temp[0] * 0.75]))
                temp = np.append(np.round(step_down, 10), temp[3:])
        updated_cell_width.append(temp)
    updated_cell_width = np.array([ii for sublst in updated_cell_width for ii in sublst])
    return updated_cell_width

def reaction_rates(flux, xs, medium_map):
    reaction = np.zeros((len(flux)))
    for cell in range(len(medium_map)):
        mat = medium_map[cell]
        reaction[cell] = np.sum(flux[cell] * np.sum(xs[mat], axis=1))
    return reaction

class FixedSource:

    def __init__(self, xs_total, xs_scatter, xs_fission, source, boundary, \
                medium_map, cell_edges, angle_x, angle_w, params, params_temp):
        self.xs_total = xs_total
        self.xs_scatter = xs_scatter
        self.xs_fission = xs_fission
        self.source = source.reshape(params["cells"], params["angles"], params["groups"])
        self.boundary = boundary
        self.medium_map = medium_map
        self.cell_edges = cell_edges
        self.cell_width = np.round(np.diff(cell_edges), 12)
        self.angle_x = angle_x
        self.angle_w = angle_w
        self.params = params
        self.params_temp = params_temp

    def run(self):
        self._numerical()
        self._curve_fit()
        self._residual()
        self._curve_fit_boundary()
        self._nearby()

    def _numerical(self):
        self.numerical_flux = x_multi_group.source_iteration(self.medium_map, \
                self.xs_total, self.xs_scatter, self.xs_fission, self.source.flatten(), \
                self.boundary, self.angle_x, self.angle_w, self.params_temp, \
                self.cell_width, angular=True)

    def _curve_fit(self):
        cell_centers = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
        material_splits = dimensions.create_slices(self.medium_map)
        self.curve_fit_flux = np.zeros(self.numerical_flux.shape)
        self.curve_fit_dflux = np.zeros(self.numerical_flux.shape)
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for split in material_splits:
                    func = interp1d.QuinticHermite(cell_centers[split], \
                                    self.numerical_flux[:,angle,group][split])
                    self.curve_fit_flux[:,angle,group][split] = func(cell_centers[split])
                    self.curve_fit_dflux[:,angle,group][split] = func.derivative()(cell_centers[split])

    def _residual(self):
        self.residual = np.zeros(self.source.shape)
        scalar_flux = np.sum(self.curve_fit_flux * self.angle_w[None,:,None], axis=1)
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for cell, mat in enumerate(self.medium_map):
                    self.residual[cell,angle,group] = (self.angle_x[angle] \
                            * self.curve_fit_dflux[cell,angle,group] \
                            + self.curve_fit_flux[cell,angle,group] * self.xs_total[mat, group]) \
                            - (scalar_flux[cell] @ self.xs_scatter[mat].T)[group] \
                            - (scalar_flux[cell] @ self.xs_fission[mat].T)[group] \
                            - self.source[cell,angle,group]

    def _curve_fit_boundary(self):
        # Updates the boundary conditions
        self.curve_fit_boundary = np.zeros((self.params["angles"], self.params["groups"]))
        cell_centers = 0.5 * (self.cell_edges[1:] + self.cell_edges[:-1])
        material_splits = dimensions.create_slices(self.medium_map)
        left, right = dimensions.create_slices(self.medium_map)[[0, -1]]
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                if self.angle_x[angle] < 0:
                    func = interp1d.QuinticHermite(cell_centers[right], \
                                        self.numerical_flux[:,angle,group][right])
                    self.curve_fit_boundary[angle, group] = func([self.cell_edges[-1]])
                elif self.angle_x[angle] > 0:
                    func = interp1d.QuinticHermite(cell_centers[left], \
                                        self.numerical_flux[:,angle,group][left])
                    self.curve_fit_boundary[angle, group] = func([self.cell_edges[0]])

    def _nearby(self):
        # Runs the nearby problem
        updated_source = (self.source + self.residual).flatten()
        updated_boundary = self.curve_fit_boundary.flatten()
        self.nearby_flux = x_multi_group.source_iteration(self.medium_map, \
                self.xs_total, self.xs_scatter, self.xs_fission, updated_source, \
                updated_boundary, self.angle_x, self.angle_w, self.params_temp, \
                self.cell_width, angular=True)
