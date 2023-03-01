########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# The main driver for the ants package. This file is automatically 
# loaded with the __init__ file.
# 
########################################################################

from ants import problem_setup
# from ants.constants import PARAMS_DICT
from ants.constants import *
from ants.utils import dimensions

import numpy as np
import os
import shutil
import pkg_resources
import warnings

ENR_PATH = pkg_resources.resource_filename("ants","data/energy/")

class Transport:
    
    __parameters = ("FILE LOCATION", "GEOMETRY", "SPATIAL", "X CELLS", \
                    "X LENGTH", "X CELL WIDTH", "Y CELLS", "Y LENGTH", \
                    "Y CELL WIDTH", "ANGLES", "TEMPORAL", "TIME STEPS", \
                    "TIME STEP SIZE", "ENERGY GROUPS", "ENERGY GRID", \
                    "GRID INDEX", "MATERIAL", "SOURCE NAME", "SOURCE FILE", \
                    "SOURCE DIMENSION", "BOUNDARY X", "BOUNDARY Y", \
                    "BOUNDARY NAME", "BOUNDARY LOCATION", "BOUNDARY DIMENSION")

    def __init__(self, input_file):
        self.input_file = input_file
        self._read_input()
        self.create_problem()

    def __str__(self):
        string = ""
        space = 33
        for kk, vv in self.info.items():
            if kk == "MATERIAL" and "\n" in vv:
                for ele in vv.split("\n"):
                    temp = "{: <{}}{: >{}}\n".format(kk, space, ele, space)
                    temp = temp.replace("  ", "..")
                    string += temp
            else:
                temp = "{: <{}}{: >{}}\n".format(kk, space, vv, space)
                temp = temp.replace("  ", "..")
                string += temp
        return string

    def create_problem(self):
        self._generate_general()
        self._generate_medium_map()
        self._generate_x_angles()
        self._generate_energy_grid()
        # self._generate_xy_angles()
        self._generate_cross_sections()
        self._generate_sources()
        self._generate_boundary_conditions()
        # self._generate_parameter_list()
        self._generate_parameters()

    def change_param(self, name, value):
        if name.upper() in self.__class__.__parameters:
            self.info[name.upper()] = "-".join(str(value).lower().split())
        else:
            raise KeyError("Not an Input File Key\nAvailable Keys:\
                            \n{}".format(self.__class__.__parameters))
        self.create_problem()
        
    def _read_input(self):
        self.info = {}
        self.materials = []
        notes = []
        with open(self.input_file, "r") as fp:
            for line in fp:
                if (line[0] == "#") or (line[:2] == "\n"):
                    continue
                key, value = line.split(":")
                value = value.strip().lower()
                if key == "MATERIAL":
                    self.materials.append(value)
                else:
                    value = "-".join(value.split())
                    self.info[self.info.get(key, key)] = value
        self.info["MATERIAL"] = "\n".join(self.materials)

    def _generate_general(self):
        self.cells = int(self.info.get("X CELLS"))
        if self.info.get("X CELL WIDTH", "uniform") == "uniform":
            medium_width = float(self.info.get("X LENGTH"))
            self.delta_x = np.repeat(medium_width / self.cells, self.cells)
        else:
            self.delta_x = np.load(os.path.join(self.info.get( \
                    "FILE LOCATION", "."), self.info.get("X CELL WIDTH")))
            if self.delta_x.shape[0] != self.cells:
                message = ("Mismatch in cell widths and number of cells, "
                "adjusting spatial cells to equal number of cell widths")
                warnings.warn(message)
                self.cells = self.delta_x.shape[0]
        self.edges_x = np.insert(np.round(np.cumsum(self.delta_x), 8), 0, 0)
        self.angles = int(self.info.get("ANGLES"))
        assert (self.angles % 2 == 0), "Must be even number of angles"
        self.groups = int(self.info.get("ENERGY GROUPS"))
        self.bc = self.info.get("BOUNDARY X").split("-")
        self.bc = [PARAMS_DICT[ii] for ii in self.bc]
        
    def _generate_x_angles(self):
        self.angle_x, self.angle_w = np.polynomial.legendre.leggauss(self.angles)
        self.angle_w /= np.sum(self.angle_w)
        # left hand boundary at cell_x = 0 is reflective - negative
        if self.bc in [[1, 0]] or self.info.get("GEOMETRY") == "sphere":
            self.angles = int(0.5 * self.angles)
            self.angle_x = self.angle_x[self.angle_x < 0].copy()
            self.angle_w = self.angle_w[self.angle_x < 0].copy()
        elif self.bc in [[0, 1]]:
            self.angles = int(0.5 * self.angles)
            self.angle_x = self.angle_x[self.angle_x > 0].copy()
            self.angle_w = self.angle_w[self.angle_x > 0].copy()

    def _ordering_xy_angles(w, nx, ny, bc):
        angles = np.vstack((w, nx, ny))
        if np.sum(bc) == 1:
            if bc[0] == [0, 1]:
                angles = angles[:,angles[1].argsort()[::-1]]
            elif bc[0] == [1, 0]:
                angles = angles[:,angles[1].argsort()]
            elif bc[1] == [0, 1]:
                angles = angles[:,angles[2].argsort()[::-1]]
            elif bc[1] == [1, 0]:
                angles = angles[:,angles[2].argsort()]
        elif np.sum(bc) == 2:
            if bc[0] == [0, 1] and bc[1] == [0, 1]:
                angles = angles[:,angles[1].argsort()]
                angles = angles[:,angles[2].argsort(kind="mergesort")[::-1]]
            elif bc[0] == [1, 0] and bc[1] == [0, 1]:
                angles = angles[:,angles[1].argsort()[::-1]]
                angles = angles[:,angles[2].argsort(kind="mergesort")[::-1]]
            elif bc[0] == [0, 1] and bc[1] == [1, 0]:
                angles = angles[:,angles[1].argsort()[::-1]]
                angles = angles[:,angles[2].argsort(kind="mergesort")]
            elif bc[0] == [1, 0] and bc[1] == [1, 0]:
                angles = angles[:,angles[1].argsort()]
                angles = angles[:,angles[2].argsort(kind="mergesort")]
        elif np.sum(bc) > 2:
            message = ("There must only be one reflected boundary "
                        "in each direction")
            warnings.warn(message)
        return angles

    def _xy_angles(self):
        # eta, xi, mu: direction cosines (x,y,z) 
        xx, wx = np.polynomial.legendre.leggauss(self.angles)
        yy, wy = np.polynomial.chebyshev.chebgauss(self.angles)
        idx = 0
        eta = np.zeros(2 * self.angles**2)
        xi = np.zeros(2 * self.angles**2)
        mu = np.zeros(2 * self.angles**2)
        w = np.zeros(2 * self.angles**2)
        for ii in range(self.angles):
            for jj in range(self.angles):
                mu[idx:idx+2] = xx[ii]
                eta[idx] = np.sqrt(1 - xx[ii]**2) * np.cos(np.arccos(yy[jj]))
                eta[idx+1] = np.sqrt(1 - xx[ii]**2) * np.cos(-np.arccos(yy[jj]))
                xi[idx] = np.sqrt(1 - xx[ii]**2) * np.sin(np.arccos(yy[jj]))
                xi[idx+1] = np.sqrt(1 - xx[ii]**2) * np.sin(-np.arccos(yy[jj]))
                w[idx:idx+2] = wx[ii] * wy[jj]
                idx += 2
        w, eta, xi = Transport._ordering_xy_angles(w[mu > 0] / np.sum(w[mu > 0]), \
                                            eta[mu > 0], xi[mu > 0], self.bc)
        # Convert to naming convention
        self.angle_w = w.copy()
        self.angle_x = eta.copy()
        self.angle_y = xi.copy()

    def _generate_medium_map(self):
        mat_widths = []
        mat_id = []
        mat_start = []
        self.material_key = {}
        starting_cell = 0
        idx = 0
        for material in self.materials:
            material = material.split("//")
            self.material_key[material[1].strip()] = int(material[0])
            mat_id.append(int(material[0].strip()))
            starting_loc = np.round(float(material[2].split("-")[0]), 5)
            ending_loc = np.round(float(material[2].split("-")[1]), 5)
            one_width = int(np.argwhere(self.edges_x == ending_loc)) \
                        - int(np.argwhere(self.edges_x == starting_loc))
            starting_cell += idx
            mat_widths.append(one_width)
            mat_start.append(starting_cell)
            idx = one_width
        mat_id = np.array(mat_id)[np.argsort(mat_start)]
        mat_widths = np.array(mat_widths, dtype=np.int32)[np.argsort(mat_start)]
        mat_start = np.sort(mat_start)
        self.medium_map = np.ones((self.cells)) * -1
        for idx, size, mat in zip(mat_start, mat_widths, mat_id):
            self.medium_map[idx:idx+size] = mat
        assert np.all(self.medium_map != -1)
        self.medium_map = self.medium_map.astype(np.int32)

    def _generate_cross_sections(self):
        reversed_key = {v: k for k, v in self.material_key.items()}
        self.xs_total = []
        self.xs_scatter = []
        self.xs_fission = []
        creator = problem_setup.Materials(list(self.material_key.keys()), \
                            self.groups, self.energy_grid, self.grid_index)
        for idx in range(len(self.material_key.items())):
            material = reversed_key[idx]
            self.xs_total.append(creator.material_key[material][0])
            self.xs_scatter.append(creator.material_key[material][1])
            self.xs_fission.append(creator.material_key[material][2])
        self.xs_total = np.array(self.xs_total)
        self.xs_scatter = np.array(self.xs_scatter)
        self.xs_fission = np.array(self.xs_fission)
        # self.velocity = creator.velocity

    def _generate_sources(self):
        name = self.info.get("SOURCE NAME", "none")
        qdim = int(self.info.get("SOURCE DIMENSION", 1))
        self.source = problem_setup.Source(name, self.cells, self.edges_x, \
                self.angle_x, self.groups, self.energy_grid, qdim)._source()
        if self.info.get("SOURCE FILE", False):
            add_source = np.load(os.path.join(self.info.get("FILE LOCATION", "."), \
                            self.info.get("SOURCE FILE")))
            assert (self.source.shape == add_source.shape), \
                ("External source size will not match with problem")
            self.source += add_source

    def _generate_energy_grid(self):
        # Create energy grid
        grid = int(self.info.get("ENERGY GRID", 0))
        if grid in [87, 361, 618]:
            name = "energy_bounds.npz"
            self.energy_grid = np.load(ENR_PATH + name)[str(grid)]
        else:
            self.energy_grid = np.arange(self.groups + 1)
        if self.groups == 361:
            label = str(self.groups).zfill(3)
            self.grid_index = np.load(ENR_PATH + "G361_grid_index.npz")
            self.grid_index = self.grid_index[label]
        else:
            self.grid_index = dimensions.index_generator( \
                                len(self.energy_grid - 1), self.groups)

    def _generate_boundary_conditions(self):
        name = self.info.get("BOUNDARY NAME", "zero")
        location = self.info.get("BOUNDARY LOCATION").split("-")
        location = [PARAMS_DICT[ii] for ii in location]
        bcdim = int(self.info.get("BOUNDARY DIMENSION", 0))
        self.boundary = problem_setup.BoundaryCondition(name, location, \
                self.angle_x, self.groups, bcdim, self.energy_grid)._run()

    def _generate_parameters(self):
        self.params = {"cells": self.cells, "angles": self.angles,
                       "groups": self.groups, "bc": self.bc,
                       "materials": len(self.material_key.items()),
                       "geometry": PARAMS_DICT[self.info.get("GEOMETRY")],
                       "spatial": PARAMS_DICT[self.info.get("SPATIAL")],
                       "qdim": int(self.info.get("SOURCE DIMENSION", 0)),
                       "bcdim": int(self.info.get("BOUNDARY DIMENSION", 0)),
                       "steps": 0, "dt": 1,
                       "angular": True, "adjoint": False}

def calculate_velocity(groups, energy_edges=None):
    """ Convert energy edges to speed at cell centers, Relative Physics
    Arguments:
        groups: Number of energy groups
        energy_edges: energy grid bounds
    Returns:
        speeds at cell centers (cm/s)   """
    if energy_edges == None:
        return np.ones((groups))
    energy_centers = 0.5 * (energy_edges[1:] + energy_edges[:-1])
    gamma = (EV_TO_JOULES * energy_centers) / (MASS_NEUTRON * LIGHT_SPEED**2) + 1
    velocity = LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100
    return velocity

# def calculate_x_angles(angles, bc=[0, 0], geometry="slab"):
def calculate_x_angles(params):
    angle_x, angle_w = np.polynomial.legendre.leggauss(params["angles"])
    angle_w /= np.sum(angle_w)
    # left hand boundary at cell_x = 0 is reflective - negative
    if params["bc"] in [[1, 0]] or params["geometry"] == 2:
        params["angles"] = int(0.5 * params["angles"])
        angle_x = angle_x[angle_x < 0].copy()
        angle_w = angle_w[angle_x < 0].copy()
    elif params["bc"] in [[0, 1]]:
        params["angles"] = int(0.5 * params["angles"])
        angle_x = angle_x[angle_x > 0].copy()
        angle_w = angle_w[angle_x > 0].copy()
    return angle_x, angle_w