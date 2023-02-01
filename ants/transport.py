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
from ants.constants import PARAMS_DICT
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
        self._generate_parameters()
        self._generate_medium_map()
        # self._generate_x_angles()
        self._x_angles()
        self._energy_grid()
        # self._generate_xy_angles()
        
        self._generate_cross_sections()
        self._external_source()
        # self._generate_boundary()
        self._boundary_conditions()
        # self._generate_parameter_list()
        self._parameters()
        # Move These To Run Function
        self.external_source = self.external_source.flatten()
        self.boundary = self.boundary.flatten()

    def change_param(self, name, value):
        if name.upper() in self.__class__.__parameters:
            self.info[name.upper()] = "-".join(str(value).lower().split())
        else:
            raise KeyError("Not an Input File Key\nAvailable Keys:\
                            \n{}".format(self.__class__.__parameters))
        self.create_problem()

    def save_input_file(self, file_name=None):
        PATH = pkg_resources.resource_filename("ants","../examples/")
        if file_name is None:
            response = input("Overwrite current file (Y/N): ")
            if response.lower() == "n":
                file_name = input("Type Save File Name: ")
            elif response.lower() == "y":
                file_name = self.input_file
            else:
                print("Invalid Input")
        shutil.copyfile(PATH + "template.inp", file_name)
        with open(file_name) as fp:
            text = [x for x in fp.read().splitlines()]
        for kk, vv in self.info.items():
            kk += ": "
            if kk not in text:
                continue
            ii = text.index(kk)
            if "\n" in vv:
                for jj, item in enumerate(vv.split("\n")):
                    text.insert(ii+jj, kk + item)
            else:
                text[ii] = kk + vv
        remove_excess = lambda line: (len(line) == 0) \
                    or ((len(line) > 0) and (line.rstrip()[-1] != ":"))
        text = list(filter(remove_excess, text))
        with open(file_name, "w") as fp:
            for line in text:
                fp.write("{}\n".format(line))
        
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

    def _generate_parameters(self):
        self.cells = int(self.info.get("X CELLS"))
        if self.info.get("X CELL WIDTH", "uniform") == "uniform":
            medium_width = float(self.info.get("X LENGTH"))
            self.cell_width = np.repeat(medium_width / self.cells, self.cells)
        else:
            self.cell_width = np.load(os.path.join(self.info.get( \
                    "FILE LOCATION", "."), self.info.get("X CELL WIDTH")))
            if self.cell_width.shape[0] != self.cells:
                message = ("Mismatch in cell widths and number of cells, "
                "adjusting spatial cells to equal number of cell widths")
                warnings.warn(message)
                self.cells = self.cell_width.shape[0]
        self.cell_edges = np.insert(np.round(np.cumsum(self.cell_width), 8), 0, 0)
        self.angles = int(self.info.get("ANGLES"))
        assert (self.angles % 2 == 0), "Must be even number of angles"
        self.groups = int(self.info.get("ENERGY GROUPS"))
        self.bc = self.info.get("BOUNDARY X").split("-")
        self.bc = [PARAMS_DICT[ii] for ii in self.bc]
        
    def _generate_x_angles(self):
        self.angle_x, self.angle_w = np.polynomial.legendre.leggauss(self.angles)
        self.angle_w /= np.sum(self.angle_w)
        # left hand boundary at cell_x = 0 is reflective - negative
        if self.info.get("BOUNDARY X") == "reflected":
            self.angles = int(0.5 * self.angles)
            self.angle_x = self.angle_x[self.angle_x > 0].copy()
            self.angle_w = self.angle_w[self.angle_x > 0].copy()
        elif self.info.get("GEOMETRY") == "sphere":
            self.angles = int(0.5 * self.angles)
            self.angle_x = self.angle_x[self.angle_x < 0].copy()
            self.angle_w = self.angle_w[self.angle_x < 0].copy()

    def _x_angles(self):
        self.angle_x, self.angle_w = np.polynomial.legendre.leggauss(self.angles)
        self.angle_w /= np.sum(self.angle_w)
        if self.bc in [[1, 0]]:
            self.angle_x = sorted(self.angle_x, key=lambda x: (abs(x), x < 0))[::-1]
        elif self.bc in [[0, 0], [0, 1]]:
            self.angle_x = sorted(self.angle_x, key=lambda x: (abs(x), x > 0))[::-1]
        self.angle_w = np.sort(self.angle_w)
        self.angle_x = np.array(self.angle_x)

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
            one_width = int(np.argwhere(self.cell_edges == ending_loc)) \
                        - int(np.argwhere(self.cell_edges == starting_loc))
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
        self.velocity = creator.velocity

    def _external_source(self):
        self.external_source = problem_setup.ExternalSource(self.info.get("SOURCE NAME"), \
                    self.cells, self.cell_edges, self.angle_x, self.groups, \
                    self.energy_grid, int(self.info.get("SOURCE DIMENSION", 1)))._source()
        if self.info.get("SOURCE FILE", False):
            add_source = np.load(os.path.join(self.info.get("FILE LOCATION", "."), \
                            self.info.get("SOURCE FILE")))
            assert (self.external_source.shape == add_source.shape), \
                ("External source size will not match with problem")
            self.external_source += add_source
        # self.external_source = self.external_source.flatten()

    # def _generate_external_source(self):
    #     # delta = np.repeat(self.cell_width, self.cells)
    #     creator = problem_setup.ExternalSource( \
    #                             self.info.get("EXTERNAL SOURCE", None), \
    #                             self.cells, self.cell_width, self.angle_x)
    #     self.external_source = creator._generate_source()
    #     if self.info.get("EXTERNAL SOURCE FILE", False):
    #         file_name = os.path.join(self.info.get("FILE LOCATION", "."), \
    #                 self.info.get("EXTERNAL SOURCE FILE"))
    #         file_source = np.load(file_name)
    #         assert (self.external_source.shape == file_source.shape), \
    #             ("External source size will not match with problem")
    #         self.external_source += file_source
        
    def _energy_grid(self):
        if self.groups == 1 or self.info.get("ENERGY GRID", None) is None:
            self.energy_grid = np.arange(self.groups + 1)
            self.grid_index = dimensions.index_generator(self.groups, self.groups)
        else:
            label = str(self.info.get("ENERGY GRID")).zfill(3)
            name = "G{}_energy_grid.npy".format(label)
            self.energy_grid = np.load(ENR_PATH + name)
            if self.groups == 361:
                label = str(self.groups).zfill(3)
                self.grid_index = np.load(ENR_PATH + "G361_grid_index.npz")
                self.grid_index = self.grid_index[label]
            else:
                self.grid_index = dimensions.index_generator( \
                        int(self.info.get("ENERGY GRID")), self.groups)


    def _boundary_conditions(self):
        # self.bc = self.info.get("BOUNDARY X").replace("vacuum", "0").replace("reflect", "1")
        # self.bc = [int(ii) for ii in self.bc.split("-")]
        if int(self.info.get("BOUNDARY DIMENSION", 0)) == 0:
            self.boundary = np.zeros((2))
        else:
            name = self.info.get("BOUNDARY NAME", "zero")
            if self.info.get("BOUNDARY LOCATION") == "left":
                location = 0
            elif self.info.get("BOUNDARY LOCATION") == "right":
                location = 1
            else:
                location = None
            ndim = int(self.info.get("BOUNDARY DIMENSION"))
            self.boundary = problem_setup.BoundaryCondition(name, location, \
                            self.angle_x, self.groups, self.energy_grid, \
                            self.grid_index, ndim)._boundary()
            # self.boundary = self.boundary.flatten()
            # self.boundary = np.zeros((self.angles))
        # self.boundary_loc = self.info.get("BOUNDARY LOCATION", 0)
        # if self.boundary_loc == "right":
        #     self.boundary_loc = int(self.info.get("X CELLS"))

    # def _generate_boundary(self):
    #     self.boundary_loc = self.info.get("BOUNDARY LOCATION", 0)
    #     if self.boundary_loc == "right":
    #         self.boundary_loc = int(self.info.get("X CELLS"))
    #     creator = problem_setup.BoundaryCondition( \
    #         self.info.get("BOUNDARY NAME", None), self.angle_x, \
    #         self.groups, self.info.get("ENERGY BOUNDS", None), \
    #         self.info.get("ENERGY INDEX", None))
    #     self.boundary = creator._generate_source()

    # def _generate_parameter_list(self):
    #     external_group_index = 1
    #     external_angle_index = 1
    #     if self.external_source.ndim == 2:
    #         external_group_index = int(self.info.get("ENERGY GROUPS"))
    #     elif self.external_source.ndim == 3:
    #         external_group_index = int(self.info.get("ENERGY GROUPS"))
    #         external_angle_index = int(self.info.get("ANGLES"))
    #     boundary_group_index = 1
    #     if self.boundary.ndim == 2:
    #         boundary_group_index = int(self.info.get("ENERGY GROUPS"))
    #     self.params = np.array([
    #         PARAMS_DICT[self.info.get("GEOMETRY")],
    #         PARAMS_DICT[self.info.get("SPATIAL")],
    #         # PARAMS_DICT[self.info.get("BOUNDARY X")],
    #         PARAMS_DICT["vacuum"],
    #         external_group_index,
    #         external_angle_index,
    #         # self.boundary_loc,
    #         0,
    #         boundary_group_index,
    #         PARAMS_DICT[self.info.get("TEMPORAL","None")],
    #         PARAMS_DICT[self.info.get("TIME STEPS","None")]
    #         ], dtype=np.int32)

    def _parameters(self):
        self.params = {"cells": self.cells, "angles": self.angles, \
                       "groups": self.groups, "bc": self.bc, \
                       "materials": len(self.material_key.items()), \
                       "geometry": PARAMS_DICT[self.info.get("GEOMETRY")], \
                       "spatial": PARAMS_DICT[self.info.get("SPATIAL")], \
                       "qdim": int(self.info.get("SOURCE DIMENSION", 0)), \
                       "bcdim": int(self.info.get("BOUNDARY DIMENSION", 0))}
