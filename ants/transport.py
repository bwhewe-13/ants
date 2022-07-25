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

import numpy as np
import os
import shutil
import pkg_resources
import warnings

class Transport:
    
    __parameters = ("PROBLEM", "OUTPUT", "FILE LOCATION", "FILE NAME", \
        "SPATIAL GEOMETRY", "SPATIAL DISCRETE", "SPATIAL X CELLS", \
        "SPATIAL X LENGTH", "SPATIAL X CELL WIDTH", "SPATIAL Y CELLS", \
        "SPATIAL Y LENGTH", "SPATIAL Y CELL WIDTH", "ANGLES", \
        "ANGLES COLLIDED", "TIME DISCRETE", "TIME STEPS", "TIME STEP SIZE", \
        "ENERGY GROUPS", "ENERGY GROUPS COLLIDED", "ENERGY BOUNDS", \
        "ENERGY INDEX", "MATERIAL", "MATERIAL", "MAP FILE", \
        "EXTERNAL SOURCE", "EXTERNAL SOURCE FILE", "POINT SOURCE LOCATION", \
        "POINT SOURCE NAME", "BOUNDARY X", "BOUNDARY Y", "SVD", "DJINN", \
        "AUTOENCODER", "DJINN-AUTOENCODER", "HYBRID", "MMS", "MNB", "NOTE")

    def __init__(self, input_file):
        self.input_file = input_file
        self._read_input()
        self.create_problem()

    def __str__(self):
        string = ""
        space = 33
        for kk, vv in self.info.items():
            if kk == "NOTE":
                continue
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
        self._generate_angles()
        self._generate_medium_map()
        self._generate_cross_sections()
        self._generate_external_source()
        self._generate_point_source()
        self._generate_parameter_list()
        # Move These To Run Function
        self.external_source = self.external_source.flatten()
        self.point_source = self.point_source.flatten()

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
        
    def save_data(self, file_name):
        file = {}
        # Medium data
        file["geometry"] = self.info.get("GEOMETRY")
        file["medium-map"] = self.medium_map
        file["material-key"] = self.material_key
        # Cross sections
        file["groups"] = self.groups
        file["xs-total"] = self.xs_total
        file["xs-scatter"] = self.xs_scatter
        file["xs-fission"] = self.xs_fission
        # Spatial Info
        file["cells"] = self.cells
        file["cell-width"] = self.cell_width
        file["spatial"] = self.info.get("SPATIAL DISCRETE")
        # Angular Info
        file["angles"] = self.angles
        file["mu"] = self.mu
        file["angle-weight"] = self.angle_weight
        if self.info.get("GEOMETRY") == "slab":
            lhs_x = "vacuum"
        elif self.info.get("GEOMETRY") == "sphere":
            lhs_x = "reflected"
        file["boundary"] = [lhs_x, self.info.get("BOUNDARY X")]
        # Time Info
        if self.info.get("TIME STEPS", None) is not None:        
            file["time-steps"] = int(self.info.get("TIME STEPS"))
            file["time-step-size"] = float(self.info.get("TIME STEP SIZE"))
            file["temporal"] = self.info.get("TIME DISCRETE")
        # Extra
        file["notes"] = self.info.get("NOTE")
        np.savez(file_name, **file)

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
                elif key == "NOTE":
                    notes.append(value)
                else:
                    value = "-".join(value.split())
                    self.info[self.info.get(key, key)] = value
        self.info["NOTE"] = "\n".join(notes)
        self.info["MATERIAL"] = "\n".join(self.materials)

    def _generate_parameters(self):
        self.cells = int(self.info.get("SPATIAL X CELLS"))
        medium_width = float(self.info.get("SPATIAL X LENGTH"))
        if "SPATIAL X CELL WIDTH" in self.info.keys():
            file_name = os.path.join(self.info.get("FILE LOCATION", "."), \
                    self.info.get("SPATIAL X CELL WIDTH"))
            self.cell_width = np.load(file_name)
            if self.cell_width.shape[0] != self.cells:
                message = ("Mismatch in cell widths and number of cells, "
                "adjusting spatial cells to equal number of cell widths")
                warnings.warn(message)
                self.cells = self.cell_width.shape[0]
        else:
            self.cell_width = np.repeat(medium_width/self.cells, self.cells)
        self.cell_edges = np.insert(np.round(np.cumsum(self.cell_width), 8), 0, 0)
        self.angles = int(self.info.get("ANGLES"))
        self.groups = int(self.info.get("ENERGY GROUPS"))
        
    def _generate_angles(self):
        self.mu, self.angle_weight = np.polynomial.legendre.leggauss(self.angles)
        self.angle_weight /= np.sum(self.angle_weight)
        # left hand boundary at cell_x = 0 is reflective - negative
        if self.info.get("BOUNDARY X") == "reflected":
            self.mu = self.mu[:int(0.5 * self.angles)]
            self.angle_weight = self.angle_weight[:int(0.5 * self.angles)]
        elif self.info.get("GEOMETRY") == "sphere":
            self.mu = self.mu[int(0.5 * self.angles):]
            self.angle_weight = self.angle_weight[int(0.5 * self.angles):]

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
            starting_loc = int(material[2].split("-")[0])
            ending_loc = int(material[2].split("-")[1])
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
                    self.groups, self.info.get("ENERGY BOUNDS", None), \
                    self.info.get("ENERGY INDEX", None))
        for idx in range(len(self.material_key.items())):
            material = reversed_key[idx]
            self.xs_total.append(creator.material_key[material][0])
            self.xs_scatter.append(creator.material_key[material][1])
            self.xs_fission.append(creator.material_key[material][2])
        self.xs_total = np.array(self.xs_total)
        self.xs_scatter = np.array(self.xs_scatter)
        self.xs_fission = np.array(self.xs_fission)
        self.velocity = creator.velocity

    def _generate_external_source(self):
        # delta = np.repeat(self.cell_width, self.cells)
        creator = problem_setup.ExternalSource( \
                                self.info.get("EXTERNAL SOURCE", None), \
                                self.cells, self.cell_width, self.mu)
        self.external_source = creator._generate_source()
        if self.info.get("EXTERNAL SOURCE FILE", False):
            file_name = os.path.join(self.info.get("FILE LOCATION", "."), \
                    self.info.get("EXTERNAL SOURCE FILE"))
            file_source = np.load(file_name)
            assert (self.external_source.shape == file_source.shape), \
                ("External source size will not match with problem")
            self.external_source += file_source
        
    def _generate_point_source(self):
        self.point_source_loc = self.info.get("POINT SOURCE LOCATION", 0)
        if self.point_source_loc == "right-edge":
            self.point_source_loc = int(self.info.get("SPATIAL X CELLS"))
        creator = problem_setup.PointSource( \
            self.info.get("POINT SOURCE NAME", None), self.mu, \
            self.groups, self.info.get("ENERGY BOUNDS", None), \
            self.info.get("ENERGY INDEX", None))
        self.point_source = creator._generate_source()

    def _generate_parameter_list(self):
        external_group_index = 1
        external_angle_index = 1
        if self.external_source.ndim == 2:
            external_group_index = int(self.info.get("ENERGY GROUPS"))
        elif self.external_source.ndim == 3:
            external_group_index = int(self.info.get("ENERGY GROUPS"))
            external_angle_index = int(self.info.get("ANGLES"))
        point_source_group_index = 1
        if self.point_source.ndim == 2:
            point_source_group_index = int(self.info.get("ENERGY GROUPS"))
        self.params = np.array([
            PARAMS_DICT[self.info.get("GEOMETRY")],
            PARAMS_DICT[self.info.get("SPATIAL DISCRETE")],
            PARAMS_DICT[self.info.get("BOUNDARY X")],
            external_group_index,
            external_angle_index,
            self.point_source_loc,
            point_source_group_index,
            PARAMS_DICT[self.info.get("TIME DISCRETE","None")],
            PARAMS_DICT[self.info.get("TIME STEPS","None")]
            ], dtype=np.int32)
