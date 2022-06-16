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

from ants.medium import MediumX
from ants.materials import Materials
from ants.mapper import Mapper
from ants.fixed_source import backward_euler
from ants.criticality import keigenvalue

import numpy as np
import os
import shutil
import pkg_resources

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
        self._generate_medium_obj()
        self._generate_materials_obj()
        self._generate_cross_section()

    def __str__(self):
        string = ""
        space = int(0.1 * len(max(self.info.keys(), key=len)) +\
                          len(max(self.info.values(), key=len)))
        for kk, vv in self.info.items():
            if kk == "NOTE":
                continue
            if kk =="MATERIAL" and "\n" in vv:
                vv = ", ".join(vv.split("\n"))
            temp = "{: <{}} {: >{}}\n".format(kk, space, vv, space)
            temp = temp.replace("  ", "..")
            string += temp
        return string

    def change_param(self, name, value):
        # if name.upper() in self.info.keys():
        if name.upper() in self.__class__.__parameters:
            self.info[name.upper()] = "-".join(str(value).lower().split())
        else:
            raise KeyError("Not an Input File Key\nAvailable Keys:\
                            \n{}".format(self.__class__.__parameters))
        self._generate_medium_obj()
        self._generate_materials_obj()
        self._generate_cross_section()

    def run(self):
        if self.info.get("PROBLEM") == "fixed-source":
            scalar, angular = self._run_fixed_source()
            return scalar, angular
        elif self.info.get("PROBLEM") == "criticality":
            self._run_criticality()

    def _run_criticality(self):
        scalar, keff = keigenvalue(self.medium_map, self.xs_total, 
                        self.xs_scatter, self.xs_fission, \
                        self.medium_obj.spatial_coef, self.medium_obj.weight, \
                        spatial=self.info.get("SPATIAL DISCRETE"))
        self.scalar = np.array(scalar)
        self.keff = keff
        return scalar, keff

    def _run_fixed_source(self):
        if self.info.get("TIME DISCRETE", "backward-euler") == "backward-euler":
            scalar, angular = backward_euler(self.medium_map, self.xs_total, \
                        self.xs_scatter, self.xs_fission, \
                        self.medium_obj.ex_source, self.point_source_locs, \
                        self.point_sources, self.medium_obj.spatial_coef, \
                        self.medium_obj.weight, self.materials_obj.velocity, \
                        int(self.info.get("TIME STEPS", "0")), \
                        float(self.info.get("TIME STEP SIZE", "0")), \
                        spatial=self.info.get("SPATIAL DISCRETE"))
        if int(self.info.get("TIME STEPS", "0")) == 0:
            scalar = np.array(scalar[0])
        else:
            scalar = np.array(scalar)
        self.scalar = scalar
        self.angular = angular
        return scalar, angular

    def save_input_file(self, file_name=None):
        PATH = pkg_resources.resource_filename("ants","../examples/")
        if file_name is None:
            file_name = self.input_file
        shutil.copyfile(PATH + "template.inp", file_name)
        # print(self.info["NOTE"])
        with open(file_name) as fp:
            text = [x for x in fp.read().splitlines()]
        for kk, vv in self.info.items():
            kk += ": "
            if kk not in text:
                # print(kk)
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
        
    def save_data(self, file_name=None):
        dictionary = {}
        if file_name is None:
            file_name = self._generate_file_name()
        # Neutron fluxes
        dictionary["flux-scalar"] = self.scalar
        try:
            dictionary["flux-angular"] = self.angular
        except NameError:
            dictionary["k-effective"] = self.keff
        # Medium data
        dictionary["medium-map"] = self.medium_map
        dictionary["medium-key"] = self.map_obj.map_key
        # Cross sections
        dictionary["groups"] = int(self.info.get("ENERGY GROUPS"))
        dictionary["xs-total"] = self.xs_total
        dictionary["xs-scatter"] = self.xs_scatter
        dictionary["xs-fission"] = self.xs_fission
        # Spatial Info
        dictionary["cells-x"] = int(self.info.get("SPATIAL X CELLS"))
        dictionary["cell-width-x"] = self.cell_width
        dictionary["spatial-disc"] = self.info.get("SPATIAL DISCRETE")
        # Angular Info
        dictionary["angles"] = int(self.info.get("ANGLES"))
        if self.info.get("GEOMETRY") == "slab":
            lhs_x = "vacuum"
        elif self.info.get("GEOMETRY") == "sphere":
            lhs_x = "reflected"
        dictionary["boundary-x"] = [lhs_x, self.info.get("BOUNDARY X")]
        # Time Info
        dictionary["time-steps"] = int(self.info.get("TIME STEPS"))
        dictionary["time-step-size"] \
                             = float(self.info.get("TIME STEP SIZE"))
        dictionary["time-disc"] = self.info.get("TIME DISCRETE")
        # Extra
        dictionary["notes"] = self.info.get("NOTE")
        np.savez(file_name, **dictionary)

    def _read_input(self):
        self.info = {}
        with open(self.input_file, "r") as fp:
            for line in fp:
                if line[0] == "#":
                    continue
                if line[:2] == "\n":
                    continue
                key, value = line.split(":")
                if key in self.info.keys() and key == "NOTE":
                    self.info[key] += "\n" + value.strip().lower()
                elif key == "NOTE":
                    self.info[key] = value.strip().lower() 
                elif key in self.info.keys():
                    self.info[key] += "\n" \
                              + "-".join(value.strip().lower().split())
                else:
                    self.info[key] = "-".join(value.strip().lower().split())

    def _generate_boundaries(self):
        def boundary(string):
            if string == "vacuum":
                return 0
            elif string == "reflected":
                return 1
        if self.info.get("GEOMETRY") == "slab":
            lhs_x = 0
        elif self.info.get("GEOMETRY") == "sphere":
            lhs_x = 1
        rhs_x = boundary(self.info.get("BOUNDARY X"))
        xbounds = np.array([lhs_x, rhs_x])
        return xbounds

    def _generate_cross_section(self):
        map_obj_loc = os.path.join(self.info.get("FILE LOCATION", "."), \
                               self.info.get("MAP FILE"))
        self.map_obj = Mapper.load_map(map_obj_loc)
        if int(self.info.get("SPATIAL X CELLS")) != self.map_obj.cells_x:
            self.map_obj.adjust_widths(int(self.info.get("SPATIAL X CELLS")))
        self.medium_map = self.map_obj.map_x.astype(int)
        self.medium_map_key = self.map_obj.map_key
        reversed_key = {v: k for k, v in self.map_obj.map_key.items()}
        total = []
        scatter = []
        fission = []
        for position in range(len(self.map_obj.map_key)):
            map_material = reversed_key[position]
            total.append(self.materials_obj.data[map_material][0])
            scatter.append(self.materials_obj.data[map_material][1])
            fission.append(self.materials_obj.data[map_material][2])
        self.xs_total = np.array(total)
        self.xs_scatter = np.array(scatter)
        self.xs_fission = np.array(fission)

    def _generate_file_name(self):
        file_name = os.path.join(self.info.get("FILE LOCATION", "."), \
                            self.info.get("FILE NAME"))
        file_number = 0
        while os.path.exists(file_name + str(file_number) + ".npz"):
            file_number += 1
        return file_name + str(file_number)

    def _generate_medium_obj(self):
        xbounds = self._generate_boundaries()
        cells = int(self.info.get("SPATIAL X CELLS"))
        if "SPATIAL X LENGTH" in self.info.keys():
            medium_width = float(self.info.get("SPATIAL X LENGTH"))
            cell_width = medium_width / cells
        else:
            cell_width = float(self.info.get("CELL WIDTH X"))
        self.cell_width = cell_width
        angles = int(self.info.get("ANGLES"))
        self.medium_obj = MediumX(cells, cell_width, angles, xbounds)
        self.medium_obj.add_external_source(self.info.get("EXTERNAL SOURCE"))
        if self.info.get("EXTERNAL SOURCE FILE", False):
            file_name = os.path.join(self.info.get("FILE LOCATION", "."), \
                    self.info.get("EXTERNAL SOURCE FILE"))
            file_source = np.load(file_name)
            assert (cells == len(file_source)), ("External source size"
                "will not match with problem")
            self.medium_obj.ex_source += file_source
        
    def _generate_materials_obj(self):
        self.materials_obj = Materials(self.info.get("MATERIAL").split("\n"), \
                                   int(self.info.get("ENERGY GROUPS")), 
                                   None)
        ps_locs = self.info.get("POINT SOURCE LOCATION", "").split("\n")
        ps_names = self.info.get("POINT SOURCE NAME", "").split("\n")
        for loc, name in zip(ps_locs, ps_names):
            try:
                self.materials_obj.add_point_source(name, int(loc), \
                                                    self.medium_obj.mu)
            except ValueError:
                if loc == "right-edge":
                    edge = int(self.info.get("SPATIAL X CELLS"))
                    self.materials_obj.add_point_source(name, edge, \
                                    self.medium_obj.mu)
        locations = []
        point_source = []
        for value in self.materials_obj.p_sources.values():
            locations.append(value[0])
            point_source.append(value[1])
        self.point_source_locs = np.array(locations)
        self.point_sources = np.array(point_source)


if __name__ == "__main__":
    print("Spatial Discretizations")
    print("="*30,"\t1 = Step Method\n\t2 = Diamond Difference\n")
    print("Boundary Conditions")
    print("="*30,"\t0 = Vacuum\n\t1 = Reflected\n")
    print("Temporal Discretizations")
    print("="*30,"\t1 = BDF1 (Backward Euler)\n\t2 = BDF2\n")