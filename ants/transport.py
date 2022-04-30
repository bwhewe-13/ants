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

from ants.problem.medium import MediumX
from ants.problem.materials import Materials
from ants.problem.mapper import Mapper
from ants.fixed_source import backward_euler
from ants.criticality import keigenvalue

import numpy as np
import os
import matplotlib.pyplot as plt

class Transport:

    def __init__(self, input_file):
        self.input_file = input_file
        self._read_input()
        self._generate_medium_obj()
        self._generate_materials_obj()
        self._generate_cross_section()

    def __str__(self):
        string = ""
        for kk, vv in self.info.items():
            if kk == "NOTE":
                continue
            temp = "{: <22} {: >22}\n".format(kk, vv)
            temp = temp.replace("  ", "..")
            string += temp
        return string

    def change_param(self, name, value):
        if name in self.info.keys():
            self.info[name] = "-".join(str(value).lower().split())
        else:
            raise KeyError("Not in Input File Keys\nAvailable Keys:\
                                    \n{}".format(self.info.keys()))
        self._generate_medium_obj()
        self._generate_materials_obj()
        self._generate_cross_section()

    def graph(self):
        if int(self.info.get("TIME STEPS")) > 0:
            self._generate_plot_rate()
        else:
            self._generate_plot_rate_density()
        plt.show()

    def fission_rate(self, cross_section):
        time_data = np.zeros((int(self.info.get("TIME STEPS"))))
        for time_step in range(len(time_data)):
            time_data[time_step] = np.sum(self.cell_width_x * \
                               self.fission_rate_density(cross_section))
        return time_data

    def fission_rate_density(self, cross_section):
        cell_data = np.zeros((int(self.info.get("CELLS X"))))
        for cell, mat in enumerate(self.medium_map):
            cell_data[cell] = np.sum(self.scalar[cell] \
                                * np.sum(cross_section[mat],axis=1))
        return cell_data

    def run(self):
        if self.info.get("PROBLEM") == "fixed-source":
            self._run_fixed_source()
        elif self.info.get("PROBLEM") == "criticality":
            self._run_criticality()

    def _run_criticality(self):
        scalar, keff = keigenvalue(self.medium_map, \
                                   self.xs_total, \
                                   self.xs_scatter, 
                                   self.xs_fission, \
                                   self.medium_obj.spatial_coef_x, \
                                   self.medium_obj.weight, \
                                   spatial=self.info.get("SPACE DISCRETE").lower())
        self.scalar = np.array(scalar)
        self.keff = keff
        return scalar, keff

    def _run_fixed_source(self):
        if self.info.get("TIME DISCRETE") == "backward-euler":
            scalar, angular = backward_euler(self.medium_map, \
                                    self.xs_total, \
                                    self.xs_scatter, \
                                    self.xs_fission, \
                                    self.medium_obj.ex_source, \
                                    self.point_source_locs, \
                                    self.point_sources, \
                                    self.medium_obj.spatial_coef_x, \
                                    self.medium_obj.weight, \
                                    self.materials_obj.velocity, \
                                    int(self.info.get("TIME STEPS")), \
                                    float(self.info.get("TIME STEP SIZE")), \
                                    spatial=self.info.get("SPACE DISCRETE").lower())
        if int(self.info.get("TIME STEPS")) == 0:
            scalar = np.array(scalar[0])
        else:
            scalar = np.array(scalar)
        self.scalar = scalar
        self.angular = angular
        return scalar, angular

    def run_save(self, file_name=None):
        scalar, angular = self.run()
        self.save(file_name)
        return scalar, angular

    def run_graph(self):
        scalar, angular = self.run()
        self.graph()
        return scalar, angular
        
    def run_graph_save(self, file_name=None):
        scalar, angular = self.run()
        self.save(file_name)
        self.graph()
        return scalar, angular

    def save(self, file_name=None):
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
        dictionary["cells-x"] = int(self.info.get("CELLS X"))
        dictionary["cell-width-x"] = self.cell_width_x
        dictionary["spatial-disc"] = self.info.get("SPACE DISCRETE")
        # Angular Info
        dictionary["angles-x"] = int(self.info.get("ANGLES X"))
        dictionary["boundary-x"] = [self.info.get("BOUNDARY (x = 0)"),
                                    self.info.get("BOUNDARY (x = X)")]
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
                    self.info[key] = \
                                "-".join(value.strip().lower().split())

    def _generate_boundaries(self):
        def boundary(string):
            if string == "vacuum":
                return 0
            elif string == "reflected":
                return 1
        lhs_x = boundary(self.info.get("BOUNDARY (x = 0)"))
        rhs_x = boundary(self.info.get("BOUNDARY (x = X)"))
        xbounds = np.array([lhs_x, rhs_x])
        return xbounds

    def _generate_cross_section(self):
        map_obj_loc = os.path.join(self.info.get("PATH","."), \
                               self.info.get("MAP FILE"))
        self.map_obj = Mapper.load_map(map_obj_loc)
        if int(self.info.get("CELLS X")) != self.map_obj.cells_x:
            self.map_obj.adjust_widths(int(self.info.get("CELLS X")))
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
        file_name = os.path.join(self.info.get("PATH","."), \
                            self.info.get("FILE NAME"))
        file_number = 0
        while os.path.exists(file_name + str(file_number) + ".npz"):
            file_number += 1
        return file_name + str(file_number)

    def _generate_medium_obj(self):
        xbounds = self._generate_boundaries()
        cells_x = int(self.info.get("CELLS X"))
        if "MEDIUM WIDTH X" in self.info.keys():
            medium_width_x = float(self.info.get("MEDIUM WIDTH X"))
            cell_width_x = medium_width_x / cells_x
        else:
            cell_width_x = float(self.info.get("CELL WIDTH X"))
        self.cell_width_x = cell_width_x
        angles_x = int(self.info.get("ANGLES X"))
        self.medium_obj = MediumX(cells_x, cell_width_x, angles_x, xbounds)
        self.medium_obj.add_external_source(self.info.get("EXTERNAL SOURCE").lower())

    def _generate_materials_obj(self):
        self.materials_obj = Materials(self.info.get("MATERIAL").split("\n"), \
                                   int(self.info.get("ENERGY GROUPS")), 
                                   None)
        ps_locs = self.info.get("POINT SOURCE LOCATION").split("\n")
        ps_names = self.info.get("POINT SOURCE NAME").split("\n")
        for loc, name in zip(ps_locs, ps_names):
            try:
                self.materials_obj.add_point_source(name, int(loc), \
                                                    self.medium_obj.mu_x)
            except ValueError:
                if loc == "right-edge":
                    edge = int(self.info.get("CELLS X"))
                    self.materials_obj.add_point_source(name, edge, \
                                    self.medium_obj.mu_x)
        locations = []
        point_source = []
        for value in self.materials_obj.p_sources.values():
            locations.append(value[0])
            point_source.append(value[1])
        self.point_source_locs = np.array(locations)
        self.point_sources = np.array(point_source)

    def _generate_plot_angular(self, angle):
        ...

    def _generate_plot_rate(self):
        time_x = int(self.info.get("TIME STEPS"))
        time_size = float(self.info.get("TIME STEP SIZE"))
        tspace = np.linspace(0, time_x*time_size, time_x, endpoint=False)
        scatter_rate = self.fission_rate(self.xs_scatter)
        fission_rate = self.fission_rate(self.xs_fission)
        fig, ax = plt.subplots()
        ax.plot(xspace, scatter_rate, c="b", alpha=0.8)
        ax.set_title("Scatter Rate")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Scatter Rate (s$^{-1}$)")
        fig, ax = plt.subplots()
        ax.plot(xspace, fission_rate, c="r", alpha=0.8)
        ax.set_title("Fission Rate")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Fission Rate (s$^{-1}$)")

    def _generate_plot_rate_density(self):
        cells_x = int(self.info.get("CELLS X"))
        xspace = np.linspace(0, cells_x * self.cell_width_x, cells_x+1)
        xspace = 0.5*(xspace[:-1] + xspace[1:])
        scatter_rate = self.fission_rate_density(self.xs_scatter)
        fission_rate = self.fission_rate_density(self.xs_fission)
        fig, ax = plt.subplots()
        ax.plot(xspace, scatter_rate, c="b", alpha=0.8)
        ax.set_title("Scatter Rate Density")
        ax.set_xlabel("Location (cm)")
        ax.set_ylabel(r"Scatter Rate Density (cc$^{-1}$ s$^{-1}$)")
        fig, ax = plt.subplots()
        ax.plot(xspace, fission_rate, c="r", alpha=0.8)
        ax.set_title("Fission Rate Density")
        ax.set_xlabel("Location (cm)")
        ax.set_ylabel(r"Fission Rate Density (cc$^{-1}$ s$^{-1}$)")

    def _generate_plot_scalar(self):
        ...