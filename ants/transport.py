########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
#
########################################################################

from ants.medium import MediumX
from ants.materials import Materials
from ants.mapper import Mapper
from ants.fixed_source import backward_euler

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

    def _read_input(self):
        self.problem = {}
        with open(self.input_file, "r") as fp:
            for line in fp:
                if line[0] == "#":
                    continue
                if line[:2] == "\n":
                    continue
                key, value = line.split(":")
                if key in self.problem.keys() and key == "NOTE":
                    self.problem[key] += "\n" + value.strip().lower()
                elif key == "NOTE":
                    self.problem[key] = value.strip().lower() 
                elif key in self.problem.keys():
                    self.problem[key] += "\n" \
                              + "-".join(value.strip().lower().split())
                else:
                    self.problem[key] = \
                                "-".join(value.strip().lower().split())

    def _generate_boundaries(self):
        def boundary(string):
            if string == "vacuum":
                return 0
            elif string == "reflected":
                return 1
        lhs_x = boundary(self.problem.get("BOUNDARY (x = 0)"))
        rhs_x = boundary(self.problem.get("BOUNDARY (x = X)"))
        xbounds = np.array([lhs_x, rhs_x])
        return xbounds

    def _generate_medium_obj(self):
        xbounds = self._generate_boundaries()
        cells_x = int(self.problem.get("CELLS X"))
        if "MEDIUM WIDTH X" in self.problem.keys():
            medium_width_x = float(self.problem.get("MEDIUM WIDTH X"))
            cell_width_x = medium_width_x / cells_x
        else:
            cell_width_x = float(self.problem.get("CELL WIDTH X"))
        self.cell_width_x = cell_width_x
        angles_x = int(self.problem.get("ANGLES X DIRECTION"))
        self.medium_obj = MediumX(cells_x, cell_width_x, angles_x, xbounds)
        self.medium_obj.add_external_source(self.problem.get("EXTERNAL SOURCE"))

    def _generate_materials_obj(self):
        self.materials_obj = Materials(self.problem.get("MATERIAL").split("\n"), \
                                   int(self.problem.get("ENERGY GROUPS")), 
                                   None)
        ps_locs = self.problem.get("POINT SOURCE LOCATION").split("\n")
        ps_names = self.problem.get("POINT SOURCE NAME").split("\n")
        for loc, name in zip(ps_locs, ps_names):
            self.materials_obj.add_point_source(name, int(loc), \
                                                self.medium_obj.mu_x)
        locations = []
        point_source = []
        for value in self.materials_obj.p_sources.values():
            locations.append(value[0])
            point_source.append(value[1])
        self.point_source_locs = np.array(locations)
        self.point_sources = np.array(point_source)
    
    def _generate_cross_section(self):
        map_obj = os.path.join(self.problem.get("PATH","."), \
                               self.problem.get("MAP FILE"))
        map_obj = Mapper.load_map(map_obj)
        if int(self.problem.get("CELLS X")) != map_obj.cells_x:
            map_obj.adjust_widths(int(self.problem.get("CELLS X")))
        self.medium_map = map_obj.map_x.astype(int)
        self.medium_map_key = map_obj.map_key
        reversed_key = {v: k for k, v in map_obj.map_key.items()}
        total = []
        scatter = []
        fission = []
        for position in range(len(map_obj.map_key)):
            map_material = reversed_key[position]
            total.append(self.materials_obj.data[map_material][0])
            scatter.append(self.materials_obj.data[map_material][1])
            fission.append(self.materials_obj.data[map_material][2])
        self.xs_total = np.array(total)
        self.xs_scatter = np.array(scatter)
        self.xs_fission = np.array(fission)

    def run(self):
        if self.problem.get("TIME DISCRETE") == "backward-euler":
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
                                    int(self.problem.get("TIME STEPS")), \
                                    float(self.problem.get("TIME STEP SIZE")), \
                                    spatial=self.problem.get("SPACE DISCRETE"))
        if int(self.problem.get("TIME STEPS")) == 0:
            scalar = np.array(scalar[0])
        else:
            scalar = np.array(scalar)
        self.scalar = scalar
        self.angular = angular
        print(self.scalar.shape)
        print(self.angular.shape)
        return scalar, angular

    def run_save(self):
        scalar, angular = self.run()
        self.save()
        return scalar, angular

    def run_graph(self):
        scalar, angular = self.run()
        self.graph()
        return scalar, angular
        
    def run_graph_save(self):
        scalar, angular = self.run()
        self.save()
        self.graph()
        return scalar, angular

    def _generate_file_name(self):
        file_name = os.path.join(self.problem.get("PATH","."), \
                            self.problem.get("FILE NAME"))
        file_number = 0
        while os.path.exists(file_name + str(file_number) + ".npz"):
            file_number += 1
        return file_name + str(file_number)

    def save(self):
        dictionary = {}
        file_name = self._generate_file_name()
        # Neutron fluxes
        dictionary["flux-scalar"] = self.scalar
        dictionary["flux-angular"] = self.angular
        # Medium data
        dictionary["medium-map"] = self.medium_map
        dictionary["medium-key"] = self.map_obj.map_key
        # Cross sections
        dictionary["groups"] = int(self.problem.get("ENERGY GROUPS"))
        dictionary["xs-total"] = self.xs_total
        dictionary["xs-scatter"] = self.xs_scatter
        dictionary["xs-fission"] = self.xs_fission
        # Spatial Info
        dictionary["cells-x"] = int(self.problem.get("CELLS X"))
        dictionary["cell-width-x"] = self.cell_width_x
        dictionary["spatial-disc"] = self.problem.get("SPACE DISCRETE")
        # Angular Info
        dictionary["angles-x"] = int(self.problem.get("ANGLES X DIRECTION"))
        dictionary["boundary-x"] = [self.problem.get("BOUNDARY (x = 0)"),
                                    self.problem.get("BOUNDARY (x = X)")]
        # Time Info
        dictionary["time-steps"] = int(self.problem.get("TIME STEPS"))
        dictionary["time-step-size"] \
                             = float(self.problem.get("TIME STEP SIZE"))
        dictionary["time-disc"] = self.problem.get("TIME DISCRETE")
        # Extra
        dictionary["notes"] = self.problem.get("NOTE")
        np.savez(file_name, **dictionary)

    def change_param(self, name, value):
        if name in self.problem.keys():
            self.problem[name] = value
        else:
            raise KeyError("Not in Input File Keys\nAvailable Keys:\
                                    \n{}".format(self.problem.keys()))
        self._generate_medium_obj()
        self._generate_materials_obj()
        self._generate_cross_section()

    def fission_rate_density(self, cross_section):
        cell_data = np.zeros((int(self.problem.get("CELLS X"))))
        for cell, mat in enumerate(self.medium_map):
            cell_data[cell] = np.sum(self.scalar[cell] \
                                * np.sum(cross_section[mat],axis=1))
        return cell_data

    def fission_rate(self, cross_section):
        time_data = np.zeros((int(self.problem.get("TIME STEPS"))))
        for time_step in range(len(time_data)):
            time_data[time_step] = np.sum(self.cell_width_x * \
                               self.fission_rate_density(cross_section))
        return time_data

    def _generate_angular_plot(self, angle):
        ...

    def _generate_scalar_plot(self):
        ...

    def _generate_rate_density_plot(self):
        cells_x = int(self.problem.get("CELLS X"))
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

    def _generate_rate_plot(self):
        time_x = int(self.problem.get("TIME STEPS"))
        time_size = float(self.problem.get("TIME STEP SIZE"))
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

    def graph(self):
        if int(self.problem.get("TIME STEPS")) > 0:
            self._generate_rate_plot()
        else:
            self._generate_rate_density_plot()
        plt.show()

# if __name__ == "__main__":

#     input_file = 'borderless.inp'
#     test = Transport(input_file)
#     _, psi = test.run()
#     mu_x = test.medium_obj.mu_x

#     medium_width = 1.
#     angles = 2

#     cells_x = 20
#     cell_width_x = medium_width / cells_x

#     xspace = np.linspace(0, medium_width, cells_x+1)
#     xspace = 0.5*(xspace[:-1] + xspace[1:])

#     def ref01(mu, x):
#         if mu > 0:
#             return np.ones(x.shape)
#         elif mu < 0:
#             return 1 - np.exp((medium_width - x) / mu)

#     # def ref02(mu, x):
#     #     return 0.5 - 0.5 * np.exp((medium_width - x) / mu)

#     for angle in range(2):
#         fig, ax = plt.subplots()
#         ax.plot(xspace, ref01(mu_x[angle], xspace), label='True', c='k', ls='--')
#         ax.plot(xspace, psi[:,angle],'-o', label='Angle {}'.format(angle), alpha=0.6, c='r')
#         ax.grid()
#         ax.legend(loc=0)

#     plt.show()