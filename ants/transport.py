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
        return np.array(locations), np.array(point_source)
    
    def _generate_cross_section(self):
        map_obj = os.path.join(self.problem.get("PATH","."), \
                               self.problem.get("MAP FILE"))
        map_obj = Mapper.load_map(map_obj)
        if int(self.problem.get("CELLS X")) != map_obj.cells_x:
            map_obj.adjust_widths(int(self.problem.get("CELLS X")))
        self.medium_map = map_obj.map_x.astype(int)
        reversed_key = {v: k for k, v in map_obj.map_key.items()}
        total = []
        scatter = []
        fission = []
        for position in range(len(map_obj.map_key)):
            map_material = reversed_key[position]
            total.append(self.materials_obj.data[map_material][0])
            scatter.append(self.materials_obj.data[map_material][1])
            fission.append(self.materials_obj.data[map_material][2])
        return np.array(total), np.array(scatter), np.array(fission)

    def run(self):
        point_source_locs, point_sources = self._generate_materials_obj()
        xs_total, xs_scatter, xs_fission = self._generate_cross_section()
        if self.problem.get("TIME DISCRETE") == "backward-euler":
            scalar, angular = backward_euler(self.medium_map, 
                                    xs_total, xs_scatter, xs_fission, \
                                    self.medium_obj.ex_source, \
                                    point_source_locs, point_sources, \
                                    self.medium_obj.spatial_coef_x, \
                                    self.medium_obj.weight, \
                                    self.materials_obj.velocity, \
                                    int(self.problem.get("TIME STEPS")), \
                                    float(self.problem.get("TIME STEP SIZE")), \
                                    spatial=self.problem.get("SPACE DISCRETE"))
        return scalar, angular

if __name__ == "__main__":

    input_file = 'borderless.inp'
    test = Transport(input_file)
    _, psi = test.run()
    mu_x = test.medium_obj.mu_x

    medium_width = 1.
    angles = 2

    cells_x = 20
    cell_width_x = medium_width / cells_x

    xspace = np.linspace(0, medium_width, cells_x+1)
    xspace = 0.5*(xspace[:-1] + xspace[1:])

    def ref01(mu, x):
        if mu > 0:
            return np.ones(x.shape)
        elif mu < 0:
            return 1 - np.exp((medium_width - x) / mu)

    # def ref02(mu, x):
    #     return 0.5 - 0.5 * np.exp((medium_width - x) / mu)

    for angle in range(2):
        fig, ax = plt.subplots()
        ax.plot(xspace, ref01(mu_x[angle], xspace), label='True', c='k', ls='--')
        ax.plot(xspace, psi[:,angle],'-o', label='Angle {}'.format(angle), alpha=0.6, c='r')
        ax.grid()
        ax.legend(loc=0)

    plt.show()