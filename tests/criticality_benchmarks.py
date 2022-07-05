########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Benchmarks from "Analytical Benchmark Test Set for Criticality Code 
# Verification" from Los Alamos National Lab
#
########################################################################

import numpy as np


def angles(boundary):
    angles = 16
    mu, angle_weight = np.polynomial.legendre.leggauss(angles)
    angle_weight /= np.sum(angle_weight)
    if boundary == "reflected":
        angles = int(0.5*angles)
        mu = mu[angles:]
        angle_weight = angle_weight[angles:]
    return mu, angle_weight

def normalize_phi(phi, geometry, boundary):
    normalize_phi = []
    phi = phi.flatten()
    cells = phi.shape[0]
    if geometry == "sphere":
        phi /= phi[0]
        normalize_phi.append(phi[int(cells*0.25)])
        normalize_phi.append(phi[int(cells*0.5)])
        normalize_phi.append(phi[int(cells*0.75)])
        normalize_phi.append(phi[-1])
    elif geometry == "slab":
        if boundary == "vacuum":
            phi /= phi[int(cells*0.5)]
            normalize_phi.append(phi[int(cells*0.375)])
            normalize_phi.append(phi[int(cells*0.25)])
            normalize_phi.append(phi[int(cells*0.125)])
            normalize_phi.append(phi[0])
        elif boundary == "reflected":
            phi /= phi[-1]
            normalize_phi.append(phi[int(cells*0.75)])
            normalize_phi.append(phi[int(cells*0.5)])
            normalize_phi.append(phi[int(cells*0.25)])
            normalize_phi.append(phi[0])
    return np.array(normalize_phi)

def create_param(geometry, boundary):
    if geometry == "sphere":
        return np.array([2, 2, 0, 1, 1, 0, 1, 1, 1], dtype=np.int32)
    elif geometry == "slab":
        if boundary == "reflected":
            return np.array([1, 2, 1, 1, 1, 0, 1, 1, 1], dtype=np.int32)
        elif boundary == "vacuum":
            return np.array([1, 2, 0, 1, 1, 0, 1, 1, 1], dtype=np.int32)

class OneGroup:
    
    def __init__(self, geometry, boundary, cells=200):
        self.geometry = geometry.lower()
        self.boundary = boundary.lower()
        self.cells = cells

    def plutonium_01a(self):
        self.xs_total = np.array([[0.32640]])
        self.xs_scatter = np.array([[[0.225216]]])
        self.xs_fission = np.array([[[3.24*0.0816]]])
        self.params = create_param(self.geometry, self.boundary)
        if (self.geometry == "slab") and (self.boundary == "reflected"):
            medium_width = 1.853722
        elif (self.geometry == "slab") and (self.boundary == "vacuum"):
            medium_width = 1.853722 * 2
            self.cells *= 2
        self.medium_map = np.zeros((self.cells), dtype=np.int32)
        self.cell_width = medium_width / self.cells

    def plutonium_01b(self):
        self.xs_total = np.array([[0.32640]])
        self.xs_scatter = np.array([[[0.225216]]])
        self.xs_fission = np.array([[[2.84*0.0816]]])
        self.params = create_param(self.geometry, self.boundary)
        if self.geometry == "sphere":
            medium_width = 6.082547
            self.flux_scale = np.array([0.93538006, 0.75575352, 0.49884364, 0.19222603])
        elif self.geometry == "slab":
            self.flux_scale =  np.array([0.9701734, 0.8810540, 0.7318131, 0.4902592])
            if self.boundary == "reflected":
                medium_width = 2.256751
            elif self.boundary == "vacuum":
                medium_width = 2.256751 * 2
                self.cells *= 2
        self.medium_map = np.zeros((self.cells), dtype=np.int32)
        self.cell_width = medium_width / self.cells

    def uranium_01a(self):
        self.xs_total = np.array([[0.32640]])
        self.xs_scatter = np.array([[[0.248064]]])
        self.xs_fission = np.array([[[2.70*0.065280]]])
        self.params = create_param(self.geometry, self.boundary)
        if self.geometry == "sphere":
            medium_width = 7.428998
            self.flux_scale = np.array([0.93244907, 0.74553332, 0.48095413, 0.17177706])
        elif self.geometry == "slab":
            self.flux_scale = np.array([0.9669506, 0.8686259, 0.7055218, 0.4461912])
            if self.boundary == "reflected":
                medium_width = 2.872934
            elif self.boundary == "vacuum":
                medium_width = 2.872934 * 2
                self.cells *= 2
        self.medium_map = np.zeros((self.cells), dtype=np.int32)
        self.cell_width = medium_width / self.cells

    def heavy_water_01a(self):
        self.cells = 300
        self.xs_total = np.array([[0.54628]])
        self.xs_scatter = np.array([[[0.464338]]])
        self.xs_fission = np.array([[[1.70*0.054628]]])
        self.params = create_param(self.geometry, self.boundary)
        if self.geometry == "sphere":
            medium_width = 22.017156
            self.flux_scale = np.array([0.91063756, 0.67099621, 0.35561622, 0.04678614])
        elif self.geometry == "slab":
            self.flux_scale = np.array([0.93945236, 0.76504084, 0.49690627, 0.13893858])
            if self.boundary == "reflected":
                medium_width = 10.371065
            elif self.boundary == "vacuum":
                medium_width = 10.371065 * 2
                self.cells *= 2
        self.medium_map = np.zeros((self.cells), dtype=np.int32)
        self.cell_width = medium_width / self.cells

    def uranium_reactor_01a(self):
        self.xs_total = np.array([[0.407407]])
        self.xs_scatter = np.array([[[0.328042]]])
        self.xs_fission = np.array([[[2.50*0.06922744]]])
        self.params = create_param(self.geometry, self.boundary)
        self.medium_map = np.zeros((self.cells), dtype=np.int32)
        medium_width = 250
        self.cell_width = medium_width / self.cells
        self.k_infinite = 2.1806667