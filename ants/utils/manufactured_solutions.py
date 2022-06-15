########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Solutions for the different Method of Manufactured Solutions
#
########################################################################

import numpy as np

def solution_one_material_01(xspace, angles_mu):
    angular_flux = np.zeros((len(xspace), len(angles_mu)))
    for n, angle in enumerate(angles_mu):
        if angle > 0:
            angular_flux[:,n] = 1
        else:
            angular_flux[:,n] = 1 - np.exp((1 - xspace)/angle)
    return angular_flux

def solution_one_material_02(xspace, angles_mu):
    angular_flux = np.zeros((len(xspace), len(angles_mu)))
    for n, angle in enumerate(angles_mu):
        if angle > 0:
            angular_flux[:,n] = 0.5 + 0.5 * np.exp(-xspace / angle)
        else:
            angular_flux[:,n] = 0.5 - 0.5 * np.exp((1 - xspace) / angle)
    return angular_flux

def solution_one_material_03(xspace, angles_mu):
    angular_flux = np.zeros((len(xspace), len(angles_mu)))
    for n, angle in enumerate(angles_mu):
        angular_flux[:,n] = 0.25 + 0.25 * xspace**2 * np.exp(angle)
    return angular_flux

def solution_two_material_01(xspace, angles_mu):
    angular_flux = np.zeros((len(xspace), len(angles_mu)))
    length = np.round(xspace[-1])
    for n, angle in enumerate(angles_mu):
        angular_flux[xspace < 1,n] = -2 * xspace[xspace < 1]**2 \
                                        + 2 * length * xspace[xspace < 1]
        angular_flux[xspace > 1,n] = 0.25 * xspace[xspace > 1] \
                                    - 0.125 * length + 0.5 * length**2
    return angular_flux

def solution_two_material_02(xspace, angles_mu):
    angular_flux = np.zeros((len(xspace), len(angles_mu)))
    length = np.round(xspace[-1])
    for n, angle in enumerate(angles_mu):
        angular_flux[xspace < 1,n] = -2 * np.exp(angle) * xspace[xspace < 1]**2 \
                                    + 2 * length**2 * xspace[xspace < 1]
        angular_flux[xspace > 1,n] = length * np.exp(angle) * xspace[xspace > 1] \
                                    + length**2 * (length - np.exp(angle))
    return angular_flux
