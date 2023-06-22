########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Two Dimensional Solutions for different problems using the Method of
# Manufactured Solutions
#
########################################################################

# import ants

import numpy as np

def solution_mms_01(centers_x, centers_y, angle_x, angle_y):
    """ One material, no scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y)
    flux = np.zeros(x.shape + (angles, 1))
    for n, mu in enumerate(angle_x):
        # Add x direction
        if mu > 0.0:
            flux[:,:,n,0] = 0.5 + np.exp(-x / mu)
        else:
            flux[:,:,n,0] = 0.5 + np.exp((1 - x) / mu)
    return flux

def solution_mms_02(centers_x, centers_y, angle_x, angle_y):
    """ One material, no scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y)
    flux = np.ones(x.shape + (angles,1))
    for n, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        # Add x direction
        if mu > 0.0:
            flux[:,:,n,0] += 0.5 * np.exp(-x / mu)
        else:
            flux[:,:,n,0] += 0.5 * np.exp((1 - x) / mu)
        # Add y direction
        if eta > 0.0:
            flux[:,:,n,0] += 0.5 * np.exp(-y / eta)
        else:
            flux[:,:,n,0] += 0.5 * np.exp((1 - y) / eta)
    return flux

def solution_mms_03(centers_x, centers_y, angle_x, angle_y):
    """ One material, scattering """
    angles = angle_x.shape[0]
    y, x = np.meshgrid(centers_x, centers_y)
    flux = np.zeros(x.shape + (angles,1))
    for n, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        flux[:,:,n,0] = np.exp(mu) + np.exp(eta)
    return flux

def solution_mms_04(centers_x, centers_y, angle_x, angle_y):
    """ One material, scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y)
    flux = np.zeros(x.shape + (angles,1))
    for n, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        flux[:,:,n,0] = 1 + 0.1 * x**2 * np.exp(mu) + 0.1 * y**2 * np.exp(eta)
    return flux