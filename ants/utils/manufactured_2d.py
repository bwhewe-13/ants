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

import numpy as np


def solution_ss_01(centers_x, centers_y, angle_x, angle_y):
    """ One material, no scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y, indexing="ij")
    flux = np.zeros(x.shape + (angles, 1))
    for n, mu in enumerate(angle_x):
        # Add x direction
        if mu > 0.0:
            flux[:,:,n,0] = 0.5 + np.exp(-x / mu)
        else:
            flux[:,:,n,0] = 0.5 + np.exp((1 - x) / mu)
    return flux


def solution_ss_02(centers_x, centers_y, angle_x, angle_y):
    """ One material, no scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y, indexing="ij")
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


def solution_ss_03(centers_x, centers_y, angle_x, angle_y):
    """ One material, scattering """
    angles = angle_x.shape[0]
    y, x = np.meshgrid(centers_x, centers_y, indexing="ij")
    flux = np.zeros(x.shape + (angles,1))
    for n, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        flux[:,:,n,0] = np.exp(mu) + np.exp(eta)
    return flux


def solution_ss_04(centers_x, centers_y, angle_x, angle_y):
    """ One material, scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y, indexing="ij")
    flux = np.zeros(x.shape + (angles,1))
    for n, (mu, eta) in enumerate(zip(angle_x, angle_y)):
        flux[:,:,n,0] = 1 + 0.1 * x**2 * np.exp(mu) + 0.1 * y**2 * np.exp(eta)
    return flux


def solution_td_01(x, y, angle_x, angle_y, edges_t):
    flux = np.zeros((edges_t.shape[0], x.shape[0], y.shape[0], angle_x.shape[0], 1))
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    for cc, tt in enumerate(edges_t):
        for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
            flux[cc,...,nn,0] = 2 + np.sin(mesh_x * mesh_y - 0.1 * tt) \
                    * (-mesh_x) * (mesh_x - 2) * (-mesh_y) * (mesh_y - 2)

    return flux


def solution_td_02(x, y, angle_x, angle_y, edges_t):
    flux = np.zeros((edges_t.shape[0], x.shape[0], y.shape[0], angle_x.shape[0], 1))
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    for cc, tt in enumerate(edges_t):
        for nn, (mu, eta) in enumerate(zip(angle_x, angle_y)):
            flux[cc,...,nn,0] = 1 + np.sin(mesh_x - 0.5 * tt) + np.cos(mu) \
                                  + np.cos(mesh_y - 0.25 * tt) + np.sin(eta)
    return flux