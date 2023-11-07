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


def solution_ss_01(x, angle_x):
    """ One material, single direction """
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        if mu > 0:
            flux[:,n] = 1.
        else:
            flux[:,n] = 1 - np.exp((1 - x) / mu)
    return flux


def solution_ss_02(x, angle_x):
    """ One material, angular dependent"""
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        if mu > 0:
            flux[:,n] = 0.5 + 0.5 * np.exp(-x / mu)
        else:
            flux[:,n] = 0.5 - 0.5 * np.exp((1 - x) / mu)
    return flux


def solution_ss_03(x, angle_x):
    """ One material, angular dependent, with source"""
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[:,n] = 0.5 + 0.25 * x**2 * np.exp(mu)
    return flux


def solution_ss_04(x, angle_x):
    """ Two materials, angular independent """
    length_x = 2
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[x <= 1,n] = -2 * x[x <= 1]**2 + 2 * length_x * x[x <= 1]
        flux[x > 1,n] = 0.25 * x[x > 1] - 0.125 * length_x + 0.5 * length_x**2
    return flux


def solution_ss_05(x, angle_x):
    """ Two materials, angular dependent """
    length_x = 2
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[x <= 1,n] = -2 * np.exp(mu) * x[x <= 1]**2 + 2 * length_x**2 * x[x <= 1]
        flux[x > 1,n] = length_x * np.exp(mu) * x[x > 1] + length_x**2 * (length_x - np.exp(mu))
    return flux


def solution_td_01(x, angle_x, edges_t):
    flux = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            flux[cc,:,nn,0] = (-x) * (x - 2) * np.sin(x - 0.1 * tt) + 2
    return flux


def solution_td_02(x, angle_x, edges_t):
    flux = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            flux[cc,:,nn,0] = 1 + np.sin(x - 0.5 * tt) + np.cos(mu)
    return flux