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

def solution_mms_01(x, angle_x):
    """ One material, single direction """
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        if mu > 0:
            flux[:,n] = 1.
        else:
            flux[:,n] = 1 - np.exp((1 - x) / mu)
    return flux

def solution_mms_02(x, angle_x):
    """ One material, angular dependent"""
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        if mu > 0:
            flux[:,n] = 0.5 + 0.5 * np.exp(-x / mu)
        else:
            flux[:,n] = 0.5 - 0.5 * np.exp((1 - x) / mu)
    return flux

def solution_mms_03(x, angle_x):
    """ One material, angular dependent, with source"""
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[:,n] = 0.25 + 0.25 * x**2 * np.exp(mu)
    return flux

def solution_mms_04(x, angle_x):
    """ Two materials, angular independent """
    width = 2
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[x < 1,n] = -2 * x[x < 1]**2 + 2 * width * x[x < 1]
        flux[x > 1,n] = 0.25 * x[x > 1] - 0.125 * width + 0.5 * width**2
    return flux

def solution_mms_05(x, angle_x):
    """ Two materials, angular dependent """
    width = 2
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[x < 1,n] = -2 * np.exp(mu) * x[x < 1]**2 + 2 * width**2 * x[x < 1]
        flux[x > 1,n] = width * np.exp(mu) * x[x > 1] + width**2 * (width - np.exp(mu))
    return flux
