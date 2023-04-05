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
    flux = np.ones(x.shape + (angles,1))
    for n in range(angles):
        # Add x direction
        if angle_x[n] > 0.0:
            flux[:,:,n,0] = 0.5 + np.exp(-x / angle_x[n])
        else:
            flux[:,:,n,0] = 0.5 + np.exp((1 - x) / angle_x[n])
    return flux

def solution_mms_02(centers_x, centers_y, angle_x, angle_y):
    """ One material, no scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y)
    flux = np.ones(x.shape + (angles,1))
    for n in range(angles):
        # Add x direction
        if angle_x[n] > 0.0:
            flux[:,:,n,0] += 0.5 * np.exp(-x / angle_x[n])
        else:
            flux[:,:,n,0] += 0.5 * np.exp((1 - x) / angle_x[n])
        # Add y direction
        if angle_y[n] > 0.0:
            flux[:,:,n,0] += 0.5 * np.exp(-y / angle_y[n])
        else:
            flux[:,:,n,0] += 0.5 * np.exp((1 - y) / angle_y[n])
    return flux
#         # if (angle_x[n] > 0.0) and (angle_y[n] > 0.0):
#         #     flux[:,:,n,0] = 1 + 0.5 * np.exp(-x / angle_x[n]) + 0.5 * np.exp(-y / angle_y[n])
#         # elif (angle_x[n] < 0.0) and (angle_y[n] > 0.0):
#         #     flux[:,:,n,0] = 1 - 0.5 * np.exp((1 - x) / angle_x[n]) + 0.5 * np.exp(-y / angle_y[n])
#         # elif (angle_x[n] > 0.0) and (angle_y[n] < 0.0):
#         #     flux[:,:,n,0] = 1 + 0.5 * np.exp(-x / angle_x[n]) - 0.5 * np.exp((1 - y) / angle_y[n])
#         # elif (angle_x[n] < 0.0) and (angle_y[n] < 0.0):
#         #     flux[:,:,n,0] = 1 - 0.5 * np.exp((1 - x) / angle_x[n]) - 0.5 * np.exp((1 - y) / angle_y[n])
#         # else:
#         #     raise Exception("Angles must be even number")
#     return flux

def solution_mms_03(centers_x, centers_y, angle_x, angle_y):
    """ One material, scattering """
    angles = angle_x.shape[0]
    x, y = np.meshgrid(centers_x, centers_y)
    flux = np.ones(x.shape + (angles,1))
    for n in range(angles):
        flux[:,:,n,0] = np.cos(x * angle_x[n]) + np.cos(y * angle_y[n])
    return flux