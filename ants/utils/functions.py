########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from ants import constants as const

import numpy as np

def angles_x_calc(angles, bc):
    angle_x, angle_w = np.polynomial.legendre.leggauss(angles)
    angle_w /= np.sum(angle_w)
    # if bc == [1, 0]:
    #     return angle_x[angle_x < 0], angle_w[angle_x < 0]
    # elif bc == [0, 1]:
    #     return angle_x[angle_x > 0], angle_w[angle_x > 0]
    return angle_x, angle_w

def coarsen_groups_calc(fine, coarse):
    """  Get the indices for resizing matrices
    Arguments:
        fine: larger energy group size, int
        coarse: coarseer energy group size, int
    Returns:
        array of indicies of length (coarse + 1) """
    index = np.ones((coarse)) * int(fine / coarse)
    index[np.linspace(0, coarse-1, fine % coarse, dtype=np.int32)] += 1
    assert (index.sum() == fine)
    return np.cumsum(np.insert(index, 0, 0), dtype=np.int32)

def velocity_calc(groups, energy_edges, index):
    """ Convert energy edges to speed at cell centers, Relative Physics
    Arguments:
        groups: Number of energy groups
        energy_edges: energy grid bounds
        index: indices of cell edges, if len(energy_edges) != (groups + 1), 
                used for collapsing
    Returns:
        speeds at cell centers (cm/s)   """
    energy_edges = np.asarray(energy_edges)
    return np.ones((groups))
    energy_centers = 0.5 * (energy_edges[1:] + energy_edges[:-1])
    gamma = (const.EV_TO_JOULES * energy_centers) / \
            (const.MASS_NEUTRON * const.LIGHT_SPEED**2) + 1
    velocity = const.LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100
    if len(energy_centers) == groups:
        return velocity
    collapsed = []
    for left, right in zip(index[:-1], index[1:]):
        collapsed.append(np.mean(velocity[left:right]))
    return np.array(collapsed)

def xs_total_coarsen(xs_total_u, energy_edges_u, index):
    xs_total_c = []
    energy_edges_u = np.asarray(energy_edges_u)
    delta_u = np.diff(energy_edges_u)
    delta_c = np.diff(energy_edges_u[index])
    for mat in range(len(xs_total_u)):
        one_mat = xs_total_u[mat] * delta_u
        collapsed = [np.sum(one_mat[aa1:aa2]) for aa1, aa2 \
                        in zip(index[:-1], index[1:])]
        xs_total_c.append(np.array(collapsed) / delta_c)
    return np.array(xs_total_c)

def xs_scatter_coarsen(xs_scatter_u, energy_edges_u, index):
    xs_scatter_c = []
    energy_edges_u = np.asarray(energy_edges_u)
    delta_u = np.diff(energy_edges_u)
    delta_c = np.diff(energy_edges_u[index])
    for mat in range(len(xs_scatter_u)):
        one_mat = xs_scatter_u[mat] * delta_u
        collapsed = [[np.sum(one_mat[aa1:aa2, bb1:bb2]) for bb1, bb2 \
                        in zip(index[:-1], index[1:])] for aa1, aa2 \
                        in zip(index[:-1], index[1:])]
        xs_scatter_c.append(np.array(collapsed) / delta_c)
    return np.array(xs_scatter_c)    

def index_collided_calc(fine, index):
    # Which collided group the uncollided is a part of
    index_collided = np.zeros((fine), dtype=np.int32)
    splits = [slice(ii,jj) for ii,jj in zip(index[:-1],index[1:])]
    for count, split in enumerate(splits):
        index_collided[split] = count
    return index_collided

# def hybrid_factor_calc(fine, coarse, energy_edges_u, index):
#     print(fine, coarse)
#     energy_edges_u = np.asarray(energy_edges_u)
#     print(index)

#     factor = np.diff(energy_edges_u)
#     delta_c = np.diff(energy_edges_u[index])
#     splits = [slice(ii,jj) for ii,jj in zip(index[:-1],index[1:])]
#     for count, split in enumerate(splits):
#         for ii in range(split.start, split.stop):
#             factor[ii] /= delta_c[count]
#     return factor

def hybrid_factor_calc(fine, coarse, delta_u, delta_c, index):
    factor = delta_u.copy()
    splits = [slice(ii,jj) for ii,jj in zip(index[:-1],index[1:])]
    for count, split in enumerate(splits):
        for ii in range(split.start, split.stop):
            factor[ii] /= delta_c[count]
    return factor

def index_uncollided_calc(coarse, index):
    # Number of uncollided groups per uncollided group
    index_uncollided = np.zeros((coarse+1), dtype=np.int32)
    splits = [slice(ii,jj) for ii,jj in zip(index[:-1],index[1:])]
    for count, split in enumerate(splits):
        index_uncollided[count+1] = split.stop
    return index_uncollided

def energy_widths(energy_edges, index):
    # Return delta_u, delta_c
    energy_edges = np.asarray(energy_edges)
    return np.diff(energy_edges), np.diff(energy_edges[index])