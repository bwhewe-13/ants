########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Functions for running hybrid problems
#
########################################################################

import numpy as np

from ants.constants import *

########################################################################
# Coarsening Arrays for Hybrid Methods
########################################################################

def energy_coarse_index(fine, coarse):
    """  Get the indices for resizing matrices
    Arguments:
        fine (int): larger energy group size
        coarse (int): coarseer energy group size
    Returns:
        array of indicies (int [coarse + 1])
    """
    index = np.ones((coarse)) * int(fine / coarse)
    index[np.linspace(0, coarse-1, fine % coarse, dtype=np.int32)] += 1
    assert (index.sum() == fine)
    return np.cumsum(np.insert(index, 0, 0), dtype=np.int32)


def coarsen_materials(xs_total, xs_scatter, xs_fission, edges_g, \
        edges_gidx):
    """ Coarsen (materials x groups) arrays to (materials x groups')
    Arguments:
        xs_total (float [materials x groups]): Total cross section
        xs_scatter (float [materials x groups x groups]): scatter cross section
        xs_fission (float [materials x groups x groups]): fission cross section
        edges_g (float [groups + 1]): Energy group bounds
        edges_gidx (int [groups' + 1]): Index of energy group bounds for
                                        new energy grid
    Returns:
        coarse_total (float [materials x groups']): Coarsened total cross section
        coarse_scatter (float [materials x groups' x groups']): Coarsened
                    scatter cross section
        coarse_fission (float [materials x groups' x groups']): Coarsened
                    fission cross section
    """
    coarse_total = _xs_vector_coarsen(xs_total, edges_g, edges_gidx)
    coarse_scatter = _xs_matrix_coarsen(xs_scatter, edges_g, edges_gidx)
    coarse_fission = _xs_matrix_coarsen(xs_fission, edges_g, edges_gidx)
    return coarse_total, coarse_scatter, coarse_fission


def _xs_vector_coarsen(vector, edges_g, edges_gidx):
    """ Coarsen (materials x groups) arrays to (materials x groups')
    Arguments:
        vector (float [materials x groups]): Array to coarsen
        edges_g (float [groups + 1]): Energy group bounds
        edges_gidx (int [groups' + 1]): Index of energy group bounds for
                                        new energy grid
    Returns:
        coarse (float [materials x groups']): Coarsened array
    """
    materials = vector.shape[0]
    groups_coarse = edges_gidx.shape[0] - 1
    # Create coarsened array
    coarse = np.zeros((materials, groups_coarse))
    # Create energy bin widths
    delta_fine = np.diff(edges_g)
    delta_coarse = np.diff(np.asarray(edges_g)[edges_gidx])
    # Condition vector with energy bin width
    fine = np.asarray(vector) * delta_fine[None,:]
    # Loop over all materials
    for mat in range(materials):
        coarse[mat] = [np.sum(fine[mat,gg1:gg2]) for gg1, gg2 \
                        in zip(edges_gidx[:-1], edges_gidx[1:])]
    # Coarsen
    coarse /= delta_coarse[None,:]
    return coarse


def _xs_matrix_coarsen(matrix, edges_g, edges_gidx):
    """ Coarsen (materials x groups x groups) arrays to
        (materials x groups' x groups')
    Arguments:
        matrix (float [materials x groups x groups]): Array to coarsen
        edges_g (float [groups + 1]): Energy group bounds
        edges_gidx (int [groups' + 1]): Index of energy group bounds for
                                        new energy grid
    Returns:
        coarse (float [materials x groups' x groups']): Coarsened array
    """
    materials = matrix.shape[0]
    groups_coarse = edges_gidx.shape[0] - 1
    # Create coarsened array
    coarse = np.zeros((materials, groups_coarse, groups_coarse))
    # Create energy bin widths
    delta_fine = np.diff(edges_g)
    delta_coarse = np.diff(np.asarray(edges_g)[edges_gidx])
    fine = np.asarray(matrix) * delta_fine[None,:]
    for mat in range(materials):
        coarse[mat] = [[np.sum(fine[mat][aa1:aa2, bb1:bb2]) for bb1, bb2 \
                        in zip(edges_gidx[:-1], edges_gidx[1:])] for aa1, aa2 \
                        in zip(edges_gidx[:-1], edges_gidx[1:])]
    # Coarsen
    coarse /= delta_coarse[None,:]
    return coarse


def coarsen_velocity(vector, edges_gidx):
    """ Coarsen (groups) vector to (groups')
    Arguments:
        vector (float [groups]): Array to coarsen
        edges_gidx (int [groups' + 1]): Index of energy group bounds for
                                        new energy grid
    Returns:
        coarse (float [groups']): Coarsened vector
    """
    groups_coarse = edges_gidx.shape[0] - 1
    # Create coarsened array
    coarse = np.zeros((groups_coarse))
    for gg, (gg1, gg2) in enumerate(zip(edges_gidx[:-1], edges_gidx[1:])):
        coarse[gg] = np.mean(vector[gg1:gg2])
    return coarse

########################################################################
# Indexing for Hybrid Methods
########################################################################

def indexing(groups_fine, groups_coarse, edges_g, edges_gidx):
    """ Calculate the variables needed for refining and coarsening fluxes
    Arguments:
        groups_fine (int): Number of fine energy groups
        groups_coarse (int): Number of coarse energy groups where
                            groups_fine >= groups_coarse
        edges_g (float [groups_fine + 1]): Energy group bounds for fine grid
        edges_gidx (int [groups_coarse + 1]): Index of energy group
                            bounds for coarse energy grid
    Returns:
        coarse_idx (int [groups_fine]): Coarse group mapping
        fine_idx (int [groups_coarse + 1]): Location of edges between the
                            coarse and fine energy grids
        factor (float [groups_fine]): Fine energy bin width / coarse energy
                            bin width for specific location
    """
    fine_idx = _uncollided_index(groups_coarse, edges_gidx)
    coarse_idx = _collided_index(groups_fine, edges_gidx)
    # Convert from memoryview
    edges_g = np.asarray(edges_g)
    # Calculate energy bin widths
    delta_fine = np.diff(edges_g)
    delta_coarse = np.diff(edges_g[edges_gidx])
    factor = _hybrid_factor(delta_fine, delta_coarse, edges_gidx)
    return fine_idx, coarse_idx, factor


def _collided_index(groups_fine, edges_gidx):
    """ Calculate which coarse group a fine energy group is a part of,
    i.e.:
    fine grid:      |---g1---|--g2--|---g3---|--g4--|
    coarse grid:    |-------g1------|-------g2------|
    results in coarse_idx = [0, 0, 1, 1]
    Arguments:
        groups_fine (int): Number of fine energy groups
        edges_gidx (int [groups_coarse + 1]): Index of energy group
                            bounds for coarse energy grid
    Returns:
        coarse_idx (int [groups_fine]): Coarse group mapping
    """
    coarse_idx = np.zeros((groups_fine), dtype=np.int32)
    splits = [slice(ii,jj) for ii, jj in zip(edges_gidx[:-1], edges_gidx[1:])]
    for count, split in enumerate(splits):
        coarse_idx[split] = count
    return coarse_idx


def _uncollided_index(groups_coarse, edges_gidx):
    """ Calculate the location of edges between the coarse and fine
    energy grids, i.e.:
    edge:           0        1      2        3      4
    fine grid:      |---g1---|--g2--|---g3---|--g4--|
    coarse grid:    |-------g1------|-------g2------|
    results in fine_idx = [0, 2, 4]
    Arguments:
        groups_coarse (int): Number of coarse energy groups where
                            groups_fine >= groups_coarse
        edges_gidx (int [groups_coarse + 1]): Index of energy group
                            bounds for coarse energy grid
    Returns:
        fine_idx (int [groups_coarse + 1]): Location of edges between the
                            coarse and fine energy grids
    """
    fine_idx = np.zeros((groups_coarse + 1), dtype=np.int32)
    splits = [slice(ii,jj) for ii, jj in zip(edges_gidx[:-1], edges_gidx[1:])]
    for count, split in enumerate(splits):
        fine_idx[count+1] = split.stop
    return fine_idx


def _hybrid_factor(delta_fine, delta_coarse, edges_gidx):
    """ Calculate the fine energy bin width per coarse energy bin width
    for specific location, i.e.:
    location (eV):  0        3      5        8      10
    fine grid:      |---g1---|--g2--|---g3---|--g4--|
    coarse grid:    |-------g1------|-------g2------|
    results in factor = [3 / 5, 2 / 5, 3 / 5, 2 / 5] with widths [3, 2, 2, 3]
    for the fine groups and [5, 5] for the coarse groups
    Arguments:
        delta_fine (float [groups_fine]): Energy group width for fine grid
        delta_coarse (float [groups_fine]): Energy group width for coarse grid
        edges_gidx (int [groups_coarse + 1]): Index of energy group
                            bounds for coarse energy grid
    Returns:
        factor (float [groups_fine]): Fine energy bin width / coarse energy
                            bin width for specific location
    """
    factor = delta_fine.copy()
    splits = [slice(ii,jj) for ii, jj in zip(edges_gidx[:-1], edges_gidx[1:])]
    for count, split in enumerate(splits):
        for ii in range(split.start, split.stop):
            factor[ii] /= delta_coarse[count]
    return factor
