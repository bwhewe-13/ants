########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np

def index_generator(full, reduced):
    grid = np.ones((reduced)) * int(full / reduced)
    grid[np.linspace(0, reduced - 1, full % reduced, dtype=int)] += 1
    assert (grid.sum() == full)
    return np.cumsum(np.insert(grid, 0, 0), dtype=int)

def create_slices(array):
    # Convert Medium Map to List of Slices
    idx = np.where(array[:-1] != array[1:])[0]
    idx = np.append(idx, len(array)-1)
    splits = []
    start = 0
    for ii in idx:
        splits.append(slice(start, ii+1))
        start = ii + 1
    return splits

def matrix_reduction(matrix, idx):
    energy_groups = len(idx) - 1
    return [[np.sum(matrix[idx[ii]:idx[ii+1],idx[jj]:idx[jj+1]]) \
        for jj in range(energy_groups)] for ii in range(energy_groups)]

def vector_reduction(vector, idx):
    energy_groups = len(idx) - 1
    return [sum(vector[idx[ii]:idx[ii+1]]) for ii in range(energy_groups)]