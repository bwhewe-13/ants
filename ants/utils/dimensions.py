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

def half_spatial_grid(medium_map, cell_widths, error, epsilon=0.1):
    cells = medium_map.shape[0]
    add_cells = cells + error[error > epsilon].shape[0]
    add_medium_map = np.ones((add_cells)) * -1
    add_cell_widths = np.ones((add_cells)) * -1
    idx = 0
    for cell in range(cells):
        if error[cell] > epsilon:
            add_medium_map[cell+idx] = medium_map[cell]
            add_medium_map[cell+idx+1] = medium_map[cell]
            add_cell_widths[cell+idx] = 0.5 * cell_widths[cell]
            add_cell_widths[cell+idx+1] = 0.5 * cell_widths[cell]
            idx += 1
        else:
            add_medium_map[cell+idx] = medium_map[cell]
            add_cell_widths[cell+idx] = cell_widths[cell]
    return add_medium_map, add_cell_widths

def coarsen_flux(fine_flux, fine_edges, coarse_edges):
    coarse_flux = np.zeros((coarse_edges.shape[0] - 1))
    for cell, (low, high) in enumerate(zip(coarse_edges[:-1], coarse_edges[1:])):
        zone = fine_flux[np.logical_and(fine_edges >= low, fine_edges < high)[:-1]]
        coarse_flux[cell] = np.sum(zone) * 1 / zone.shape[0]
    return coarse_flux