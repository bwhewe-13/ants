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
    new_medium_map = []
    new_cell_widths = []
    for cell in range(medium_map.shape[0]):
        if error[cell] > epsilon:
            new_medium_map.append(medium_map[cell])
            new_medium_map.append(medium_map[cell])
            new_cell_widths.append(0.5 * cell_widths[cell])
            new_cell_widths.append(0.5 * cell_widths[cell])
        else:
            new_medium_map.append(medium_map[cell])
            new_cell_widths.append(cell_widths[cell])
    new_medium_map, new_cell_widths = smooth_spatial_grid( \
                    np.array(new_medium_map), np.array(new_cell_widths))
    return new_medium_map, new_cell_widths

def smooth_spatial_grid(medium_map, widths):
    idx = 0
    for cell in range(len(widths)):
        if (int(widths[cell+idx] / widths[cell-1+idx]) > 2 and cell > 0) \
            or (cell < (len(widths)-idx-1) \
                and int(widths[cell+idx] / widths[cell+1+idx]) > 2):
            widths[cell+idx] = 0.5 * widths[cell+idx]
            widths = np.insert(widths, cell+idx, widths[cell+idx])
            medium_map = np.insert(medium_map, cell+idx, medium_map[cell+idx])
            idx += 1
    return medium_map, widths

def coarsen_flux(fine_flux, fine_edges, coarse_edges, summation=False):
    coarse_flux = np.zeros((coarse_edges.shape[0] - 1))
    count = 0
    for ii in range(len(coarse_edges) - 1):
        idx = np.argwhere((fine_edges < coarse_edges[ii+1]) & (fine_edges >= coarse_edges[ii]))
        count += len(idx)
        if summation:
            coarse_flux[ii] = np.sum(fine_flux[idx]) #/ len(idx)
        else:
            coarse_flux[ii] = np.sum(fine_flux[idx]) / len(idx)
    assert count == len(fine_flux), "Not including all cells"
    return coarse_flux

def flux_edges(centers, direction, split=None, dtype="diamond"):
    # Calculate the flux edge from the center - full medium
    edges = np.zeros((len(centers)+1))
    if direction > 0: # Sweep from left to right
        if dtype == "step":
            edges[1:] = centers.copy()
        elif dtype == "diamond":
            for cell in range(1, len(centers) + 1):
                edges[cell] = 2 * centers[cell-1] - edges[cell-1]
    elif direction < 0: # Sweep from right to left
        if dtype == "step":
            edges[:-1] = centers.copy()
        elif dtype == "diamond":
            for cell in range(len(centers)-1, -1, -1):
                edges[cell] = 2 * centers[cell] - edges[cell+1]
    if split is None:
        return edges
    return edges[split]

def spatial_edges(centers, widths):
    # Calculate the spatial edge from the center
    edges = [centers[0] - 0.5 * widths[0]]
    for center, width in zip(centers, widths):
        edges.append(center + 0.5 * width)
    return np.array(edges)

def mesh_centers_edges(centers, edges):
    # Combine the cell edge and centers into an array
    both = np.zeros((len(edges) + len(centers)))
    both[::2] = edges.copy()
    both[1::2] = centers.copy()
    return both

def mesh_refinement(x, y):
    # Add points between each known value of x and y
    x_plus = np.zeros((len(x) * 2 - 1))
    x_plus[::2] = x.copy()
    y_plus = np.zeros(x_plus.shape)
    y_plus[::2] = y.copy()
    for pt in range(len(x) - 1):
        x_plus[2*pt+1] = 0.5 * (x[pt] + x[pt+1])
        y_plus[2*pt+1] = 0.5 * (y[pt] + y[pt+1])
    return x_plus, y_plus    