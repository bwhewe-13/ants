########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from ants.constants import *

import numpy as np

def index_generator(full, reduced):
    grid = np.ones((reduced)) * int(full / reduced)
    grid[np.linspace(0, reduced - 1, full % reduced, dtype=int)] += 1
    assert (grid.sum() == full)
    return np.cumsum(np.insert(grid, 0, 0), dtype=int)

def create_slices(array, double_count=False):
    # This is for cell edges
    coef_ = 2 if double_count else 1
    # Convert Medium Map to List of Slices
    idx = np.where(array[:-1] != array[1:])[0]
    idx = np.append(idx, len(array)-1)
    splits = []
    start = 0
    for ii in idx:
        splits.append(slice(start, ii + coef_))
        start = ii + 1
    return np.array(splits)

# def matrix_reduction(matrix, idx):
#     energy_groups = len(idx) - 1
#     return [[np.sum(matrix[idx[ii]:idx[ii+1],idx[jj]:idx[jj+1]]) \
#         for jj in range(energy_groups)] for ii in range(energy_groups)]

# def vector_reduction(vector, idx):
#     energy_groups = len(idx) - 1
#     return [sum(vector[idx[ii]:idx[ii+1]]) for ii in range(energy_groups)]

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

def coarsen_flux(fine_flux, fine_edges, coarse_edges):
    cells = coarse_edges.shape[0] - 1
    coarse_flux = np.zeros(((cells,) + fine_flux.shape[1:]))
    count = 0
    for ii in range(cells):
        idx = np.argwhere((fine_edges < coarse_edges[ii+1]) \
                        & (fine_edges >= coarse_edges[ii]))
        count += len(idx)
        coarse_flux[ii] = np.sum(fine_flux[idx.flatten()], axis=0) / len(idx)
    assert count == len(fine_flux), "Not including all cells"
    return coarse_flux

def flux_edges(center_flux, angle_x, boundary):
    # discretize 1: step, 2: diamond
    # Calculate the flux edge from the center
    edge_flux = np.zeros((center_flux.shape[0]+1, *center_flux.shape[1:]))
    # center flux is of shape (I x N x G)
    for gg in range(center_flux.shape[2]):
        for nn in range(center_flux.shape[1]):
            if angle_x[nn] > 0.0: # forward sweep
                edge_flux[0, nn, gg] = boundary[0,nn,gg]
                # edge_flux[1:, nn, gg] = center_flux[:,nn,gg]
                for ii in range(1, center_flux.shape[0]+1):
                    edge_flux[ii,nn,gg] = 2 * center_flux[ii-1,nn,gg] \
                                            - edge_flux[ii-1,nn,gg]
            elif angle_x[nn] < 0.0: # backward sweep
                edge_flux[-1, nn, gg] = boundary[1,nn,gg]
                # edge_flux[:-1, nn, gg] = center_flux[:,nn,gg]
                for ii in range(center_flux.shape[0]-1, -1, -1):
                    edge_flux[ii,nn,gg] = 2 * center_flux[ii,nn,gg] \
                                            - edge_flux[ii+1,nn,gg]
    return edge_flux

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


########################################################################
# Coarsening Arrays for Hybrid Methods
########################################################################

def calculate_coarse_edges(fine, coarse):
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

def xs_vector_coarsen(xs_total_u, energy_edges_u, index):
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

def xs_matrix_coarsen(xs_scatter_u, energy_edges_u, index):
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
    return memoryview(np.array(xs_scatter_c))

def velocity_mean_coarsen(velocity_u, index):
    velocity_c = np.zeros((len(index) - 1))
    for group, (left, right) in enumerate(zip(index[:-1], index[1:])):
        velocity_c[group] = np.mean(velocity_u[left:right])
    return velocity_c

########################################################################
# Indexing for Hybrid Methods
########################################################################

def calculate_collided_index(fine, index):
    # Of length (fine)
    # Which coarse group the fine group is a part of 
    index_collided = np.zeros((fine), dtype=np.int32)
    splits = [slice(ii,jj) for ii,jj in zip(index[:-1],index[1:])]
    for count, split in enumerate(splits):
        index_collided[split] = count
    return index_collided

def calculate_hybrid_factor(fine, coarse, delta_u, delta_c, index):
    factor = delta_u.copy()
    splits = [slice(ii,jj) for ii,jj in zip(index[:-1],index[1:])]
    for count, split in enumerate(splits):
        for ii in range(split.start, split.stop):
            factor[ii] /= delta_c[count]
    return factor

def calculate_uncollided_index(coarse, index):
    # Of length (coarse + 1)
    # Location of edges between collided and uncollided grids
    index_uncollided = np.zeros((coarse+1), dtype=np.int32)
    splits = [slice(ii,jj) for ii,jj in zip(index[:-1],index[1:])]
    for count, split in enumerate(splits):
        index_uncollided[count+1] = split.stop
    return index_uncollided

def energy_bin_widths(energy_edges, index):
    # Return delta_u, delta_c
    energy_edges = np.asarray(energy_edges)
    return np.diff(energy_edges), np.diff(energy_edges[index])