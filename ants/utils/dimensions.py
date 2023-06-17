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
# Cylinder cross sections on cartesian grid
########################################################################

def cylinder_cross_sections(weight_map, xs_total, xs_scatter, xs_fission, \
                            cells_x, cells_y):
    # Convert cross sections to weight map
    cy_xs_total = []
    cy_xs_scatter = []
    cy_xs_fission = []
    if weight_map.shape[0] != (cells_x * cells_y):
        cells_x = int(0.5 * cells_x)
        cells_y = int(0.5 * cells_y)
    medium_map = np.zeros((cells_x * cells_y), dtype=np.int32)
    for mat, weight in enumerate(np.unique(weight_map, axis=0)):
        cy_xs_total.append(np.sum(xs_total * weight[:,None], axis=0))
        cy_xs_scatter.append(np.sum(xs_scatter * weight[:,None,None], axis=0))
        cy_xs_fission.append(np.sum(xs_fission * weight[:,None,None], axis=0))
        medium_map[np.where(np.all(weight_map == weight, axis=1))] = mat
    medium_map = medium_map.reshape(cells_x, cells_y)
    cy_xs_total = np.array(cy_xs_total)
    cy_xs_scatter = np.array(cy_xs_scatter)
    cy_xs_fission = np.array(cy_xs_fission)
    return medium_map, cy_xs_total, cy_xs_scatter, cy_xs_fission

def expand_cylinder_medium_map(quad4, quadrants=[1,2,3,4]):
    # Assumes that medium_map is quadrant IV
    quad1 = np.flip(quad4, axis=0).copy()
    quad2 = np.flip(quad4, axis=(1,0)).copy()
    quad3 = np.flip(quad4, axis=1).copy()
    if quadrants == [1,2,3,4]: # Full circle
        medium_map = np.block([[quad2, quad1], [quad3, quad4]])
    elif quadrants == [1,2]: # Upper semi
        medium_map = np.block([quad2, quad1])
    elif quadrants == [1,4]: # Right semi
        medium_map = np.block([[quad1], [quad4]])
    elif quadrants == [2,3]: # Left semi
        medium_map = np.block([[quad2], [quad3]])
    elif quadrants == [3,4]: # Lower semi
        medium_map = np.block([quad3, quad4])
    else:
        medium_map = quad1.copy()
    return medium_map