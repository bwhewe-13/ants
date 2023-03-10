########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np

def monte_carlo_weight_matrix(delta_x, delta_y, center, radii, samples=100000):
    # Calculate the fraction of each material inside and outside radii
    weight_map = []
    # Global spatial grid edges
    np.random.seed(42)
    edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
    edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
    # Set local grid points
    grid_x = edges_x[(edges_x >= center[0]) & (edges_x <= center[0] \
                        + max(radii)[1])] - center[0]
    grid_y = edges_y[(edges_y >= center[1]) & (edges_y <= center[1] \
                        + max(radii)[1])] - center[1]
    # Calculate all samples
    samples_x = np.random.uniform(0, max(radii)[1], samples)
    samples_y = np.random.uniform(0, max(radii)[1], samples)
    # Iterate over spatial grid
    for yy in range(len(grid_y) - 1):
        for xx in range(len(grid_x) - 1):
            weight_map.append(_weight_grid_cell(samples_x, samples_y, radii, \
                        grid_x[xx], grid_x[xx+1], grid_y[yy], grid_y[yy+1]))
    return weight_map / np.sum(weight_map, axis=1)[:,None]

def _weight_grid_cell(x, y, radii, x1, x2, y1, y2):
    ppr = []
    for iir, oor in radii:
        # Collect particles in circle
        idx = np.argwhere(((x**2 + y**2) > iir**2) & ((x**2 + y**2) <= oor**2))
        temp_x = x[idx].copy()
        ppr.append(len(temp_x[np.where((temp_x >= x1) & (temp_x < x2) \
                                    & (y[idx] >= y1) & (y[idx] < y2))]))
    # Collect particles outside circle
    temp_x = x[(x**2 + y**2) > max(radii)[1]**2]
    temp_y = y[(x**2 + y**2) > max(radii)[1]**2]
    ppr.append(len(temp_x[np.where((temp_x >= x1) & (temp_x < x2) \
                                    & (temp_y >= y1) & (temp_y < y2))]))
    return ppr

# def add_circle(total, scatter, fission, circ_total, circ_scatter, \
#                     circ_fission, medium_map, circle_map, delta_x, delta_y, \
#                     center, radii):
#     # Adding circle map and cross sections to established medium map
#     edges_x = np.round(np.insert(np.cumsum(delta_x), 0, 0), 12)
#     edges_y = np.round(np.insert(np.cumsum(delta_y), 0, 0), 12)
#     x1 = np.argwhere(edges_x == center[0] - max(radii)[1])[0,0]
#     x2 = np.argwhere(edges_x == center[0] + max(radii)[1])[0,0]
#     y1 = np.argwhere(edges_y == center[1] - max(radii)[1])[0,0]
#     y2 = np.argwhere(edges_y == center[1] + max(radii)[1])[0,0]
#     # print(idx1, idx2)
#     # current = np.max(medium_map)
#     current = np.max(medium_map) if x1 == 0 else np.max(medium_map) + 1
#     circle_map = circle_map + current
#     medium_map[x1:x2, y1:y2] = circle_map
#     xs_total = np.concatenate((total[:current,:], circ_total))
#     xs_scatter = np.concatenate((scatter[:current,:,:], circ_scatter))
#     xs_fission = np.concatenate((fission[:current,:,:], circ_fission))
#     return medium_map, xs_total, xs_scatter, xs_fission