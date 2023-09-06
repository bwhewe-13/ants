########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Writing python functions for block interpolation
# 
########################################################################

import numpy as np

def _material_index(medium_map):
    """ Finding index of one-dimensional medium map
    Arguments:
        medium_map (int [cells_x]): one-dimensional array of materials, 
            where each different number represents a new material
    Returns:
        array of indexes between material interfaces
    """
    # Get middle indices
    splits = list(np.where(medium_map[:-1] != medium_map[1:])[0] + 1)
    # Add endpoints
    splits = np.array([0] + splits + [len(medium_map)], dtype=np.int32)
    return splits

def _to_block(medium_map):
    """ Separating medium map into chunks for MNP for each chunk
    Arguments:
        medium_map (int [cells_x, cells_y]): two-dimensional array of 
            materials where each different number represents a new material
    Returns:
        x_splits (int []): array of index for x direction
        y_splits (int []): array of index for y direction
    """
    # X direction - for each y cell
    x_splits = []
    for jj in range(medium_map.shape[1]):
        x_splits.append(list(_material_index(medium_map[:,jj])))
    x_splits = np.unique([ii for lst in x_splits for ii in lst])
    # Y direction - for each x cell
    y_splits = []
    for ii in range(medium_map.shape[0]):
        y_splits.append(list(_material_index(medium_map[ii])))
    y_splits = np.unique([ii for lst in y_splits for ii in lst])
    # Return two arrays (might be different lengths)
    return x_splits, y_splits

def _global_splits(medium_map, nx, ny):
    """ Convert 2D medium map of size (I x J) to size (Nx x Ny)
    Arguments:
        medium_map (int [cells_x, cells_y]): two-dimensional array of 
            materials where each different number represents a new material
        nx (double [nx]): one-dimensional array of points to interpolate 
            in x direction
        ny (double [ny]): one-dimensional array of points to interpolate 
            in y direction
    Returns:
        n_medium_map (int [nx, ny])
    """
    ratio_x = int(nx.shape[0] / medium_map.shape[0])
    ratio_y = int(ny.shape[0] / medium_map.shape[1])
    n_medium_map = np.repeat(medium_map, ratio_x, axis=0)
    n_medium_map = np.repeat(n_medium_map, ratio_y, axis=1)
    return n_medium_map
