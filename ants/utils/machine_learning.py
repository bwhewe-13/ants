########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# This is for cleaning, running and transforming DJINN models
# 
########################################################################

import numpy as np
from glob import glob


def _combine_flux_reaction(flux, xs_matrix, medium_map, labels):
    # Flux parameters
    iterations, cells_x, groups = flux.shape
    # Initialize training data
    data = np.zeros((2, iterations, cells_x, groups + 1))
    # Initialize counter
    count = 0
    # Iterate over iterations and spatial cells
    for cc in range(iterations):
        for ii in range(cells_x):
            mat = medium_map[ii]
            if np.sum(xs_matrix[mat]) == 0.0:
                count += 1
                continue
            # Add labels
            data[:,cc,ii,0] = labels[mat]
            # Add flux (x variable)
            data[0,cc,ii,1:] = flux[cc,ii].copy()
            # Add reaction rate (y variable)
            data[1,cc,ii,1:] = flux[cc,ii] @ xs_matrix[mat].T
    # Collapse iteration and spatial dimensions
    data = data.reshape(2, iterations * cells_x, groups + 1)
    # Remove zero values
    idx = np.argwhere(np.sum(data[...,1:], axis=(0,2)) != 0)
    data = data[:,idx.flatten(),:].copy()
    assert (data.shape[1] + count) == (iterations * cells_x), "Need to equal"
    return data


def clean_data_fission(path, labels):
    """ Takes the flux before the fission rates are calculated (x data), 
    calculates the reaction rates (y data), and adds a label for the 
    enrichment level (G+1). Also removes non-fissioning materials. 
    Arguments:
        path (str): location of all files named in djinn1d.collections()
        labels (float [materials]): labels for each of the materials
    Returns:
        Processed data saved to path
    """
    # Load the data
    flux = np.load(path + "flux_fission_model.npy")
    xs_fission = np.load(path + "fission_cross_sections.npy")
    medium_map = np.load(path + "medium_map.npy")
    training_data = _combine_flux_reaction(flux, xs_fission, medium_map, labels)
    np.save(path + "fission_training_data", training_data)
    # return training_data


def clean_data_scatter(path, labels):
    """ Takes the flux before the scattering rates are calculated (x data), 
    calculates the reaction rates (y data), and adds a label for the 
    enrichment level (G+1).
    Arguments:
        path (str): location of all files named in djinn1d.collections()
        labels (float [materials]): labels for each of the materials
    Returns:
        Processed data saved to path
    """
    # Load the data
    files = np.sort(glob(path + "flux_scatter_model*.npy"))
    xs_scatter = np.load(path + "scatter_cross_sections.npy")
    medium_map = np.load(path + "medium_map.npy")
    training_data = np.empty((2, 0, xs_scatter.shape[1] + 1))
    for file in files:
        flux = np.load(file)
        single_iteration = _combine_flux_reaction(flux, xs_scatter, \
                                                  medium_map, labels)
        training_data = np.hstack((training_data, single_iteration))
    np.save(path + "scatter_training_data", training_data)
    # return training_data


