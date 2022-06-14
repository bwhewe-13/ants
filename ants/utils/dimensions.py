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
