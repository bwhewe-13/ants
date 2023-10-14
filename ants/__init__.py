########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

# Creating angular dimensions
from ants.main import angular_x, angular_xy

# Creating energy grids
from ants.main import energy_velocity, energy_grid

# Time steps
from ants.main import gamma_time_steps

# Creating medium maps
from ants.main import spatial1d, spatial2d, weight_spatial2d, weight_matrix2d

# To remove later
from ants.main import weight_triangle2d, weight_cylinder2d

# Creating sources
from ants.sources import materials, externals1d, boundaries1d
from ants.sources import externals2d, boundaries2d
