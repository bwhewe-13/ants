########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
########################################################################

from ants import boundary1d, boundary2d, external1d, external2d

# Creating medium maps
# Time steps
# Creating energy grids
# Creating angular dimensions
from ants.main import (
    _angular_x,
    _angular_xy,
    _energy_grid,
    angular_x,
    angular_xy,
    energy_grid,
    energy_velocity,
    gamma_time_steps,
    spatial1d,
    spatial2d,
    weight_matrix2d,
    weight_spatial2d,
)

# Creating materials/sources
from ants.materials import materials
