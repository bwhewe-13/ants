########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from typing import Final

INNER_TOLERANCE: Final[float] = 1E-12
OUTER_TOLERANCE: Final[float] = 1E-8
MAX_ITERATIONS: Final[int] = 100

MASS_NEUTRON: Final[float] = 1.67493E-27
EV_TO_JOULES: Final[float] = 1.60218E-19
LIGHT_SPEED: Final[float] = 2.9979246E8
AVAGADRO_NUMBER: Final[float] = 6.022E23
CM_TO_BARNS: Final[float] = 1E-24

URANIUM_235_MM: Final[float] = 235.04393
URANIUM_238_MM: Final[float] = 238.0289
HYDROGEN_MM: Final[float] = 1.00784

URANIUM_HYDRIDE_RHO: Final[float] = 10.95
URANIUM_RHO: Final[float] = 19.1