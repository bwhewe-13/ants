########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
########################################################################

"""Shared physical and material constants used across ANTS.

Constants are grouped into three categories:

- Unit/physics conversion constants
- Material molar masses
- Material mass densities

All names are module-level uppercase constants intended for direct import.
"""


# Conversion Between Units
MASS_NEUTRON = 1.67493e-27
EV_TO_JOULES = 1.60218e-19
LIGHT_SPEED = 2.9979246e8
AVAGADRO = 6.022e23
CM_TO_BARNS = 1e-24
PI = 3.141592653589793


# Isotope and Material Molar Mass
URANIUM_MM = 238.0289
URANIUM_235_MM = 235.04393
URANIUM_238_MM = 238.0289
URANIUM_HYDRIDE_MM = 240.60467449999996

HYDROGEN_MM = 1.00784
CARBON_MM = 12.0116
HDPE_MM = 15.03512
MANGANESE_MM = 54.9380471
OXYGEN_MM = 15.9994
STAINLESS_440_MM = 52.68213573619713


# Isotope and Material Densities
URANIUM_RHO = 19.1
URANIUM_235_RHO = 18.8
URANIUM_238_RHO = 18.95
URANIUM_HYDRIDE_RHO = 10.95

HYDROGEN_RHO = 0.07
CARBON_RHO = 2.26
HDPE_RHO = 0.97
MANGANESE_RHO = 7.3
OXYGEN_RHO = 1.429
STAINLESS_440_RHO = 7.85
