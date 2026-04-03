################################################################################
#                            ___    _   _____________
#                           /   |  / | / /_  __/ ___/
#                          / /| | /  |/ / / /  \__ \
#                         / ___ |/ /|  / / /  ___/ /
#                        /_/  |_/_/ |_/ /_/  /____/
#
# Declaration file for cytools_shared.pyx — fused-type shared implementations.
# Import this from cytools_1d.pyx and cytools_2d.pyx.
#
################################################################################

from ants.parameters cimport params

################################################################################
# Fused types
################################################################################

ctypedef fused scalar_flux_nd:
    double[:,:]
    double[:,:,:]

ctypedef fused spatial_nd:
    double[:]
    double[:,:]

ctypedef fused medium_map_nd:
    int[:]
    int[:,:]

################################################################################
# Function declarations
################################################################################

cdef double group_convergence(scalar_flux_nd arr1, scalar_flux_nd arr2,
                               params info)

cdef double angle_convergence(spatial_nd arr1, spatial_nd arr2, params info)

cdef void _normalize_flux(scalar_flux_nd flux, params info)

cdef double _update_keffective(scalar_flux_nd flux_new, scalar_flux_nd flux_old,
                                double[:,:,:] xs_fission,
                                medium_map_nd medium_map,
                                params info, double keff)

cdef void _total_velocity(double[:,:]& xs_total, double[:]& velocity,
                           double constant, params info)
