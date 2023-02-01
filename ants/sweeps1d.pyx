########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Mono-energetic spatial sweeps for single ordinates. Different spatial
# discretizations are considered.
#
########################################################################

# distutils: language=c++
# cython: cdivision=True

from ants.cytools_1d cimport params1d

from libcpp cimport float

cdef double sweep(double flux, double flux_old, double xs_total, \
                double xs_matrix, double off_scatter, double source, \
                double mu, double angle_w, double cell_width, \
                double[:] known_edge, float xs1_const, float xs2_const, \
                params1d params, bint angular):
    if params.geometry == 1:
        return slab_sweep(flux, flux_old, xs_total, xs_matrix, off_scatter, \
                    source, mu, angle_w, cell_width, known_edge, xs1_const, \
                    xs2_const, params, angular)


cdef double slab_sweep(double flux, double flux_old, double xs_total, \
                double xs_matrix, double off_scatter, double source, \
                double mu, double angle_w, double cell_width, \
                double[:] known_edge, float xs1_const, float xs2_const, \
                params1d params, bint angular):
    cdef double unknown_edge
    unknown_edge = (xs_matrix * flux_old + source + off_scatter \
            + known_edge[0] * (abs(mu / cell_width) + xs1_const * xs_total)) \
            * 1/(abs(mu / cell_width) + xs2_const * xs_total)
    if angular == True:
        angle_w = 1
    if params.spatial == 1:
        flux += angle_w * unknown_edge
    elif params.spatial == 2:
        flux += 0.5 * angle_w * (known_edge[0] + unknown_edge) 
    known_edge[0] = unknown_edge
    return flux

cdef double find_boundary(double prev_edge, double mu, double[:] boundary, \
                            params1d params):
    if (mu > 0 and params.bc[0] == 1) or (mu < 0 and params.bc[1] == 1):
        return prev_edge
    elif mu > 0:
        return boundary[0]
    else:
        return boundary[1]
