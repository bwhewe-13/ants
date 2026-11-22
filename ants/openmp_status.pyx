########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Reports whether the compiled extensions were built with OpenMP.
# Without OpenMP, Cython's prange runs serially and num_threads has
# no effect.
#
########################################################################

# _OPENMP is defined only when compiled with OpenMP
cdef extern from *:
    """
    #ifdef _OPENMP
    #define ANTS_OPENMP_ENABLED 1
    #else
    #define ANTS_OPENMP_ENABLED 0
    #endif
    """
    int ANTS_OPENMP_ENABLED


def openmp_enabled():
    """Return True if the ANTS extensions were compiled with OpenMP."""
    return bool(ANTS_OPENMP_ENABLED)
