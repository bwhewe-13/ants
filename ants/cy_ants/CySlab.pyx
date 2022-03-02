
# distutils: language=c++
# cython: cdivision=True

from libcpp.vector cimport vector
from libcpp.algorithm cimport for_each, any_of, fill, copy
from libcpp cimport float
from libc.math cimport sqrt, pow

import numpy as np

cdef double INNER_TOLERANCE = 1E-12
cdef size_t MAX_ITERATIONS = 100

cdef float width = 16
cdef size_t cells = 1000
cdef float delta_x = width / cells
cdef float cell_width = 1/delta_x

cdef int angles = 16
cdef vector[double] mu
cdef vector[double] w

mu.resize(angles)
w.resize(angles)
mu, w = np.polynomial.legendre.leggauss(angles)
# w = w / np.sum(np.asarray(w))

# python setup.py build_ext --inplace

# Copy array 2 into array 1 and zero out array 2
# cdef copy(vector[double]& arr1, vector[double]& arr2):
#     n = arr1.size()
#     for cell in range(<int> n):
#         arr1[cell] = arr2[cell]

cdef double convergence(vector[double]& arr1, vector[double]& arr2):
    n = arr1.size()
    cdef double change = 0.0
    for cell in range(<int> n):
        change += pow((arr1[cell] - arr2[cell]) / arr1[cell] / n, 2)
    change = sqrt(change)
    return change

def slab_cython(vector[float] total, vector[float] scatter, vector[float] source, \
                vector[double] phi_old):
    # cdef vector[double] phi_old
    # phi_old.resize(cells)
    # copy(phi_old, guess)

    cdef vector[double] phi 
    phi.resize(cells)
    fill(phi.begin(), phi.end(), 0.0)

    cdef bint converged = False
    cdef size_t count = 1
    cdef double psi_bottom, psi_top
    cdef double change = 0.0

    cdef size_t angle
    cdef size_t cell

    while not (converged):
        fill(phi.begin(), phi.end(), 0.0)
        for angle in range(angles):
            if mu[angle] > 0: # From 0 to I
                psi_bottom = 0.0
                for cell in range(cells):
                    psi_top = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_bottom * (mu[angle] * cell_width - 0.5 * total[cell]))\
                              * 1/(mu[angle] * cell_width  + 0.5 * total[cell]);

                    phi[cell] += (w[angle] * 0.25 * (psi_top + psi_bottom))
                    psi_bottom = psi_top
            elif mu[angle] < 0:
                psi_top = 0.0
                for cell in range(cells-1, -1, -1):
                    psi_bottom = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_top * (abs(mu[angle]) * cell_width - 0.5 * total[cell]))\
                              * 1/(abs(mu[angle]) * cell_width  + 0.5 * total[cell]);

                    phi[cell] += (w[angle] * 0.25 * (psi_top + psi_bottom))
                    psi_top = psi_bottom
        change = convergence(phi, phi_old)
        # print('Change:',change,'Flux:',np.sum(phi))
        # np.save('cython_count_{}'.format(count),np.asarray(phi))
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS) 
        count += 1
        # copy(phi, phi_old)
        copy(phi.begin(), phi.end(), phi_old.begin())
    # print(count, np.sum(np.asarray(phi)))
    return np.asarray(phi)




