
from ants.cy_ants.CySlab import slab_cython

import numpy as np
import cupy as cp
import ctypes
import numba


width = 16
cells = 1000
delta_x = width / cells
# delta_x = cells / width
cell_width = 1/delta_x


angles = 16
mu, w = np.polynomial.legendre.leggauss(angles)
w /= np.sum(w)


INNER_TOLERANCE = 1E-12
OUTER_TOLERANCE = 1E-8
MAX_ITERATIONS = 100


def slab_python(total, scatter, source, guess=None):
    phi_old = np.zeros((cells),dtype='float64') if guess is None else guess.copy()
    converged = 0; 
    count = 1
    while not (converged):
        phi = np.zeros((cells),dtype='float64')
        psi_bottom = 0.0
        for angle in range(angles):            
            if mu[angle] > 0: # From 0 to I
                psi_bottom = 0.0
                for cell in range(cells):
                    psi_top = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_bottom * (mu[angle] * cell_width - 0.5 * total[cell]))\
                              * 1/(mu[angle] * cell_width  + 0.5 * total[cell]);
                    phi[cell] += (w[angle] * 0.5 * (psi_top + psi_bottom))
                    psi_bottom = psi_top
            elif mu[angle] < 0:
                psi_top = 0.0
                for cell in range(cells-1, -1, -1):
                    psi_bottom = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_top * (abs(mu[angle]) * cell_width - 0.5 * total[cell]))\
                              * 1/(abs(mu[angle]) * cell_width  + 0.5 * total[cell]);
                    phi[cell] += (w[angle] * 0.5 * (psi_top + psi_bottom))
                    psi_top = psi_bottom
        change = np.linalg.norm((phi - phi_old)/phi/(cells))
        # print('Change:',change,'Flux:',np.sum(phi))
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS) 
        count += 1
        phi_old = phi.copy()
    return phi

# ctypes
# gcc -fPIC -shared -o cSlab.so cSlab.c 
def slab_ctypes(total, scatter, source, guess=None):
    clibrary = ctypes.cdll.LoadLibrary('c_ants/cSlab.so')
    sweep = clibrary.vacuum

    phi_old = np.zeros((cells),dtype='float64') if guess is None else guess.copy()
    source = source.astype('float64')
    source_ptr = ctypes.c_void_p(source.ctypes.data)

    converged = 0; 
    count = 1
    while not (converged):
        phi = np.zeros((cells),dtype='float64')
        for angle in range(angles):
            direction = ctypes.c_int(int(np.sign(mu[angle])))
            weight = np.sign(mu[angle]) * mu[angle] * cell_width

            top_mult = (weight - 0.5 * total).astype('float64')
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

            bottom_mult = (1/(weight + 0.5 * total)).astype('float64')
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

            temp_scat = (scatter * phi_old).astype('float64')
            ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
            
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            
            sweep(phi_ptr, ts_ptr, source_ptr, top_ptr, bot_ptr, \
                ctypes.c_double(w[angle]), direction)

        change = np.linalg.norm((phi - phi_old)/phi/(cells))
        # if np.isnan(change) or np.isinf(change):
        #     change = 0.
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS) 
        count += 1
        phi_old = phi.copy()
    return phi

@numba.jit(nopython=True, cache=True)
def slab_numba(total, scatter, source, guess=None):
    phi_old = np.zeros((cells),dtype='float64') if guess is None else guess.copy()
    converged = 0; 
    count = 1
    while not (converged):
        phi = np.zeros((cells),dtype='float64')
        psi_bottom = 0.0
        for angle in range(angles):
        # for angle in numba.prange(angles):
            if mu[angle] > 0: # From 0 to I
                psi_bottom = 0.0
                for cell in range(cells):
                    psi_top = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_bottom * (mu[angle] * cell_width - 0.5 * total[cell]))\
                              * 1/(mu[angle] * cell_width  + 0.5 * total[cell]);

                    phi[cell] += (w[angle] * 0.5 * (psi_top + psi_bottom))
                    psi_bottom = psi_top
            elif mu[angle] < 0:
                psi_top = 0.0
                for cell in range(cells-1, -1, -1):
                    psi_bottom = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_top * (abs(mu[angle]) * cell_width - 0.5 * total[cell]))\
                              * 1/(abs(mu[angle]) * cell_width  + 0.5 * total[cell]);

                    phi[cell] += (w[angle] * 0.5 * (psi_top + psi_bottom))
                    psi_top = psi_bottom
        change = np.linalg.norm((phi - phi_old)/phi/(cells))
        # print('Change:',change,'Flux:',np.sum(phi))
        converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS) 
        count += 1
        phi_old = phi.copy()
    # print(count, np.sum(phi))
    return phi

def multigroup(function, total, scatter, source):
    converged = 0
    count = 1
    g = 0
    phi_old = np.zeros((cells))
    while not (converged):
        phi = np.zeros((cells),dtype='float64')
        phi = function(total[:,g], scatter[:,g,g], source[:,g], phi_old)
        # phi = np.array(phi)
        # print(np.sum(phi))
        change = cp.linalg.norm((phi - phi_old)/phi/(cells))
        count += 1
        converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS) 
        phi_old = phi.copy()
    return phi
