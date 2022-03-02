########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import ants.constants as constants

import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def slab_x(phi_old, total, scatter, source, cells_x, cell_widths_x, mu_x, weight):
    converged = 0
    count = 1
    while not (converged):
        phi = np.zeros((cells_x),dtype='float64')
        psi_bottom = 0.0
        # psi_top = 0.0
        for angle in range(len(mu_x)):
            if mu_x[angle] > 0: # From 0 to I
                psi_bottom = 0.0
                for cell in range(cells_x):
                    psi_top = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_bottom * (mu_x[angle] / cell_widths_x - 0.5 * total[cell]))\
                              * 1/(mu_x[angle] / cell_widths_x  + 0.5 * total[cell])
                    phi[cell] += (weight[angle] * 0.5 * (psi_top + psi_bottom))
                    psi_bottom = psi_top
                    # print(type(psi_top), type(psi_bottom))
            elif mu_x[angle] < 0:
                psi_top = 0.0
                for cell in range(cells_x-1, -1, -1):
                    psi_bottom = (scatter[cell] * phi_old[cell] + source[cell]\
                             + psi_top * (abs(mu_x[angle]) / cell_widths_x - 0.5 * total[cell]))\
                              * 1/(abs(mu_x[angle]) / cell_widths_x  + 0.5 * total[cell]);
                    phi[cell] += (weight[angle] * 0.5 * (psi_top + psi_bottom))
                    psi_top = psi_bottom
        change = np.linalg.norm((phi - phi_old)/phi/(cells_x))
        # print('Change:',change,'Flux:',np.sum(phi))
        converged = (change < constants.INNER_TOLERANCE) \
                    or (count >= constants.MAX_ITERATIONS) 
        count += 1
        phi_old = phi.copy()
    return phi

# def slab_x(phi_old, total, scatter, source):
#     converged = 0
#     count = 1
#     while not (converged):
#         phi = np.zeros((cells),dtype='float64')
#         psi_bottom = 0.0
#         for angle in range(angles):
#             if mu[angle] > 0: # From 0 to I
#                 psi_bottom = 0.0
#                 for cell in range(cells):
#                     psi_top = (scatter[cell] * phi_old[cell] + source[cell]\
#                              + psi_bottom * (mu[angle] * cell_width - 0.5 * total[cell]))\
#                               * 1/(mu[angle] * cell_width  + 0.5 * total[cell]);

#                     phi[cell] += (w[angle] * 0.5 * (psi_top + psi_bottom))
#                     psi_bottom = psi_top
#             elif mu[angle] < 0:
#                 psi_top = 0.0
#                 for cell in range(cells-1, -1, -1):
#                     psi_bottom = (scatter[cell] * phi_old[cell] + source[cell]\
#                              + psi_top * (abs(mu[angle]) * cell_width - 0.5 * total[cell]))\
#                               * 1/(abs(mu[angle]) * cell_width  + 0.5 * total[cell]);

#                     phi[cell] += (w[angle] * 0.5 * (psi_top + psi_bottom))
#                     psi_top = psi_bottom
#         change = np.linalg.norm((phi - phi_old)/phi/(cells))
#         # print('Change:',change,'Flux:',np.sum(phi))
#         converged = (change < constants.INNER_TOLERANCE) \
#                     or (count >= constants.MAX_ITERATIONS) 
#         count += 1
#         phi_old = phi.copy()
#     return phi

if __name__ == '__main__':
	print(constants.MAX_ITERATIONS)
