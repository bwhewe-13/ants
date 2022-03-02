
from ants import slab

import numpy as np
import matplotlib.pyplot as plt
import timeit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store', dest='clock')
usr_input = parser.parse_args()

def reeds():
    width = 16.
    cells = 1000
    delta_x = width/cells
    group = 1

    boundaries = [slice(0,int(2/delta_x)),slice(int(2/delta_x),int(3/delta_x)),
        slice(int(3/delta_x),int(5/delta_x)),slice(int(5/delta_x),int(6/delta_x)),
        slice(int(6/delta_x),int(10/delta_x)),slice(int(10/delta_x),int(11/delta_x)),
        slice(int(11/delta_x),int(13/delta_x)),slice(int(13/delta_x),int(14/delta_x)),
        slice(int(14/delta_x),int(16/delta_x))]

    total_xs = np.zeros((cells,group),dtype='float64')
    total_vals = [10,10,0,5,50,5,0,10,10]
    # total_vals = [1,1,0,5,50,5,0,1,1]

    scatter_xs = np.zeros((cells,group,group),dtype='float64')
    scatter_vals = [9.9,9.9,0,0,0,0,0,9.9,9.9]
    # scatter_vals = [0.9,0.9,0,0,0,0,0,0.9,0.9]

    source = np.zeros((cells,group),dtype='float64')
    source_vals = [0,1,0,0,50,0,0,1,0]

    for ii in range(len(boundaries)):
        total_xs[boundaries[ii]] = total_vals[ii]
        scatter_xs[boundaries[ii]] = np.diag(np.repeat(scatter_vals[ii],group))
        source[boundaries[ii]] = source_vals[ii]
    
    # scatter_xs = np.ones((cells,group,group),dtype='float64') * 0.1
    return total_xs, scatter_xs, source


def sweep_timer(sweep):
    SETUP_CODE = '''
from __main__ import reeds
import slab '''

    TEST_CODE = '''
slab.multigroup(slab.slab_{}, *reeds())
'''.format(sweep)
    times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, number=87, repeat=12)
    times.remove(max(times))
    times.remove(min(times))
    return times


if usr_input.clock:
    # python_times = sweep_timer('python')
    print('Ctypes...')
    ctypes_times = sweep_timer('ctypes')
    print('Numba...')
    numba_times = sweep_timer('numba')
    print('Cython...')
    cython_times = sweep_timer('cython')

    # print('{}\nPython Times\n{}\nAvg. {}\nStd. {}\n'.format('='*30, '-'*30, \
    #     np.mean(python_times), np.std(python_times)))
    print('{}\nCtypes Times\n{}\nAvg. {}\nStd. {}\n'.format('='*30, '-'*30, \
        np.mean(ctypes_times), np.std(ctypes_times)))
    print('{}\nNumba Times\n{}\nAvg. {}\nStd. {}\n'.format('='*30, '-'*30, \
        np.mean(numba_times), np.std(numba_times)))
    print('{}\nCython Times\n{}\nAvg. {}\nStd. {}\n'.format('='*30, '-'*30, \
        np.mean(cython_times), np.std(cython_times)))

else:
    # print('Python...')
    # phi_python = slab.multigroup(slab.slab_python, *reeds())

    # print('Ctypes...')
    # phi_ctypes = slab.multigroup(slab.slab_ctypes, *reeds())

    # print('Cython...')
    # phi_cython = slab.multigroup(slab.slab_cython, *reeds())
    
    print('Numba...')
    phi_numba = slab.multigroup(slab.slab_numba, *reeds())
    np.save('../reed_solution', phi_numba)

    # print(np.sum(phi_numba))
    # print(np.sum(phi_cython))

    # print(np.array_equal(phi_numba, phi_cython))
    # print(np.isclose(phi_numba, phi_cython).sum())

    fig, ax = plt.subplots()
    # ax.plot(phi_python, label='Python')
    # ax.plot(phi_ctypes, label='Ctypes')
    ax.plot(phi_numba, label='Numba')
    # ax.plot(phi_cython, label='Cython')

    # fig, ax = plt.subplots()            
    # ax.plot(phi_numba - phi_cython, label='Numba - Cython')
    # # ax.plot(phi_numba - phi_ctypes, label='Numba - Ctypes')
    # ax.legend(loc=0)
    plt.show()


# ==============================
# Ctypes Times
# ------------------------------
# Avg. 46.9437615682
# Std. 0.11777830462015924
# ==============================
# Numba Times
# ------------------------------
# Avg. 8.201501240900019
# Std. 0.0011661706839828383
# ==============================
# Cython Times
# ------------------------------
# Avg. 10.705369132900023
# Std. 0.0124044609971045
# ==============================