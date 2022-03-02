
# distutils: language=c++
# cython: cdivision=True

from libcpp.vector cimport vector
from libcpp cimport float

import numpy as np
# ctypedef vector[int] int_vec

cdef vector[int] sweep():
    # cdef size_t cells = 100
    cdef int a = 3
    cdef vector[int] flux 
    flux.resize(a)
    cdef Py_ssize_t cell
    for cell in range(<int> a):
        flux[cell] = 1
    cdef double[:,::1] mine = {1, 2, 3, 4.0}
    cdef vector[int] v = range(10)
    return v

def sweep_py(int a):
    cdef vector[int] myvec = range(10)
    myvec = sweep()
    print(a, myvec)


    # cdef int n = flux.size();
    # cdef size_t ii
    # cdef int arr = {1, 2, 3}
    # for ii in range(3):
    #     arr[ii] += flux[ii]


    
