import numpy as np
cimport numpy as np
import random
import cython

from libc.stdlib cimport rand
cdef extern from "limits.h":
    int INT_MAX

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def construct_levels(Py_ssize_t n_levels, Py_ssize_t dim):
    L = np.zeros(shape=(n_levels, dim), dtype=np.intc)
    L[0, :] = np.random.randint(2, size=[dim])
    cdef int[:,:] L_view = L
    cdef Py_ssize_t i, j
    cdef double prob = 1 / <double>n_levels
    for i in range(1, n_levels):
        for j in range(dim):
            if random.random() < prob:
                L_view[i, j] = 1 - L_view[i-1, j]
            else:
                L_view[i, j] = L_view[i-1, j]
    return L

cdef class record_based:
    cdef int q
    cdef np.ndarray L
    cdef np.ndarray ID
    def __cinit__(self, int real_dim, int HD_dim, int q):
        self.ID = np.random.randint(2, size=[real_dim, HD_dim])
        self.L = construct_levels(n_levels=q+1, dim=HD_dim)
        self.q = q
        
    def quantize(self, np.ndarray x):
        # x is a real vector of size real_dim.
        x = (x * self.q).astype(int)
        z = np.logical_xor(self.L[x, :], self.ID)
        z = np.round(np.mean(z, axis=0)).astype(int)
        return z
    
    def quantize_batch(self, np.ndarray x):
        return np.array([self.quantize(xi) for xi in x])

cdef class N_gram_based:
    cdef int q
    cdef int real_dim
    cdef int HD_dim
    cdef np.ndarray L
    cdef np.ndarray rho
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def __cinit__(self, int real_dim, int HD_dim, int q):
        self.L = construct_levels(n_levels=q+1, dim=HD_dim)
        self.real_dim = real_dim
        self.HD_dim = HD_dim
        self.q = q
        
        tmp_rho = np.random.permutation(HD_dim)
        rho = np.zeros(shape=[real_dim, HD_dim], dtype=int)
        rho[0, :] = np.arange(HD_dim)
        
        cdef Py_ssize_t[::1] tmp_rho_view = tmp_rho
        cdef Py_ssize_t[:,:] rho_view = rho
        cdef Py_ssize_t i, j
        for i in range(1, real_dim):
            for j in range(HD_dim):
                rho_view[i, j] = tmp_rho_view[rho_view[i-1, j]]
        self.rho = rho
                
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def quantize(self, np.ndarray x):

        assert len(x) == self.real_dim
        
        result = np.zeros(shape=[self.real_dim, self.HD_dim], dtype=np.intc)
        cdef double[::1] x_view = x
        cdef int[:,:] result_view = result
        cdef int[:,:] L_view = self.L

        cdef Py_ssize_t[:,:] rho_view = self.rho

        cdef Py_ssize_t i, j
        for i in range(self.real_dim):
            idx = <int>(x_view[i] * self.q)
            for j in range(self.HD_dim):
                result_view[i, j] = L_view[idx, rho_view[i,j]]

        result = (np.sum(result, axis=0) > self.real_dim / 2).astype(int)

        return result

    def quantize_batch(self, np.ndarray x):
        return np.array([self.quantize(xi) for xi in x])

cdef class distance_preserving:
    cdef int q
    cdef int real_dim
    def __cinit__(self, int real_dim, int q):
        self.real_dim = real_dim
        self.q = q
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def quantize(self, np.ndarray x):
        
        assert len(x) == self.real_dim
        
        result = np.zeros(shape=[self.real_dim * self.q], dtype=np.intc)
        cdef double[::1] x_view = x
        cdef int[::1] result_view = result
        cdef Py_ssize_t i, j
        for i in range(self.real_dim):
            idx = <int>(x_view[i] * self.q)
            for j in range(idx):
                result_view[i * self.q + j] = 1
        return result