'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport DMDA, Mat, SNES, Vec


cdef class PETScPoisson(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef np.float64_t hx2inv
    cdef np.float64_t hy2inv
    
    cdef DMDA da1
    
    cdef Vec Op
    
    cdef Vec localO
    cdef Vec localP


    cpdef mult(self, Mat mat, Vec P, Vec Y)
    cpdef matrix_mult(self, Vec P, Vec F)
    cpdef matrix_snes_mult(self, SNES snes, Vec P, Vec Y)
    cpdef function(self, Vec P, Vec F)
    cpdef function_snes_mult(self, SNES snes, Vec P, Vec Y)
