'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, SNES, Vec


cdef class PETScVorticity(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  ny
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hy
    
    cdef np.float64_t arakawa_factor
    
    cdef object da1
    
    cdef Vec Pp
    cdef Vec Oh
    cdef Vec Ph
    
    cdef Vec localOp
    cdef Vec localOh
    cdef Vec localPp
    cdef Vec localPh
    

    cpdef mult(self, Mat mat, Vec O, Vec Y)
    cpdef matrix_mult(self, Vec O, Vec F)
    cpdef matrix_snes_mult(self, SNES snes, Vec O, Vec Y)
    cpdef function(self, Vec O, Vec F)
    cpdef function_snes_mult(self, SNES snes, Vec O, Vec Y)
    cpdef explicit(self, Vec O, Vec P, Vec F)
