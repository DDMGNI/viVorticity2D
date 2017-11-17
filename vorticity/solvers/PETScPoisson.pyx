'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, SNES, Vec

from PETScDerivatives import PETScDerivatives


cdef class PETScPoisson(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, object da1,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.hx = hx
        self.hy = hy
        
        self.hx2inv = 1. / self.hx**2
        self.hy2inv = 1. / self.hy**2
        
        # create vorticity vector
        self.Op = da1.createGlobalVec()
        
        # create local vectors
        self.localO = da1.createLocalVec()
        self.localP = da1.createLocalVec()
        
    
    def updateVorticity(self, Vec O):
        O.copy(self.Op)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                row.index = (i,j)
                
                for index, value in [
                        ((i,   j-1),                    - 1. * self.hy2inv),
                        ((i-1, j  ), - 1. * self.hx2inv                   ),
                        ((i,   j  ), + 2. * self.hx2inv + 2. * self.hy2inv),
                        ((i+1, j  ), - 1. * self.hx2inv                   ),
                        ((i,   j+1),                    - 1. * self.hy2inv),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)

        A.assemble()
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef matrix_mult(self, Vec P, Vec Y):
        cdef int i, j, ix, jx, iy, jy
        cdef int xs, xe, sw
         
        self.da1.globalToLocal(P, self.localP)
         
        cdef double[:,:] y = self.da1.getVecArray(Y)[...]
        cdef double[:,:] p = self.da1.getVecArray(self.localP)[...]
         
        (xs, xe), (ys, ye) = self.da1.getRanges()
         
        for i in range(xs, xe):
            ix = i-xs+1
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+1
                jy = j-ys
                 
                y[iy,jy] = (2. * p[ix,jx] - p[ix-1,jx] - p[ix+1,jx]) * self.hx2inv \
                         + (2. * p[ix,jx] - p[ix,jx-1] - p[ix,jx+1]) * self.hy2inv
    
    
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
         
        (xs, xe), (ys, ye) = self.da1.getRanges()
         
        self.da1.globalToLocal(self.Op, self.localO)
         
        cdef double[:,:] b = self.da1.getVecArray(B)[...]
        cdef double[:,:] o = self.da1.getVecArray(self.localO)[...]
        
        cdef double omean = self.Op.sum() / (self.nx * self.ny)
        
        for i in range(xs, xe):
            ix = i-xs+1
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+1
                jy = j-ys
                 
                b[iy,jy] = - o[ix, jx] + omean
                

    @cython.boundscheck(False)
    cpdef function(self, Vec P, Vec F):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        self.da1.globalToLocal(P,       self.localP)
        self.da1.globalToLocal(self.Op, self.localO)
        
        cdef double[:,:] f = self.da1.getVecArray(F)[...]
        cdef double[:,:] o = self.da1.getVecArray(self.localO)[...]
        cdef double[:,:] p = self.da1.getVecArray(self.localP)[...]
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double omean = self.Op.sum() / (self.nx * self.ny)
        
        for i in range(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                f[iy,jy] = (2. * p[ix,jx] - p[ix-1,jx] - p[ix+1,jx]) * self.hx2inv \
                         + (2. * p[ix,jx] - p[ix,jx-1] - p[ix,jx+1]) * self.hy2inv \
                         + (o[ix,jx] - omean)
    

    
    cpdef mult(self, Mat mat, Vec P, Vec Y):
        self.matrix_mult(P, Y)
          
    cpdef matrix_snes_mult(self, SNES snes, Vec P, Vec Y):
        self.matrix_mult(P, Y)
         
    cpdef function_snes_mult(self, SNES snes, Vec P, Vec Y):
        self.function(P, Y)
