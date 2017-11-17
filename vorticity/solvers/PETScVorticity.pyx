'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport DMDA, SNES, Mat, Vec


cdef class PETScVorticity(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DMDA da1,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy

        self.arakawa_factor = 1. / (12. * self.hx * self.hy)
        
        # create streaming function vector
        self.Pp = da1.createGlobalVec()
        
        # create history vectors
        self.Oh = da1.createGlobalVec()
        self.Ph = da1.createGlobalVec()
        
        # create local vectors
        self.localOp = da1.createLocalVec()
        self.localOh = da1.createLocalVec()
        self.localPp = da1.createLocalVec()
        self.localPh = da1.createLocalVec()
        
    
    def updateHistory(self, Vec Oh, Vec Ph):
        Oh.copy(self.Oh)
        Ph.copy(self.Ph)
    
    
    def updateStreamingFunction(self, Vec P):
        P.copy(self.Pp)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(self.Pp, self.localPp)
        self.da1.globalToLocal(self.Ph, self.localPh)
        
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.localPp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = self.da1.getVecArray(self.localPh)[...]
        
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        
        cdef double time_fac  = 1.0
        cdef double arak_fac  = 0.5 * self.arakawa_factor * self.ht
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        for i in range(xs, xe):
            ix = i-xs+1
            
            for j in range(ys, ye):
                jx = j-ys+1
                
                row.index = (i,j)
                
                for index, value in [
                        ((i-1, j-1), + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), time_fac),
                        ((i,   j+1), + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * arak_fac),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)

        A.assemble()
        

    @cython.boundscheck(False)
    cpdef matrix_mult(self, Vec O, Vec F):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        cdef double jpp, jpc, jcp
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(O,       self.localOp)
        self.da1.globalToLocal(self.Pp, self.localPp)
        self.da1.globalToLocal(self.Ph, self.localPh)
        
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.localPp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = self.da1.getVecArray(self.localPh)[...]
        
        cdef double[:,:] f     = self.da1.getVecArray(F)[...]
        cdef double[:,:] Op    = self.da1.getVecArray(self.localOp)[...]
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        
        
        for i in range(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                jpp = (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * (Op[ix,   jx+1] - Op[ix,   jx-1]) \
                    - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * (Op[ix+1, jx  ] - Op[ix-1, jx  ])
                
                jpc = P_ave[ix+1, jx  ] * (Op[ix+1, jx+1] - Op[ix+1, jx-1]) \
                    - P_ave[ix-1, jx  ] * (Op[ix-1, jx+1] - Op[ix-1, jx-1]) \
                    - P_ave[ix,   jx+1] * (Op[ix+1, jx+1] - Op[ix-1, jx+1]) \
                    + P_ave[ix,   jx-1] * (Op[ix+1, jx-1] - Op[ix-1, jx-1])
                
                jcp = P_ave[ix+1, jx+1] * (Op[ix,   jx+1] - Op[ix+1, jx  ]) \
                    - P_ave[ix-1, jx-1] * (Op[ix-1, jx  ] - Op[ix,   jx-1]) \
                    - P_ave[ix-1, jx+1] * (Op[ix,   jx+1] - Op[ix-1, jx  ]) \
                    + P_ave[ix+1, jx-1] * (Op[ix+1, jx  ] - Op[ix,   jx-1])
    
                f[iy, jy] = Op[ix,jx] + 0.5 * (jpp + jpc + jcp) * self.arakawa_factor * self.ht
            
    
#     @cython.boundscheck(False)
#     def formRHS(self, Vec B):
#         cdef np.int64_t i, j
#         cdef np.int64_t ix, iy, jx, jy
#         cdef np.int64_t xe, xs, ye, ys
#         
#         (xs, xe), (ys, ye) = self.da1.getRanges()
#         
#         self.da1.globalToLocal(self.Pp, self.localPp)
#         self.da1.globalToLocal(self.Oh, self.localOh)
#         self.da1.globalToLocal(self.Ph, self.localPh)
#         
#         cdef double[:,:] b  = self.da1.getVecArray(B)[...]
#         cdef double[:,:] Pp = self.da1.getVecArray(self.localPp)[...]
#         cdef double[:,:] Ph = self.da1.getVecArray(self.localPh)[...]
#         cdef double[:,:] Oh = self.da1.getVecArray(self.localOh)[...]
#         
#         cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
#         
#         
#         for i in range(xs, xe):
#             ix = i-xs+1
#             iy = i-xs
#             
#             for j in range(ys, ye):
#                 jx = j-ys+1
#                 jy = j-ys
#                 
#                 jpp = (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * (Oh[ix,   jx+1] - Oh[ix,   jx-1]) \
#                     - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * (Oh[ix+1, jx  ] - Oh[ix-1, jx  ])
#                 
#                 jpc = P_ave[ix+1, jx  ] * (Oh[ix+1, jx+1] - Oh[ix+1, jx-1]) \
#                     - P_ave[ix-1, jx  ] * (Oh[ix-1, jx+1] - Oh[ix-1, jx-1]) \
#                     - P_ave[ix,   jx+1] * (Oh[ix+1, jx+1] - Oh[ix-1, jx+1]) \
#                     + P_ave[ix,   jx-1] * (Oh[ix+1, jx-1] - Oh[ix-1, jx-1])
#                 
#                 jcp = P_ave[ix+1, jx+1] * (Oh[ix,   jx+1] - Oh[ix+1, jx  ]) \
#                     - P_ave[ix-1, jx-1] * (Oh[ix-1, jx  ] - Oh[ix,   jx-1]) \
#                     - P_ave[ix-1, jx+1] * (Oh[ix,   jx+1] - Oh[ix-1, jx  ]) \
#                     + P_ave[ix+1, jx-1] * (Oh[ix+1, jx  ] - Oh[ix,   jx-1])
#     
#                 b[iy, jy] = Oh[ix,jx] - 0.5 * (jpp + jpc + jcp) * self.arakawa_factor * self.ht
        

    @cython.boundscheck(False)
    cpdef function(self, Vec O, Vec F):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        cdef double jpp, jpc, jcp
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(O,       self.localOp)
        self.da1.globalToLocal(self.Oh, self.localOh)
        self.da1.globalToLocal(self.Pp, self.localPp)
        self.da1.globalToLocal(self.Ph, self.localPh)
        
        cdef np.ndarray[np.float64_t, ndim=2] Op = self.da1.getVecArray(self.localOp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Oh = self.da1.getVecArray(self.localOh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.localPp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = self.da1.getVecArray(self.localPh)[...]
        
        cdef double[:,:] f     = self.da1.getVecArray(F)[...]
        cdef double[:,:] O_ave = 0.5 * (Op + Oh)
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        
        
        for i in range(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                jpp = (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * (O_ave[ix,   jx+1] - O_ave[ix,   jx-1]) \
                    - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * (O_ave[ix+1, jx  ] - O_ave[ix-1, jx  ])
                
                jpc = P_ave[ix+1, jx  ] * (O_ave[ix+1, jx+1] - O_ave[ix+1, jx-1]) \
                    - P_ave[ix-1, jx  ] * (O_ave[ix-1, jx+1] - O_ave[ix-1, jx-1]) \
                    - P_ave[ix,   jx+1] * (O_ave[ix+1, jx+1] - O_ave[ix-1, jx+1]) \
                    + P_ave[ix,   jx-1] * (O_ave[ix+1, jx-1] - O_ave[ix-1, jx-1])
                
                jcp = P_ave[ix+1, jx+1] * (O_ave[ix,   jx+1] - O_ave[ix+1, jx  ]) \
                    - P_ave[ix-1, jx-1] * (O_ave[ix-1, jx  ] - O_ave[ix,   jx-1]) \
                    - P_ave[ix-1, jx+1] * (O_ave[ix,   jx+1] - O_ave[ix-1, jx  ]) \
                    + P_ave[ix+1, jx-1] * (O_ave[ix+1, jx  ] - O_ave[ix,   jx-1])
                
                f[iy, jy] = (Op[ix,jx] - Oh[ix,jx]) + (jpp + jpc + jcp) * self.arakawa_factor * self.ht
    
    
    @cython.boundscheck(False)
    cpdef explicit(self, Vec O, Vec P, Vec F):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        cdef double jpp, jpc, jcp
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(O, self.localOp)
        self.da1.globalToLocal(P, self.localPp)
        
        cdef double[:,:] Op = self.da1.getVecArray(self.localOp)[...]
        cdef double[:,:] Pp = self.da1.getVecArray(self.localPp)[...]
        cdef double[:,:] f  = self.da1.getVecArray(F)[...]
        
        
        for i in range(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+1
                jy = j-ys
                
                jpp = (Pp[ix+1, jx  ] - Pp[ix-1, jx  ]) * (Op[ix,   jx+1] - Op[ix,   jx-1]) \
                    - (Pp[ix,   jx+1] - Pp[ix,   jx-1]) * (Op[ix+1, jx  ] - Op[ix-1, jx  ])
                
                jpc = Pp[ix+1, jx  ] * (Op[ix+1, jx+1] - Op[ix+1, jx-1]) \
                    - Pp[ix-1, jx  ] * (Op[ix-1, jx+1] - Op[ix-1, jx-1]) \
                    - Pp[ix,   jx+1] * (Op[ix+1, jx+1] - Op[ix-1, jx+1]) \
                    + Pp[ix,   jx-1] * (Op[ix+1, jx-1] - Op[ix-1, jx-1])
                
                jcp = Pp[ix+1, jx+1] * (Op[ix,   jx+1] - Op[ix+1, jx  ]) \
                    - Pp[ix-1, jx-1] * (Op[ix-1, jx  ] - Op[ix,   jx-1]) \
                    - Pp[ix-1, jx+1] * (Op[ix,   jx+1] - Op[ix-1, jx  ]) \
                    + Pp[ix+1, jx-1] * (Op[ix+1, jx  ] - Op[ix,   jx-1])
                
                f[iy, jy] = (jpp + jpc + jcp) * self.arakawa_factor
    
    
    cpdef mult(self, Mat mat, Vec O, Vec Y):
        self.matrix_mult(O, Y)
        
    cpdef matrix_snes_mult(self, SNES snes, Vec O, Vec Y):
        self.matrix_mult(O, Y)
        
    cpdef function_snes_mult(self, SNES snes, Vec O, Vec Y):
        self.function(O, Y)
