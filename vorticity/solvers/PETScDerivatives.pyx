'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec


cdef class PETScDerivatives(object):
    '''
    Cython Implementation of MHD Discretisation
    '''
    
    
    def __cinit__(self, object da1,
                  np.uint64_t  nx, np.uint64_t  ny,
                  np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        
        # distributed arrays
        self.da1 = da1
        
        
        # create local vectors
        self.localX = da1.createLocalVec()
        
        
        # create temporary numpy array
        (xs, xe), (ys, ye) = self.da1.getRanges()
        self.ty = np.empty((xe-xs, ye-ys))
        
    
    
    @cython.boundscheck(False)
    cdef np.float64_t arakawa(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] h,
                                    np.uint64_t i, np.uint64_t j):
        '''
        MHD Derivative: Arakawa Bracket
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + x[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1])
                     + x[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ])
                     + x[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1])
                     + x[i-1, j  ] * (h[i,   j+1] - h[i,   j-1])
                     + x[i+1, j-1] * (h[i,   j-1] - h[i+1, j  ])
                     + x[i+1, j+1] * (h[i+1, j  ] - h[i,   j+1])
                     + x[i+1, j  ] * (h[i+1, j-1] - h[i+1, j+1])
                     + x[i+1, j  ] * (h[i,   j-1] - h[i,   j+1])
                     + x[i,   j-1] * (h[i-1, j-1] - h[i+1, j-1])
                     + x[i,   j-1] * (h[i-1, j  ] - h[i+1, j  ])
                     + x[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1])
                     + x[i,   j+1] * (h[i+1, j  ] - h[i-1, j  ])        
                 ) * self.hx_inv * self.hy_inv / 12.
 
        return result
    
    
    @cython.boundscheck(False)
    cdef np.float64_t laplace(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.uint64_t i, np.uint64_t j):
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j] \
                   - 2. * x[i,   j] \
                   + 1. * x[i+1, j] \
                 ) * self.hx_inv**2 \
               + ( \
                   + 1. * x[i, j-1] \
                   - 2. * x[i, j  ] \
                   + 1. * x[i, j+1] \
                 ) * self.hy_inv**2
 
        return result
    

    @cython.boundscheck(False)
    cpdef laplace_vec(self, Vec X, Vec D, np.float64_t sign):
    
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        d = self.da1.getVecArray(D)
        
        cdef np.ndarray[np.float64_t, ndim=2] tx = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] td = self.ty
        
        for j in range(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in range(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                td[iy, jy] = sign * self.laplace(tx, ix, jx)
    
        d[xs:xe, ys:ye] = td[:,:]
        
    
    @cython.boundscheck(False)
    cpdef np.float64_t dx(self, Vec X, Vec D):
    
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        d = self.da1.getVecArray(D)
        
        cdef np.ndarray[np.float64_t, ndim=2] tx = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] td = self.ty
        
        for j in range(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in range(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                td[iy, jy] = + 0.25 * ( \
                                        + 1. * (tx[ix+1, jx-1] - tx[ix-1, jx-1]) \
                                        + 2. * (tx[ix+1, jx  ] - tx[ix-1, jx  ]) \
                                        + 1. * (tx[ix+1, jx+1] - tx[ix-1, jx+1]) \
                                      ) * 0.5 * self.hx_inv
    
        d[xs:xe, ys:ye] = td[:,:]
        
    
    @cython.boundscheck(False)
    cpdef np.float64_t dy(self, Vec X, Vec D):
    
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        x = self.da1.getVecArray(self.localX)
        d = self.da1.getVecArray(D)
        
        cdef np.ndarray[np.float64_t, ndim=2] tx = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] td = self.ty
        
        for j in range(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in range(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                td[iy, jy] = - 0.25 * ( \
                                        + 1. * (tx[ix-1, jx+1] - tx[ix-1, jx-1]) \
                                        + 2. * (tx[ix,   jx+1] - tx[ix,   jx-1]) \
                                        + 1. * (tx[ix+1, jx+1] - tx[ix+1, jx-1]) \
                                      ) * 0.5 * self.hy_inv
    
        d[xs:xe, ys:ye] = td[:,:]
