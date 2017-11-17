'''
Created on Jul 2, 2012

@author: mkraus
'''

import h5py
import numpy as np


class Diagnostics(object):
    '''
    classdocs
    '''


    def __init__(self, hdf5_file):
        '''
        Constructor
        '''

        self.hdf5 = h5py.File(hdf5_file, 'r')
        
        assert self.hdf5 != None
        
        
        self.tGrid = self.hdf5['t'][:,0,0]
        self.xGrid = self.hdf5['x'][:]
        self.yGrid = self.hdf5['y'][:]
        
        self.nt = len(self.tGrid)-1
        
        if self.nt == 1:
            self.ht = 0.
        else:
            self.ht = self.tGrid[2] - self.tGrid[1]
        
        self.Lx = (self.xGrid[-1] - self.xGrid[0]) + (self.xGrid[1] - self.xGrid[0])
        self.Ly = (self.yGrid[-1] - self.yGrid[0]) + (self.yGrid[1] - self.yGrid[0])
        
        self.nx = len(self.xGrid)
        self.ny = len(self.yGrid)
        self.n  = self.nx * self.ny
        
        self.hx = self.xGrid[1] - self.xGrid[0]
        self.hy = self.yGrid[1] - self.yGrid[0]
        
        self.tMin = self.tGrid[ 1]
        self.tMax = self.tGrid[-1]
        self.xMin = self.xGrid[ 0]
        self.xMax = self.xGrid[-1]
        self.yMin = self.yGrid[ 0]
        self.yMax = self.yGrid[-1]
        
        
        print("nt = %i (%i)" % (self.nt, len(self.tGrid)) )
        print("nx = %i" % (self.nx))
        print("ny = %i" % (self.ny))
        print
        print("ht = %f" % (self.ht))
        print("hx = %f" % (self.hx))
        print("hy = %f" % (self.hy))
        print
        print("tGrid:")
        print(self.tGrid)
        print
        print("xGrid:")
        print(self.xGrid)
        print
        print("yGrid:")
        print(self.yGrid)
        print
        
        
        self.P  = np.zeros((self.nx, self.ny))
        self.O  = np.zeros((self.nx, self.ny))
        
        self.Vx = np.zeros((self.nx, self.ny))
        self.Vy = np.zeros((self.nx, self.ny))
        
        
        self.circulation = 0.0
        self.enstrophy   = 0.0
        self.energy      = 0.0
        self.momentum_x  = 0.0
        self.momentum_y  = 0.0
        
        self.circulation_0 = 0.0
        self.enstrophy_0   = 0.0
        self.energy_0      = 0.0
        self.momentum_x_0  = 0.0
        self.momentum_y_0  = 0.0
        
        self.circulation_error = 0.0
        self.enstrophy_error   = 0.0
        self.energy_error      = 0.0
        self.momentum_error    = 0.0
        self.momentum_x_error  = 0.0
        self.momentum_y_error  = 0.0
        
        
        self.read_from_hdf5(0)
        self.update_invariants(0)
        
        
        
    def read_from_hdf5(self, iTime):
        self.P  = self.hdf5['P' ][iTime,:,:].T
        self.O  = self.hdf5['O' ][iTime,:,:].T
        self.Vx = self.hdf5['Vx'][iTime,:,:].T
        self.Vy = self.hdf5['Vy'][iTime,:,:].T
        
    
    def update_invariants(self, iTime):
        
        self.circulation = 0.0
        self.enstrophy   = 0.0
        self.energy      = 0.0
        self.momentum_x  = 0.0
        self.momentum_y  = 0.0
        
        for ix in range(self.nx):
            ixp = (ix+1) % self.nx
            
            for iy in range(self.ny):
                iyp = (iy+1) % self.ny
                
                self.circulation += self.O[ix,iy]
                self.enstrophy   += self.O[ix,iy]**2
                self.momentum_x  += self.O[ix,iy] * self.yGrid[iy]
                self.momentum_y  += self.O[ix,iy] * self.xGrid[ix]
                self.energy      += self.P[ix,iy] * self.O[ix,iy] * self.hx * self.hy
#                 self.energy      += ( self.P[ixp,iy] - self.P[ix,iy] )**2 * self.hy / self.hx
#                 self.energy      += ( self.P[ix,iyp] - self.P[ix,iy] )**2 * self.hx / self.hy

        self.circulation *= self.hx * self.hy
        self.enstrophy   *= self.hx * self.hy * 0.5
        self.energy      *= - 0.5
        self.momentum_x  *= self.hx * self.hy
        self.momentum_y  *= self.hx * self.hy
        
        self.momentum = self.momentum_x + self.momentum_y
    
        if iTime == 0:
            self.circulation_0 = self.circulation
            self.enstrophy_0   = self.enstrophy
            self.energy_0      = self.energy
            self.momentum_0    = self.momentum_x + self.momentum_y
            self.momentum_x_0  = self.momentum_x
            self.momentum_y_0  = self.momentum_y

            self.circulation_error = 0.0
            self.enstrophy_error   = 0.0
            self.energy_error      = 0.0
            self.momentum_error    = 0.0
            self.momentum_x_error  = 0.0
            self.momentum_y_error  = 0.0
        
        else:
            self.circulation_error = (self.circulation - self.circulation_0) / self.circulation_0
            self.enstrophy_error   = (self.enstrophy   - self.enstrophy_0)   / self.enstrophy_0
            self.energy_error      = (self.energy      - self.energy_0)      / self.energy_0
            self.momentum_error    = (self.momentum    - self.momentum_0)    / self.momentum_0
            self.momentum_x_error  = (self.momentum_x  - self.momentum_x_0)  / self.momentum_x_0
            self.momentum_y_error  = (self.momentum_y  - self.momentum_y_0)  / self.momentum_y_0
        
    
