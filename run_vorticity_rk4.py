'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

import numpy as np

from vorticity.config import Config

from vorticity.solvers.PETScDerivatives  import PETScDerivatives
from vorticity.solvers.PETScPoisson      import PETScPoisson
from vorticity.solvers.PETScVorticity    import PETScVorticity


class petscMHD2D(object):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # load run config file
        cfg = Config(cfgfile)
        
        # run in linear or nonlinear mode
        self.nonlinear = cfg['solver']['nonlinear']
        
        # timestep setup
        self.ht    = cfg['grid']['ht']              # timestep size
        self.nt    = cfg['grid']['nt']              # number of timesteps
        self.nsave = cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        self.nx   = cfg['grid']['nx']                    # number of points in x
        self.ny   = cfg['grid']['ny']                    # number of points in y
        
        Lx   = cfg['grid']['Lx']                    # spatial domain in x
        x1   = cfg['grid']['x1']                    # 
        x2   = cfg['grid']['x2']                    # 
        
        Ly   = cfg['grid']['Ly']                    # spatial domain in y
        y1   = cfg['grid']['y1']                    # 
        y2   = cfg['grid']['y2']                    # 
        
        if x1 != x2:
            Lx = x2-x1
        else:
            x1 = 0.0
            x2 = Lx
        
        if y1 != y2:
            Ly = y2-y1
        else:
            y1 = 0.0
            y2 = Ly
        
        
        self.hx = Lx / self.nx                       # gridstep size in x
        self.hy = Ly / self.ny                       # gridstep size in y
        
        
        self.time = PETSc.Vec().createMPI(1, PETSc.DECIDE, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol',   cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', cfg['solver']['petsc_ksp_max_iter'])

#         OptDB.setValue('ksp_monitor', '')
#         OptDB.setValue('log_info', '')
#         OptDB.setValue('log_summary', '')
        
        self.snes_rtol = cfg['solver']['petsc_snes_rtol']
        self.snes_atol = cfg['solver']['petsc_snes_atol']
        self.snes_max_iter = cfg['solver']['petsc_snes_max_iter']
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("")
            print("Config File: %s" % cfgfile)
            print("Output File: %s" % cfg['io']['hdf5_output'])
            print("")
            if self.nonlinear:
                print("nonlinear mode")
            else:
                print("linear mode")
            print("")
            print("nt = %i" % (self.nt))
            print("nx = %i" % (self.nx))
            print("ny = %i" % (self.ny))
            print("")
            print("ht = %e" % (self.ht))
            print("hx = %e" % (self.hx))
            print("hy = %e" % (self.hy))
            print("")
            print("xMin = %+12.6e" % (x1))
            print("xMax = %+12.6e" % (x2))
            print("yMin = %+12.6e" % (y1))
            print("yMax = %+12.6e" % (y2))
            print("")
            print("")
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[self.nx, self.ny],
                                    proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
                                    boundary_type=('periodic', 'periodic'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        # create DA for x grid
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[self.nx],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[self.ny],
                                    proc_sizes=[PETSc.DECIDE],
                                    boundary_type=('periodic'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=x1, xmax=x2,
                                       ymin=y1, ymax=y2)
        
        self.dax.setUniformCoordinates(xmin=x1, xmax=x2)
        self.day.setUniformCoordinates(xmin=y1, xmax=y2)
        
        
        # create solution, RHS and function vectors
        self.Pb = self.da1.createGlobalVec()
        self.Ob = self.da1.createGlobalVec()
        self.Pf = self.da1.createGlobalVec()
        self.Of = self.da1.createGlobalVec()

        # create nullspace vector for Poisson equation
        self.Pn = self.da1.createGlobalVec()
        
        # create vectors for vorticity, streaming function and velocity
        self.P  = self.da1.createGlobalVec()        # streaming function phi
        self.O  = self.da1.createGlobalVec()        # vorticity          omega
        self.Vx = self.da1.createGlobalVec()        # velocity, x-component
        self.Vy = self.da1.createGlobalVec()        # velocity, y-component
        
        self.O0 = self.da1.createGlobalVec()        # vorticity previous step
        self.O1 = self.da1.createGlobalVec()        # vorticity RK stage 1
        self.O2 = self.da1.createGlobalVec()        # vorticity RK stage 2
        self.O3 = self.da1.createGlobalVec()        # vorticity RK stage 3
        self.O4 = self.da1.createGlobalVec()        # vorticity RK stage 4

        # set variable names
        self.P.setName('P')
        self.O.setName('O')
        self.Vx.setName('Vx')
        self.Vy.setName('Vy')
        
        
        # create vorticity scheme object
        self.vorticity = PETScVorticity(self.da1, self.nx, self.ny, self.ht, self.hx, self.hy)
        
        self.vorticity_mf = PETSc.Mat().createPython([self.O.getSizes(), self.Ob.getSizes()], comm=PETSc.COMM_WORLD)
        self.vorticity_mf.setPythonContext(self.vorticity)
        self.vorticity_mf.setUp()
        
        
        # create Poisson matrix and solver
        if self.nonlinear:
            self.poisson = PETScPoisson(self.da1, self.nx, self.ny, self.hx, self.hy)
            
            self.poisson_mat = self.da1.createMat()
            self.poisson_mat.setOption(PETSc.Mat().Option.NEW_NONZERO_ALLOCATION_ERR, False)
            self.poisson_mat.setUp()
        
            self.poisson_nullspace = PETSc.NullSpace().create(constant=True)
             self.poisson_mat.setNullSpace(self.poisson_nullspace)
        
            self.poisson_ksp = PETSc.KSP().create()
            self.poisson_ksp.setFromOptions()
            self.poisson_ksp.setTolerances(rtol=1E-13)
            self.poisson_ksp.setOperators(self.poisson_mat)
            self.poisson_ksp.setType('preonly')
            self.poisson_ksp.getPC().setType('lu')
            self.poisson_ksp.getPC().setFactorSolverPackage('mumps')
#            self.poisson_ksp.setNullSpace(self.poisson_nullspace)
        
            self.poisson.formMat(self.poisson_mat)
        
        
        # create derivatives object
        self.derivatives = PETScDerivatives(self.da1, self.nx, self.ny, self.ht, self.hx, self.hy)
        
        # get coordinate vectors
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
         
        # save x coordinate arrays
        scatter, xVec = PETSc.Scatter.toAll(coords_x)
        
        scatter.begin(coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
          
        xGrid = xVec.getValues(range(0, self.nx)).copy()
          
        scatter.destroy()
        xVec.destroy()
          
        # save y coordinate arrays
        scatter, yVec = PETSc.Scatter.toAll(coords_y)
        
        scatter.begin(coords_y, yVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_y, yVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                    
        yGrid = yVec.getValues(range(0, self.ny)).copy()
          
        scatter.destroy()
        yVec.destroy()

        # set initial data
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        O_arr = self.da1.getVecArray(self.O)
        
        init_data = __import__("examples." + cfg['initial_data']['python'], globals(), locals(), ['vorticity'], 0)
        
#         txGrid, tyGrid      = np.meshgrid(xGrid[xs:xe], yGrid[ys:ye])
#         O_arr[xs:xe, ys:ye] = init_data.vorticity(txGrid, tyGrid, Lx, Ly)

        for i in range(xs, xe):
            for j in range(ys, ye):
                O_arr[i,j] = init_data.vorticity(xGrid[i], yGrid[j], Lx, Ly)
        
        
        if self.nonlinear:
            self.compute_potential()
        
        else:
            # compute streaming function from input
            P_arr = self.da1.getVecArray(self.P)
            
            init_data = __import__("examples." + cfg['initial_data']['python'], globals(), locals(), ['streaming_function'], 0)
            
#             P_arr[xs:xe, ys:ye] = init_data.streaming_function(txGrid, tyGrid, Lx, Ly)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    P_arr[i,j] = init_data.streaming_function(xGrid[i], yGrid[j], Lx, Ly)
        
        
        # set streaming function in vorticity solver
        self.vorticity.updateStreamingFunction(self.P)
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.ViewerHDF5().create(cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.pushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        coords_x.setName('x')
        coords_y.setName('y')

        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        # write initial data to hdf5 file
        self.save_to_hdf5(0)
        
        
    
    def __del__(self):
        self.hdf5_viewer.destroy()
        self.poisson_ksp.destroy()
        self.poisson_mat.destroy()
        self.vorticity_mf.destroy()
    
    
    def run(self):
        
        # loop in time
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
#                 if self.nonlinear: print("")
                print("it = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            # copy vorticity from previous timestep
            self.O.copy(self.O0)
            
            # RK4 stage 1
            self.vorticity.explicit(self.O, self.P, self.O1)
            
            # RK4 stage 2
            self.O.waxpy(0.5 * self.ht, self.O1, self.O0)
            if self.nonlinear:
                self.compute_potential()
            self.vorticity.explicit(self.O, self.P, self.O2)
            
            # RK4 stage 3
            self.O.waxpy(0.5 * self.ht, self.O2, self.O0)
            if self.nonlinear:
                self.compute_potential()
            self.vorticity.explicit(self.O, self.P, self.O3)
            
            # RK4 stage 4
            self.O.waxpy(1.0 * self.ht, self.O2, self.O0)
            if self.nonlinear:
                self.compute_potential()
            self.vorticity.explicit(self.O, self.P, self.O4)
            
            # RK4 final result
            self.O0.copy(self.O)
            self.O.axpy(self.ht * 1. / 6., self.O1)
            self.O.axpy(self.ht * 2. / 6., self.O2)
            self.O.axpy(self.ht * 2. / 6., self.O3)
            self.O.axpy(self.ht * 1. / 6., self.O4)
            
            # update streaming function
            if self.nonlinear:
                self.compute_potential()
            
            # save to hdf5 file
            self.save_to_hdf5(itime)
            
    
    def compute_potential(self, output=False):
        # build RHS and solve Poisson equation
        self.poisson.updateVorticity(self.O)
        self.poisson.formRHS(self.Pb)
        self.poisson_nullspace.remove(self.Pb)
        self.poisson_ksp.solve(self.Pb, self.P)
        
        # display some solver information
        if output:
            self.poisson.function(self.P, self.Pf)
            pnorm = self.Pf.norm()
        
            if PETSc.COMM_WORLD.getRank() == 0:
                print("   Poisson   Solver:    %5i iterations,   residual = %24.16E " % (self.poisson_ksp.getIterationNumber(), pnorm) )
    
    
    def save_to_hdf5(self, timestep):
        if timestep % self.nsave == 0 or timestep == self.nt + 1:
            # calculate V field
            self.derivatives.dy(self.P, self.Vx)
            self.derivatives.dx(self.P, self.Vy)
            
            # save timestep
            self.hdf5_viewer.setTimestep(timestep // self.nsave)
            self.hdf5_viewer(self.time)
            
            self.hdf5_viewer(self.P)
            self.hdf5_viewer(self.O)
            
            self.hdf5_viewer(self.Vx)
            self.hdf5_viewer(self.Vy)
            
        
    def updatePoissonJacobian(self, snes, X, J, P):
        pass    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vorticity Solver in 2D (explicit RK4 time stepping)')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscMHD2D(args.runfile)
    petscvp.run()
    
