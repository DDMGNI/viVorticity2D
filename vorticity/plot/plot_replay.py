'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
from matplotlib.ticker import ScalarFormatter


class PlotVorticity2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, nTime=0, nPlot=1, xmin=0., xmax=0., ymin=0., ymax=0., output=False, contours=False):
        '''
        Constructor
        '''
        
        self.eps = 1E-10
        
        self.nrows = 2
        self.ncols = 4
        
        if nTime > 0 and nTime < diagnostics.nt:
            self.nTime = nTime
        else:
            self.nTime = diagnostics.nt
        
        self.iTime = -1
        self.nPlot = nPlot
        
        self.diagnostics = diagnostics
        
        self.output   = output
        self.contours = contours
 
        self.prefix = "viVorticity2D"

        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        
        self.x = np.zeros(diagnostics.nx+1)
        self.y = np.zeros(diagnostics.ny+1)
        
        self.xpc = np.zeros(diagnostics.nx+2)
        self.ypc = np.zeros(diagnostics.ny+2)
        
        self.x[0:-1] = self.diagnostics.xGrid
        self.x[  -1] = self.x[-2] + self.diagnostics.hx
        
        self.y[0:-1] = self.diagnostics.yGrid
        self.y[  -1] = self.y[-2] + self.diagnostics.hy
        
        self.xpc[0:-1] = self.x
        self.xpc[  -1] = self.xpc[-2] + self.diagnostics.hx
        self.xpc -= 0.5 * self.diagnostics.hx
        
        self.ypc[0:-1] = self.y
        self.ypc[  -1] = self.ypc[-2] + self.diagnostics.hy
        self.ypc -= 0.5 * self.diagnostics.hy
         
        self.circulation  = np.zeros(self.nTime+1)
        self.enstrophy    = np.zeros(self.nTime+1)
        self.energy       = np.zeros(self.nTime+1)
        self.momentum_x   = np.zeros(self.nTime+1)
        self.momentum_y   = np.zeros(self.nTime+1)
        
        self.P  = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.O  = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vx = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.Vy = np.zeros((diagnostics.nx+1, diagnostics.ny+1))

        # add data for zero timepoint
        self.add_timepoint()
        self.read_data()
        self.update_boundaries()
        
        
        # set up plots
        self.figs = {}
        self.axes = {}
        self.lins = {}
        self.vecs = {}
        self.pcms = {}
        
        # set up figure/window size
        self.figure = plt.figure(num=0, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.5, wspace=0.25)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = %1.2f' % (diagnostics.tGrid[self.iTime]), horizontalalignment='center') 
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)

        # create subplots
        gs = gridspec.GridSpec(5, 5)
        self.gs = gs
        
        self.axes["P"]  = plt.subplot(gs[0:4,0:2])
        self.axes["O"]  = plt.subplot(gs[0:4,2:4])
        self.axes["Vx"] = plt.subplot(gs[0:2,4])
        self.axes["Vy"] = plt.subplot(gs[2:4,4])
        self.axes["C"]  = plt.subplot(gs[4,0])
        self.axes["W"]  = plt.subplot(gs[4,1])
        self.axes["E"]  = plt.subplot(gs[4,2])
        self.axes["Mx"] = plt.subplot(gs[4,3])
        self.axes["My"] = plt.subplot(gs[4,4])
        
        # color meshes for streaming function, vorticity and velocity
        self.pcms["P" ] = self.axes["P" ].pcolormesh(self.xpc, self.ypc, self.P.T)
        self.pcms["O" ] = self.axes["O" ].pcolormesh(self.xpc, self.ypc, self.O.T)
        self.pcms["Vx"] = self.axes["Vx"].pcolormesh(self.xpc, self.ypc, self.Vx.T, norm=self.Vnorm)
        self.pcms["Vy"] = self.axes["Vy"].pcolormesh(self.xpc, self.ypc, self.Vy.T, norm=self.Vnorm)
        
        if self.contours:
            matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
            self.axes["O"].contour(self.x, self.y, self.P.T, 7, colors='white')
        
        self.axes["P" ].set_title('$\phi (x,y)$')
        self.axes["O" ].set_title('$\omega (x,y)$')
        self.axes["Vx"].set_title('$u (x,y)$')
        self.axes["Vy"].set_title('$v (x,y)$')
        
        self.axes["P" ].set_xlim((self.x[0], self.x[-1]))
        self.axes["P" ].set_ylim((self.y[0], self.y[-1])) 
        self.axes["O" ].set_xlim((self.x[0], self.x[-1]))
        self.axes["O" ].set_ylim((self.y[0], self.y[-1]))
        self.axes["Vx"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vx"].set_ylim((self.y[0], self.y[-1])) 
        self.axes["Vy"].set_xlim((self.x[0], self.x[-1]))
        self.axes["Vy"].set_ylim((self.y[0], self.y[-1])) 
        
        # time traces
        self.lins["C"],  = self.axes["C" ].plot(self.diagnostics.tGrid[0], self.circulation[0])
        self.lins["W"],  = self.axes["W" ].plot(self.diagnostics.tGrid[0], self.enstrophy  [0])
        self.lins["E"],  = self.axes["E" ].plot(self.diagnostics.tGrid[0], self.energy     [0])
        self.lins["Mx"], = self.axes["Mx"].plot(self.diagnostics.tGrid[0], self.momentum_x [0])
        self.lins["My"], = self.axes["My"].plot(self.diagnostics.tGrid[0], self.momentum_y [0])
        
        self.axes["W"].set_title('$\Delta W (t)$')
        self.axes["E"].set_title('$\Delta E (t)$')
        
        if np.abs(self.diagnostics.circulation_0) < self.eps:
            self.axes["C"].set_title('$C (t)$')
        else:
            self.axes["C"].set_title('$\Delta C (t)$')
        
        if np.abs(self.diagnostics.momentum_x_0) < self.eps:
            self.axes["Mx"].set_title('$M_x (t)$')
        else:
            self.axes["Mx"].set_title('$\Delta M_x (t)$')
        
        if np.abs(self.diagnostics.momentum_y_0) < self.eps:
            self.axes["My"].set_title('$M_y (t)$')
        else:
            self.axes["My"].set_title('$\Delta M_y (t)$')
        
        self.axes["C" ].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
        self.axes["W" ].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
        self.axes["E" ].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
        self.axes["Mx"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
        self.axes["My"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
        
        self.axes["C" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["W" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["E" ].yaxis.set_major_formatter(majorFormatter)
        self.axes["Mx"].yaxis.set_major_formatter(majorFormatter)
        self.axes["My"].yaxis.set_major_formatter(majorFormatter)
        
        
        if self.output:
            self.figs["Os"] = plt.figure(num=1, figsize=(10,10))
            self.axes["Os"] = self.figs["Os"].add_subplot(1,1,1)
            self.pcms["Os"] = self.axes["Os"].pcolormesh(self.xpc, self.ypc, self.O.T)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.09, top=0.94, wspace=0.1, hspace=0.2)
            
            if self.contours:
                self.axes["Os"].contour(self.x, self.y, self.P.T, 10, colors='white')
            
            self.figs["Ps"] = plt.figure(num=2, figsize=(10,10))
            self.axes["Ps"] = self.figs["Ps"].add_subplot(1,1,1)
            self.pcms["Ps"] = self.axes["Ps"].pcolormesh(self.xpc, self.ypc, self.P.T)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.09, top=0.94, wspace=0.1, hspace=0.2)

            self.figs["Cs"] = plt.figure(num=3, figsize=(16,4))
            self.axes["Cs"] = self.figs["Cs"].add_subplot(1,1,1)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

            self.figs["Ws"] = plt.figure(num=4, figsize=(16,4))
            self.axes["Ws"] = self.figs["Ws"].add_subplot(1,1,1)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

            self.figs["Es"] = plt.figure(num=5, figsize=(16,4))
            self.axes["Es"] = self.figs["Es"].add_subplot(1,1,1)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

            self.figs["TT"] = plt.figure(num=6, figsize=(16,10))
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.4)

            tt_gs = gridspec.GridSpec(3, 1)
            self.axes["Ct"] = plt.subplot(tt_gs[0,0])
            self.axes["Wt"] = plt.subplot(tt_gs[1,0])
            self.axes["Et"] = plt.subplot(tt_gs[2,0])

            self.lins["Cs"], = self.axes["Cs"].plot(self.diagnostics.tGrid[0], self.circulation [0])
            self.lins["Ws"], = self.axes["Ws"].plot(self.diagnostics.tGrid[0], self.enstrophy   [0])
            self.lins["Es"], = self.axes["Es"].plot(self.diagnostics.tGrid[0], self.energy      [0])
            self.lins["Ct"], = self.axes["Ct"].plot(self.diagnostics.tGrid[0], self.circulation [0])
            self.lins["Wt"], = self.axes["Wt"].plot(self.diagnostics.tGrid[0], self.enstrophy   [0])
            self.lins["Et"], = self.axes["Et"].plot(self.diagnostics.tGrid[0], self.energy      [0])

            self.axes ["Ws"].set_title('Total Enstrophy Error $\Delta W (t)$', fontsize=24)
            self.axes ["Wt"].set_title('Total Enstrophy Error $\Delta W (t)$', fontsize=24)
            self.axes ["Es"].set_title('Total Energy Error $\Delta E (t)$',    fontsize=24)
            self.axes ["Et"].set_title('Total Energy Error $\Delta E (t)$',    fontsize=24)
            
            if np.abs(self.diagnostics.circulation_0) < self.eps:
                self.axes ["Cs"].set_title('Total Circulation $C (t)$', fontsize=24)
                self.axes ["Ct"].set_title('Total Circulation $C (t)$', fontsize=24)
            else:
                self.axes ["Cs"].set_title('Total Circulation Error $\Delta C (t)$', fontsize=24)
                self.axes ["Ct"].set_title('Total Circulation Error $\Delta C (t)$', fontsize=24)
            
            self.axes["Os"].title.set_y(1.01)
            self.axes["Ps"].title.set_y(1.01)
            self.axes["Cs"].title.set_y(1.02)
            self.axes["Ws"].title.set_y(1.02)
            self.axes["Es"].title.set_y(1.02)
            self.axes["Ct"].title.set_y(1.02)
            self.axes["Wt"].title.set_y(1.02)
            self.axes["Et"].title.set_y(1.02)
            
            self.axes["Ps"].set_xlim((self.x[0], self.x[-1]))
            self.axes["Ps"].set_ylim((self.y[0], self.y[-1])) 
            self.axes["Os"].set_xlim((self.x[0], self.x[-1]))
            self.axes["Os"].set_ylim((self.y[0], self.y[-1]))
            
            self.axes["Cs"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
            self.axes["Ws"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
            self.axes["Es"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
            self.axes["Ct"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
            self.axes["Wt"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
            self.axes["Et"].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime]))
            
            self.axes["Cs"].yaxis.set_major_formatter(majorFormatter)
            self.axes["Ws"].yaxis.set_major_formatter(majorFormatter)
            self.axes["Es"].yaxis.set_major_formatter(majorFormatter)
            self.axes["Ct"].yaxis.set_major_formatter(majorFormatter)
            self.axes["Wt"].yaxis.set_major_formatter(majorFormatter)
            self.axes["Et"].yaxis.set_major_formatter(majorFormatter)
        
            self.axes["Cs"].yaxis.set_label_coords(-0.07, 0.5)
            self.axes["Ws"].yaxis.set_label_coords(-0.07, 0.5)
            self.axes["Es"].yaxis.set_label_coords(-0.07, 0.5)
            self.axes["Ct"].yaxis.set_label_coords(-0.07, 0.5)
            self.axes["Wt"].yaxis.set_label_coords(-0.07, 0.5)
            self.axes["Et"].yaxis.set_label_coords(-0.07, 0.5)
            
            self.axes["Os"].set_xlabel('$x$', labelpad=15, fontsize=22)
            self.axes["Ps"].set_xlabel('$x$', labelpad=15, fontsize=22)
            self.axes["Cs"].set_xlabel('$t$', labelpad=15, fontsize=22)
            self.axes["Ws"].set_xlabel('$t$', labelpad=15, fontsize=22)
            self.axes["Es"].set_xlabel('$t$', labelpad=15, fontsize=22)
            self.axes["Et"].set_xlabel('$t$', labelpad=15, fontsize=22)
        
            plt.setp(self.axes["Ct"].get_xticklabels(), visible=False)
            plt.setp(self.axes["Wt"].get_xticklabels(), visible=False)

            self.axes["Os"].set_ylabel('$y$', labelpad=15, fontsize=22)
            self.axes["Ps"].set_ylabel('$y$', labelpad=15, fontsize=22)
            self.axes["Ws"].set_ylabel('$(W (t) - W_0) / W_0$', fontsize=22)
            self.axes["Wt"].set_ylabel('$(W (t) - W_0) / W_0$', fontsize=22)
            self.axes["Es"].set_ylabel('$(E (t) - E_0) / E_0$', fontsize=22)
            self.axes["Et"].set_ylabel('$(E (t) - E_0) / E_0$', fontsize=22)
        
            if np.abs(self.diagnostics.circulation_0) < self.eps:
                self.axes["Cs"].set_ylabel('$C (t)$', fontsize=22)
                self.axes["Ct"].set_ylabel('$C (t)$', fontsize=22)
            else:
                self.axes["Cs"].set_ylabel('$(C (t) - C_0) / C_0$', fontsize=22)
                self.axes["Ct"].set_ylabel('$(C (t) - C_0) / C_0$', fontsize=22)
        
        
        for ax in self.axes:
            for tick in self.axes[ax].xaxis.get_major_ticks():
                tick.set_pad(12)
                tick.label.set_fontsize(16)
            for tick in self.axes[ax].yaxis.get_major_ticks():
                tick.set_pad(8)
                tick.label.set_fontsize(16)
        
        
        plt.draw()
        
        if self.output:
            self.save_plots()
        else:
            plt.show(block=False)
        
    
    def update_boundaries(self):

        self.Vmin = +1e40
        self.Vmax = -1e40
        
        self.Vmin = min(self.Vmin, self.diagnostics.Vx.min() )
        self.Vmin = min(self.Vmin, self.diagnostics.Vy.min() )
        
        self.Vmax = max(self.Vmax, self.diagnostics.Vx.max() )
        self.Vmax = max(self.Vmax, self.diagnostics.Vy.max() )

        dV = 0.1 * (self.Vmax - self.Vmin)
        self.Vnorm = colors.Normalize(vmin=self.Vmin-dV, vmax=self.Vmax+dV)
        
    
    def read_data(self):
        self.P [0:-1, 0:-1] = self.diagnostics.P [:,:]
        self.P [  -1, 0:-1] = self.diagnostics.P [0,:]
        self.P [   :,   -1] = self.P[:,0]
        
        self.O [0:-1, 0:-1] = self.diagnostics.O [:,:]
        self.O [  -1, 0:-1] = self.diagnostics.O [0,:]
        self.O [   :,   -1] = self.O[:,0]
        
        self.Vx[0:-1, 0:-1] = self.diagnostics.Vx[:,:]
        self.Vx[  -1, 0:-1] = self.diagnostics.Vx[0,:]
        self.Vx[   :,   -1] = self.Vx[:,0]
        
        self.Vy[0:-1, 0:-1] = self.diagnostics.Vy[:,:]
        self.Vy[  -1, 0:-1] = self.diagnostics.Vy[0,:]
        self.Vy[   :,   -1] = self.Vy[:,0]
    
    
    def save_plots(self):
        plt.figure(0)
        filename = self.prefix + str('_replay_%06d' % self.iTime) + '.png'
        plt.savefig(filename, dpi=100)

        plt.figure(1)
        filename = self.prefix + str('_vorticity_%06d' % self.iTime) + '.png'
        plt.savefig(filename, dpi=100)

        plt.figure(2)
        filename = self.prefix + str('_streaming_%06d' % self.iTime) + '.png'
        plt.savefig(filename, dpi=100)

        plt.figure(3)
        filename = self.prefix + str('_circulation_%06d' % self.iTime) + '.pdf'
        plt.savefig(filename)

        plt.figure(4)
        filename = self.prefix + str('_enstrophy_%06d' % self.iTime) + '.pdf'
        plt.savefig(filename)

        plt.figure(5)
        filename = self.prefix + str('_energy_%06d' % self.iTime) + '.pdf'
        plt.savefig(filename)
    
        plt.figure(6)
        filename = self.prefix + str('_traces_%06d' % self.iTime) + '.pdf'
        plt.savefig(filename)
        
    
    def update(self, final=False):
        
        if not (self.iTime == 0 or self.iTime % self.nPlot == 0 or self.iTime == self.nTime):
            return
        
        self.read_data()
        
        self.title.set_text('t = %1.2f' % (self.diagnostics.tGrid[self.iTime]))
        
        self.pcms["P" ].set_array(self.P.T.ravel())
        self.pcms["O" ].set_array(self.O.T.ravel())
        
        self.pcms["Vx"].set_array(self.Vx.T.ravel())
        self.pcms["Vy"].set_array(self.Vy.T.ravel())
        
        self.update_timetrace("C",  self.circulation)
        self.update_timetrace("W",  self.enstrophy)
        self.update_timetrace("E",  self.energy)
        self.update_timetrace("Mx", self.momentum_x)
        self.update_timetrace("My", self.momentum_y)
        
        if self.output:
            self.pcms["Ps" ].set_array(self.P.T.ravel())
            self.pcms["Os" ].set_array(self.O.T.ravel())
            
            self.update_timetrace("Cs",  self.circulation)
            self.update_timetrace("Ws",  self.enstrophy)
            self.update_timetrace("Es",  self.energy)
            self.update_timetrace("Ct",  self.circulation)
            self.update_timetrace("Wt",  self.enstrophy)
            self.update_timetrace("Et",  self.energy)
            
        
        plt.draw()
        
        if self.output:
            self.save_plots()
        else:
            plt.show(block=final)

    
    def update_timetrace(self, index, data):
        self.lins[index].set_xdata(self.diagnostics.tGrid[0:self.iTime+1])
        self.lins[index].set_ydata(data[0:self.iTime+1])
        self.axes[index].relim()
        self.axes[index].autoscale_view()
        self.axes[index].set_xlim((self.diagnostics.tGrid[0], self.diagnostics.tGrid[self.nTime])) 
        
    
    def add_timepoint(self):
        
        self.iTime += 1
        
        self.enstrophy  [self.iTime] = self.diagnostics.enstrophy_error
        self.energy     [self.iTime] = self.diagnostics.energy_error

        if np.abs(self.diagnostics.circulation_0) < self.eps:
            self.circulation[self.iTime] = self.diagnostics.circulation
        else:
            self.circulation[self.iTime] = self.diagnostics.circulation_error
        
        if np.abs(self.diagnostics.momentum_x_0) < self.eps:
            self.momentum_x [self.iTime] = self.diagnostics.momentum_x
        else:
            self.momentum_x [self.iTime] = self.diagnostics.momentum_x_error
        
        if np.abs(self.diagnostics.momentum_y_0) < self.eps:
            self.momentum_y [self.iTime] = self.diagnostics.momentum_y
        else:
            self.momentum_y [self.iTime] = self.diagnostics.momentum_y_error
        
