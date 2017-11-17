'''
Created on Jul 02, 2012

@author: mkraus
'''

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


class PlotVorticity2D(object):
    '''
    classdocs
    '''

    def __init__(self, diagnostics, output=False):
        '''
        Constructor
        '''
        
        self.diagnostics = diagnostics
        self.output      = output
        self.prefix      = "viVorticity2D_contours"

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
         
        self.P  = np.zeros((diagnostics.nx+1, diagnostics.ny+1))
        self.O  = np.zeros((diagnostics.nx+1, diagnostics.ny+1))

        # add data for zero timepoint
        self.read_data()
        
        # compute maximum vorticity and select contours
        self.maxO = self.O.max()
        self.cntO = [0.2*self.maxO, 0.4*self.maxO, 0.7*self.maxO]
        self.colO = ('red', 'blue', 'green')
        
        # setup contour plot settings
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        
        # set up figure/window size
        self.figure = plt.figure(num=0, figsize=(16,9))
        self.axes   = plt.subplot(111)
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.5, wspace=0.25)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.07)
        
        # set up plot title
#         self.axes.set_title('Contours of $\phi (x,y)$ and $\omega (x,y)$ at $t = 0.0$' % (self.diagnostics.tGrid[0]), fontsize=22)
        
        # contours of streaming function
        self.axes.contour(self.x, self.y, self.P.T, 10, colors='black')
        
        # contours of vorticity
        self.axes.contour(self.x, self.y, self.O.T, self.cntO, colors=self.colO)
        
        # limit axis
        self.axes.set_xlim((self.x[0], self.x[-1]))
        self.axes.set_ylim((-np.pi, +np.pi)) 
        
        # set tick font size
        for tick in self.axes.xaxis.get_major_ticks():
            tick.set_pad(12)
            tick.label.set_fontsize(16)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.set_pad(8)
            tick.label.set_fontsize(16)

        plt.draw()
        
        if self.output:
            self.save_plots(0)
        else:
            plt.show(block=False)
        
    
    def read_data(self):
        self.P [0:-1, 0:-1] = self.diagnostics.P [:,:]
        self.P [  -1, 0:-1] = self.diagnostics.P [0,:]
        self.P [   :,   -1] = self.P[:,0]
        
        self.O [0:-1, 0:-1] = self.diagnostics.O [:,:]
        self.O [  -1, 0:-1] = self.diagnostics.O [0,:]
        self.O [   :,   -1] = self.O[:,0]
    
    
    def save_plots(self, iTime):
        plt.figure(0)
        filename = self.prefix + str('_replay_%06d' % iTime) + '.png'
        plt.savefig(filename, dpi=100)
        
    
    def update(self, iTime, final=False):
        
        self.read_data()
        
#         self.axes.set_title('Contours of $\phi (x,y)$ and $\omega (x,y)$ at $t = 0.0$' % (self.diagnostics.tGrid[0]), fontsize=22)
                
        # contours of vorticity
        self.axes.contour(self.x, self.y, self.O.T, self.cntO, colors=self.colO)
        
#         if final:
#             self.axes.pcolormesh(self.xpc, self.ypc, self.O.T, cmap=plt.get_cmap('autumn'))
        
        plt.draw()
        
        if self.output:
            self.save_plots(iTime)
        else:
            plt.show(block=final)
