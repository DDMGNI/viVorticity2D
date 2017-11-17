'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import matplotlib

from diagnostics import Diagnostics 


class replay(object):
    '''
    
    '''

    def __init__(self, hdf5_file, nPlot=1, nMax=0, output=False, contours=False):
        '''
        Constructor
        '''
        
        self.diagnostics = Diagnostics(hdf5_file)
        
        if nMax > 0 and nMax < self.diagnostics.nt:
            self.nMax = nMax
        else:
            self.nMax = self.diagnostics.nt
        
        self.nPlot = nPlot
        self.plot  = PlotVorticity2D(self.diagnostics, output=output)
        
    
    def run(self):
#         for iTime in range(1, self.nMax+1):
        for iTime in [5,10,20,30,60]:
            if iTime == 0 or iTime % self.nPlot == 0 or iTime == self.nMax:
                print(iTime)
                self.diagnostics.read_from_hdf5(iTime)
                self.diagnostics.update_invariants(iTime)
                self.plot.update(iTime, final=(iTime == self.nMax))
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vorticity Equation Solver in 2D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')
    parser.add_argument('-nt', metavar='i', type=int, default=0,
                        help='plot up to i\'th frame')
    parser.add_argument('-o', action='store_true', required=False,
                        help='save plots to file')
    parser.add_argument('-c', action='store_true', required=False,
                        help='plot contours of streaming function in vorticity')
    
    args = parser.parse_args()
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    if args.o == True:
        matplotlib.use('AGG')
        from plot_contours import PlotVorticity2D
        pyvp = replay(args.hdf5_file, args.np, args.nt, output=True, contours=args.c)
        pyvp.run()
    else:
        from plot_contours import PlotVorticity2D
        pyvp = replay(args.hdf5_file, args.np, args.nt, output=False, contours=args.c)
    
        print
        input('Hit any key to start replay.')
        print
    
        pyvp.run()
    
    print
    print("Replay finished.")
    print
    
