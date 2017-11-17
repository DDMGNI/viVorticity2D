
import numpy as np


xcenter = 0.2
ycenter = 0.2
xsigma  = 0.05
ysigma  = 0.05


def vorticity(x, y, Lx, Ly):
    return gaussian(x, xcenter, xsigma) * gaussian(y, ycenter, ysigma)

def streaming_function(x, y, Lx, Ly):
    return x**2 + y**2


def gaussian(x, center, sigma):
    return 1. / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-center)/sigma)**2 )
