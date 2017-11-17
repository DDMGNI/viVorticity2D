
import numpy as np


xcenter = 0.0
ycenter = 2.0
xsigma  = 0.2
ysigma  = 0.2


def vorticity(x, y, Lx, Ly):
    return gaussian(x, xcenter, xsigma) * gaussian(y, ycenter, ysigma)

def streaming_function(x, y, Lx, Ly):
    return 0.5 * y**2 + 1. - np.cos(x)


def gaussian(x, center, sigma):
    return 1. / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-center)/sigma)**2 )
