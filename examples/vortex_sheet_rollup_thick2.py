
import numpy as np

rho = 30.
r   = 0.5


def vorticity(x, y, Lx, Ly):
    t = np.zeros_like(x)
    
    i = y <= r
    t[i] = 0.05 * np.cos(2. * np.pi * x[i]) / np.cosh(rho * (y[i] - 0.25))**2
    
    i = y > r
    t[i] = 0.05 * np.cos(2. * np.pi * x[i]) / np.cosh(rho * (0.75 - y[i]))**2
    
    return t
