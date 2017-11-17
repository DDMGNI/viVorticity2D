
import numpy as np

rho = 30.
r   = 0.5


def vorticity(x, y, Lx, Ly):
    assert np.shape(x) == np.shape(y)
    
    temp = 0.1 * np.pi * np.cos(2. * np.pi * x)
    temp[y <= r] -= rho / np.cosh(rho * (y[y <= r] - 0.25))**2
    temp[y >  r] += rho / np.cosh(rho * (0.75 - y[y >  r]))**2
     
    return temp
