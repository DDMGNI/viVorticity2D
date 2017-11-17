
import numpy as np
from scipy.special import jv

lambdaR = 3.83170597020751231561
R       = 0.2
U       = 1.0
lam     = lambdaR / R

def vorticity(x, y, Lx, Ly):
    assert np.shape(x) == np.shape(y)
    
    temp = np.zeros_like(x)
    
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    jv0   = jv(0, lambdaR)
    jv1   = jv(1, lambdaR * r / R)
    
    temp[r < R] = 2. * lam * U * np.cos(theta)[r < R] * jv1[r < R] / jv0
    
    return temp
