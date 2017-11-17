
import numpy as np


def vorticity(x, y, Lx, Ly):
    return 2. * np.sin(np.pi * x) * np.sin(np.pi * y)
