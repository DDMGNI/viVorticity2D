#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vorticity.setup import *


ext_list = ["PETScDerivatives",
            "PETScPoisson",
            "PETScVorticity"]

make_extension(ext_list)

