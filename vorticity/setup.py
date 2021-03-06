#!/usr/bin/env python

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

import os
from os.path import join, isdir

INCLUDE_DIRS = []
LIBRARY_DIRS = []
LIBRARIES    = []
CARGS        = ['-O3', '-axavx', '-march=corei7-avx', '-std=c99', '-Wno-unused-function', '-Wno-unneeded-internal-declaration']
LARGS        = []

# PETSc
PETSC_DIR  = os.environ['PETSC_DIR']
PETSC_ARCH = os.environ.get('PETSC_ARCH', '')

if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
    INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                     join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
else:
    if PETSC_ARCH: pass # XXX should warn ...
    INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]

LIBRARIES    += ['petsc']

# NumPy
import numpy
INCLUDE_DIRS += [numpy.get_include()]

# PETSc for Python
import petsc4py
INCLUDE_DIRS += [petsc4py.get_include()]

# Intel MPI (MPCDF)
IMPI_DIR = '/afs/@cell/common/soft/intel/ics2013/impi/4.1.3/intel64'

if isdir(IMPI_DIR):
    INCLUDE_DIRS += [join(IMPI_DIR, 'include')]
    LIBRARY_DIRS += [join(IMPI_DIR, 'lib')]

# OpenMPI (MacPorts)
if isdir('/opt/local/include/openmpi-gcc6'):
    INCLUDE_DIRS += ['/opt/local/include/openmpi-gcc6']
if isdir('/opt/local/lib/openmpi-gcc6'):
    LIBRARY_DIRS += ['/opt/local/lib/openmpi-gcc6']


def make_extension(extension_list):
    ext_modules = [
            Extension(ext,
                      sources=[ext + ".pyx"],
                      include_dirs=INCLUDE_DIRS + [os.curdir],
                      libraries=LIBRARIES,
                      library_dirs=LIBRARY_DIRS,
                      runtime_library_dirs=LIBRARY_DIRS,
                      extra_compile_args=CARGS,
                      extra_link_args=LARGS,
                     ) for ext in extension_list]
                
    setup(
        name = 'PETSc Matrix-Free Vorticity Solver',
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules
    )
