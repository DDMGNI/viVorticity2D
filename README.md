
# viVorticity2D

*Python/Cython package providing reference implementations of variational integrators for the vorticity equation in 2D.*

[![Project Status: Inactive](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org/#inactive)
[![License: MIT](https://img.shields.io/badge/license-MIT%20License-blue.svg)](LICENSE.md)


The code must first be built by calling `make` in the main directory.
Then it can be run, e.g., in serial via

```
> python run_vorticity.py examples/gaussian_blob.cfg
```

and in parallel via

```
> mpiexec -n 4 python run_vorticity.py examples/gaussian_blob.cfg
```

The results can be visualised with the `diag_replay.py` and `diag_contours.py` scripts, either interactively via

```
> python diag_replay.py gaussian_blob.hdf5
```

or written to files via

```
> python diag_replay.py -o gaussian_blob.hdf5
```

Several options allow to control the output:

``` 
> python diag_replay.py -h

usage: diag_replay.py [-h] [-np i] [-nt i] [-o] [-c] <run.hdf5>

Vorticity Equation Solver in 2D

positional arguments:
  <run.hdf5>  Run HDF5 File

optional arguments:
  -h, --help  show this help message and exit
  -np i       plot every i'th frame
  -nt i       plot up to i'th frame
  -o          save plots to file
  -c          plot contours of streaming function in vorticity
```

The `examples` directory contains various test problems, including all runs from the paper (see reference below).


## Example Simulations

[![viVorticity2D Example Simulations](https://img.youtube.com/vi/8QZiP3T9kwU/0.jpg)](https://www.youtube.com/playlist?list=PLyiyWhorv9bnKolGYGCnJFgmfa_tVKk9Z)


## Reference

_Michael Kraus, Omar Maj_. Variational Integrators for Nonvariational Partial Differential Equations. Physica D: Nonlinear Phenomena, Volume 310, Pages 37-71, 2015.
[Journal](https://dx.doi.org/10.1016/j.physd.2015.08.002),
[arXiv:1412.2011](https://arxiv.org/abs/1412.2011).


## License

The viVorticity2D package is licensed under the [MIT "Expat" License](LICENSE.md).
