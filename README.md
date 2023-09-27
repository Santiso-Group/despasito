DESPASITO
==============================
[//]: # (Badges)
[![DOI](https://zenodo.org/badge/202225860.svg)](https://zenodo.org/badge/latestdoi/202225860)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions Build Status](https://github.com/jaclark5/despasito/workflows/CI/badge.svg)](https://github.com/jaclark5/despasito/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jaclark5/DESPASITO/branch/master/graph/badge.svg)](https://codecov.io/gh/jaclark5/DESPASITO/branch/master)
[![Documentation Status](https://readthedocs.org/projects/despasito/badge/?version=latest)](https://despasito.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jaclark5/despasito.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jaclark5/despasito/context:python)

DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output

First open-source application for thermodynamic calculations and parameter fitting for the Statistical Associating Fluid Theory (SAFT) EOS and SAFT-𝛾-Mie coarse-grained simulations. This software has two primary facets. 

The first facet is a means to evaluate implicit equations of state (EOS), such as the focus of this package, SAFT-𝛾-Mie. This framework allows easy implementation of more advanced thermodynamic calculations as well as additional forms of SAFT or other equations of state. Feel free to contribute!

The second facet is parameterization of equations of state (EOS), some of which are useful for coarse-grained (CG) simulations. The SAFT-𝛾-Mie formalism is an attractive source of simulation parameters as it offers a means to directly link the intermolecular potential with thermodynamic properties. This application has the ability to fit EOS parameters to experimental thermodynamic data in a top down approach for self and cross interaction parameters. 

In another [published work](doi.org/10.1021/acs.jpcb.1c00851), we present a method of predicting cross-interaction parameters for SAFT-𝛾-Mie from multipole moments derived from DFT calculations. This method is easily implemented in using the package, [MAPSCI](https://github.com/jaclark5/mapsci) as a plug-in. It should be noted that additional, iterative fine tuning in simulation parameters may be desired, but previous works have found close agreement between simulation parameters and those fit to the EOS.

Need Assistance?
---------------

Check out our [Documentation](https://despasito.readthedocs.io/en/latest/) first.

Documentation
--------------
Check out our [Documentation](https://despasito.readthedocs.io):

Installation
------------
**NOTE:** DESPASITO is not yet available in conda-forge, but it is available with pip.

**Prerequisites**:
  * **Python**: Available for python 3.6 to 3.8 (limited by current compatibility issues with Numba)
  * [NumPy](https://numpy.org): needed for running setup (distutils). Follow instructions outlined [here](https://docs.scipy.org/doc/numpy/user/install.html) for installation.

Options
=======

**Option 1:** Install from pip: ``pip install despasito``

**Option 2:** Install locally with pip.

 * Step 1: Install the prerequisites listed above.
 * Step 2: Download the master branch from our github page as a zip file, or clone it with git via ``git clone https://github.com/jaclark5/despasito`` to your working directory.
 * Step 3: Install with ``pip install despasito/.``, or change directories and run ``pip install .``.

**NOTE** If [pip](https://pip.pypa.io/en/stable/) is unavailable, follow the instructions outlined [here](https://pip.pypa.io/en/stable/installing/) for installation.

**Option 3:** Install locally with python.

 * Step 1: Install the prerequisites listed above.
 * Step 2: Download the master branch from our github page as a zip file, or clone it with git via ``git clone https://github.com/jaclark5/despasito`` to your working directory.
 * Step 3: After changing directories, install with ``python setup.py install --user`` .

Command Line Use
----------------
This package has been primarily designed as a command line tool but can be used as an imported package.

In any directory with the appropriate input files in .json format, run DESPASITO with ``python -m despasito -i input.json``

See [examples](despasito/examples) directory for input file structures.

### Copyright

Copyright (c) 2019, Jennifer A Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
