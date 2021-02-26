DESPASITO
==============================
[//]: # (Badges)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions Build Status](https://github.com/jaclark5/despasito/workflows/CI/badge.svg)](https://github.com/jaclark5/despasito/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jaclark5/DESPASITO/branch/master/graph/badge.svg)](https://codecov.io/gh/jaclark5/DESPASITO/branch/master)
[![Documentation Status](https://readthedocs.org/projects/despasito/badge/?version=latest)](https://despasito.readthedocs.io)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jaclark5/despasito.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jaclark5/despasito/context:python)

DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output

**WARNING!** This package is not ready for distribution.

First open-source application for thermodynamic calculations and parameter fitting for the Statistical Associating Fluid Theory (SAFT) EOS and SAFT-𝛾-Mie coarse-grained simulations. This software has two primary facets. 

The first facet is a means to evaluate the SAFT-𝛾-Mie EOS for binary VLE. This framework allows easy implementation of more advanced thermodynamic calculations as well as additional forms of SAFT or other equations of state. Feel free to contribute!

The second facet is parameterization, not only of the equation of state (EOS) but also for simulations. The SAFT-𝛾-Mie formalism is an attractive source of simulation parameters as it offers a means to directly link the intermolecular potential with thermodynamic properties. This application has the ability to fit EOS parameters to experimental thermodynamic data in a top down approach for self and cross interaction parameters. We also process an expanded multipole mixing rule for cross interaction parameters. It should be noted that it is recommended to fine tune simulation parameters in an iterative fashion, but previous works have found close agreement with those fit to the EOS.

Installation
------------
**NOTE:** DESPASITO is not yet available conda-forge, but it is available with pip.

**Prerequisites**:
  * [NumPy](https://numpy.org): needed for running setup (distutils). Follow instructions outlined [here](https://docs.scipy.org/doc/numpy/user/install.html) for installation.
  * [SetupTools](https://pypi.org/project/setuptools): needed for running setup (find_packages). Follow instructions outlined [here](https://pythonhosted.org/an_example_pypi_project/setuptools.html) for installation. 

**Step 1:** Install the prerequisites listed above.

**Step 2:** Install using pip with ``pip install -i https://test.pypi.org/simple/ despasito``

**NOTE** If [pip](https://pip.pypa.io/en/stable/) is unavailable, follow the instructions outlined [here](https://pip.pypa.io/en/stable/installing/) for installation. Alternatively, download the master branch from our github page as a zip file, or clone it with git via ``git clone https://github.com/jaclark5/despasito`` in your working directory. Install DESPASITO locally from the working directory with ``python setup.py install --user``.

Command Line Use
----------------
This package has been primarily designed as a command line tool but can be used as an imported package.

In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

See [examples](despasito/examples) directory for input file structures.

### Copyright

Copyright (c) 2019, Jennifer A Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
