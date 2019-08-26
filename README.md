DESPASITO
==============================
[//]: # (Badges)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/DESPASITO.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/DESPASITO)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/DESPASITO/branch/master)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/DESPASITO/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/DESPASITO/branch/master)

DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output

.. warning:: This package is not ready for distribution.

First open-source application for thermodynamic calculations and parameter fitting for the Statistical Associating Fluid Theory (SAFT) EOS and SAFT-𝛾-Mie coarse-grained simulations. This software has two primary facets. 

The first facet is a means to evaluate the SAFT-gamma-Mie EOS for binary VLE. This framework allows easy implementation of more advanced thermodynamic calculations as well as additional forms of SAFT or other equations of state. Feel free to contribute!

The second facet is parameterization, not only of the equation of state (EOS) but also for simulations. The SAFT-gamma-Mie formalism is an attractive source of simulation parameters as it offers a means to directly link the intermolecular potential with thermodynamic properties. This application has the ability to fit EOS parameters to experimental thermodynamic data in a top down approach for self and cross interaction parameters. We also process an expanded multipole mixing rule for cross interaction parameters. It should be noted that it is recommended to fine tune simulation parameters in an iterative fashion, but previous works have found close agreement with those fit to the EOS.

Installation
------------
.. note:: DESPASITO is not yet available on pip or conda-forge, but it is a package that can be installed in your python environment.

**Step 1:** Download the master branch from our github page as a zip file, or clone it with gitand save in your desired directory.

**Step 2:** Install with ``pip install .``

Command Line Use
----------------
This package has been primarily designed as a command line tool but can be used as an imported package.

In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

See examples directory or Input/Output documentation for input file structures.

### Copyright

Copyright (c) 2019, Jennifer A Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
