.. despasito documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DESPASITO's documentation!
=========================================================

DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output

First open-source application for thermodynamic calculations and parameter fitting for the Statistical Associating Fluid Theory (SAFT) equation of state (EOS) and SAFT-:math:`\gamma`-Mie coarse-grained simulations. This software has two primary facets. 

The first facet is a means to evaluate the SAFT-:math:`\gamma`-Mie EOS for binary vapor-liquid equilibria (VLE). This framework allows easy implementation of more advanced thermodynamic calculations as well as additional forms of SAFT or other equations of state. Feel free to contribute!

The second facet is parameterization, not only of the equation of state (EOS) but also for simulations. The SAFT-:math:`\gamma`-Mie formalism is an attractive source of simulation parameters as it offers a means to directly link the intermolecular potential with thermodynamic properties. This application has the ability to fit EOS parameters to experimental thermodynamic data in a top down approach for self and cross interaction parameters. Once published, we will also provide scripts to estimate cross interaction parameters for SAFT-:math:`\gamma`-Mie from DFT calculations. It should be noted that it is recommended to additionally fine tune simulation parameters in an iterative fashion, but previous works have found close agreement with those fit to the EOS.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   getting_started
   api
   faqs

.. toctree::
   :maxdepth: 2
   :caption: Modules:
   
   input_output
   eos
   thermo
   fitparams
   utils

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   settingup
   startfitting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

