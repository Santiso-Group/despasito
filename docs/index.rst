.. despasito documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DESPASITO's documentation!
=========================================================

DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output

First open-source application for thermodynamic calculations and parameter fitting for the Statistical Associating Fluid Theory (SAFT) equation of state (EOS) and SAFT-:math:`\gamma`-Mie coarse-grained simulations. This software has two primary facets. 

The first facet is a means to evaluate implicit equations of state (EOS), such as the focus of this package, SAFT-:math:`\gamma`-Mie. This framework allows easy implementation of more advanced thermodynamic calculations as well as additional forms of SAFT or other equations of state. Feel free to contribute!

The second facet is parameterization of the equation of state (EOS), some of which are useful for coarse-grained (CG) simulations. The SAFT-:math:`\gamma`-Mie formalism is an attractive source of simulation parameters as it offers a means to directly link the intermolecular potential with thermodynamic properties. This application has the ability to fit EOS parameters to experimental thermodynamic data in a top down approach for self and cross interaction parameters. 

In another work nearing publication, we present a method of predicting cross-interaction parameters for SAFT-:math:`\gamma`-Mie from multipole moments derived from DFT calculations. This method is easily implemented in using the package, `MAPSCI <https://github.com/jaclark5/mapsci>`_ as a plug-in. It should be noted that additional, iterative fine tuning in a simulation parameters may be desired, but previous works have found close agreement between simulation parameters and those fit to the EOS.

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
   :maxdepth: 2
   :caption: Contributing:
   
   contribute_intro
   contribute_eos
   contribute_thermo
   contribute_fitparams

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

