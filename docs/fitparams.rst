
Parametrization
=========================================================


Fit to Experimental Data
------------------------
Parameters can be fit for one component at a time, and as many parameters as desired **can** be fit. Input file structure is very similar to before with the mandatory addition of an `optimization_parameters` section whose presence will signify to DESPASITO that this is a fitting job. Here we summarize the available experimental data types that are currently supported for fitting, followed by the desciption of the main fitting function containing more details.

.. automodule:: despasito.parameter_fitting
   :members:

.. _data-types:

Available Data Types
------------------------

.. currentmodule:: despasito.parameter_fitting.data_classes
.. autosummary::
   :toctree: _autosummary

   saturation_properties.Data
   TLVE.Data
   liquid_density.Data
   solubility_parameter.Data

Supporting Thermodynamic Functions
------------------------------------------------

.. currentmodule:: despasito.parameter_fitting
.. autosummary::
   :toctree: _autosummary

   fit_functions
   global_methods
   constraint_types

Estimate with Electronic Structure Methods
------------------------------------------
In SAFT, self-interaction parameters are often fit to experimental data, and in most cases so are the cross-interaction parameters (between segments of different types). In a work nearing publication, we derived combining rules extended to utilize multipole moments of molecular fragments from density functional theory (DFT) methods using R.E.D. server [1].
Estimation with electronic structure methods involves using the extra functions in our `scripts` directory once this work is published.

[1] Vanquelef, E.; Simon, S.; Marquant, G.; Garcia, E.; Klimerak, G.; Delepine, J. C.; Cieplak, P.; Dupradeau, F.-Y. R.E.D. Server: A Web Service for Deriving RESP and ESP Charges and Building Force Field Libraries for New Molecules and Molecular Fragments. Nucleic Acids Res. 2011, 39 (suppl_2), W511–W517. https://doi.org/10.1093/nar/gkr288

