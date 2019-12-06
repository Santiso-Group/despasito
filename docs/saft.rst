
SAFT-:math:`\gamma`-Mie
=======================

The Statistical Associating Fluid Theory (SAFT) equation of state (EOS) is based on first order perturbation of the residual Helmholtz energy. The total Helmholtz energy is then defined as the sum of ideal gas, monomer (i.e. segment or group), chain (i.e. component), and association terms.

:math:`\frac{A}{N k_{B} T}=\frac{A^{ideal}}{N k_{B} T}+\frac{A^{mono.}}{N k_{B} T}+\frac{A^{chain}}{N k_{B} T}+\frac{A^{assoc.}}{N k_{B} T}`

This heteronuclear version of SAFT used the Mie potential to not only offer a group contribution EOS but a means to connect thermodynamic properties to bead definitions that can be simulated.
    
Papaioannou, V. et. al, J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455

.. autoclass::
   despasito.equations_of_state.saft.gamma_mie.saft_gamma_mie
   :members:

Supporting Functions
####################

**Calculation of Parameters Within Object**

.. currentmodule:: despasito

.. autosummary::
   :toctree: _autosummary

   equations_of_state.saft.gamma_mie_funcs
   equations_of_state.saft.nojit_exts
   equations_of_state.saft.jit_exts

