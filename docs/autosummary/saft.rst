
SAFT
====

The Statistical Associating Fluid Theory (SAFT) equation of state (EOS) is based on first order perturbation of the residual Helmholtz energy. The total Helmholtz energy is then defined as the sum of ideal gas, monomer (i.e. segment or group), chain (i.e. component), and association terms.

:math:`\frac{A}{N k_{B} T}=\frac{A^{ideal}}{N k_{B} T}+\frac{A^{mono.}}{N k_{B} T}+\frac{A^{chain}}{N k_{B} T}+\frac{A^{assoc.}}{N k_{B} T}`

SAFT-:math:`\gamma`-Mie
-----------------------
This heteronuclear version of SAFT used the Mie potential to not only offer a group contribution EOS but a means to connect thermodynamic properties to bead definitions that can be simulated.
    
Papaioannou, V. et. al, J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455

.. autoclass::
   despasito.equations_of_state.saft.gamma_mie.saft_gamma_mie
   :members:

Supporting Functions
####################

**Calculation of Parameters Within Object**

.. currentmodule:: despasito.equations_of_state.saft.gamma_mie_funcs
.. autofunction:: calc_interaction_matrices
.. autofunction:: calc_composition_dependent_variables
.. autofunction:: calc_hard_sphere_matricies
.. autofunction:: calc_assoc_matrices

Total Helmholtz Energy
''''''''''''''''''''''
.. currentmodule:: despasito.equations_of_state.saft.gamma_mie_funcs
.. autofunction:: calc_A
.. autofunction:: calc_Ares
.. autofunction:: C

Ideal Gas Contribution, :math:`A^{ideal}`
'''''''''''''''''''''''''''''''''''''''''
.. currentmodule:: despasito.equations_of_state.saft.gamma_mie_funcs
.. autofunction:: calc_Aideal

Monomer (Group) Contribution, :math:`A^{mono.}`
'''''''''''''''''''''''''''''''''''''''''''''''
.. currentmodule:: despasito.equations_of_state.saft.gamma_mie_funcs
.. autofunction:: calc_Amono
.. autofunction:: calc_a1s
.. autofunction:: calc_Bkl
.. autofunction:: calc_dkk
.. autofunction:: calc_fm

Chain (Molecular) Contribution, :math:`A^{chain}`
'''''''''''''''''''''''''''''''''''''''''''''''''
.. currentmodule:: despasito.equations_of_state.saft.gamma_mie_funcs
.. autofunction:: calc_Achain
.. autofunction:: calc_a1ii
.. autofunction:: calc_da1iidrhos
.. autofunction:: calc_a2ii_1pchi
.. autofunction:: calc_da2ii_1pchi_drhos

Association Site Contribution, :math:`A^{assoc.}`
'''''''''''''''''''''''''''''''''''''''''''''''''
.. currentmodule:: despasito.equations_of_state.saft.gamma_mie_funcs
.. autofunction:: calc_A_assoc
.. autofunction:: calc_Xika_wrap 

