
.. _contributing-eos:

Additional Equations of State
=========================================================

General Process
################
Adding another EOS, or family is straightforward in DESPASITO.
The EOS keyword in :func:`~despasito.equations_of_state.initiate_eos` is structured as EOS family and variant separated by a period (e.g. eos_family.eos_variant or saft.gamma_mie). 
However, those terms also correspond to the next two layers of respective submodules (Figure 1), as the initiating function uses those terms to locate the EOS class. 
Within a given ``despasito.equations_of_state.eos_family.eos_variant`` module is the class ``EosType`` for each EOS option. 
Each ``EosType`` references our :class:`~despasito.equations_of_state.interface.EosTemplate` abstract class to ensure consistency with the other modules. 
Here abstract methods ensure it's compatibility with the ``thermodynamics`` and ``parameter_fitting`` modules.
This structure requires an additional level of organization where each EOS must be categorized in an EOS family.
Once a new EOS is added, we have included thermodynamic consistency tests, :func:`~despasito.thermodynamics.calculation_types.verify_eos`, to ensure accuracy.

.. image:: figures/eos_module.png

*Figure 1: The Equations of State module includes an abstract class,* ``EosTemplate`` *from which all* ``EosType`` *classes inherit. Added EOS types are easily discovered by* ``initiate_eos`` *in a factory pattern, represented by the green folders. Similarly, the green color of Aideal signifies that it is a source for a factory pattern of calculating the ideal term. This module is equipped to use compiled modules to decrease computational time, although only selected modules are currently employing this option.*

.. _add-saft:

Additional Versions of SAFT
###########################
Our interest in SAFT has led us to add another level of organization for its specific use. 
As described earlier, SAFT relates non-bonded interactions to macroscopic properties through the Helmholtz free energy. 
Contributions to the Helmholtz free energy are segmented into terms.
Often these include the ideal, monomer, chain, and association site contributions, although different SAFT variants may or may not include other terms (e.g. electrostatic and solvation terms `[1]`_ or polar terms `[2]`_). An added SAFT sub-class must be added to the function :func:`~despasito.equations_of_state.saft.saft.saft_type`.
The ideal and association site terms are usually consistent throughout the variants of SAFT, while those variants are distinguished by the particulars of the remaining terms. 
Thus, a general SAFT class, :class:`despasito.equations_of_state.saft.saft.EosType`,  handles the ideal and association site terms, and separate, variant specific classes, are imported to provide their specific terms, as illustrated in Figure 1.
Because of the fundamental nature of the ideal term and the occasional variation in its method of calculation, the module, Aideal.py allows a factory pattern to exchange methods. 
Additionally, this main SAFT class also handles parameters updates in a generalized manner. 
Thus, we removed redundancy on multiple fronts, allowing rapid addition of a SAFT variants.

In the directory ``despasito.equations_of_state.saft`` we have included a commented example of a SAFT sub-class to aid in the addition of a new SAFT variant.

_`[1]` Shahriari, R.; Dehghani, M. R. Fluid Phase Equilibria 2018, 463, 128–141. https://doi.org/10.1016/j.fluid.2018.02.006.

_`[2]` Dominik, A.; Chapman, W. G.; Kleiner, M.; Sadowski, G. Ind. Eng. Chem. Res. 2005, 44 (17), 6928–6938. https://doi.org/10.1021/ie050071c.


EOS Class Interface
###################
Adding an EOS family is easily implemented by adding a new directory to the ``equations_of_state`` directory. A new EOS is added as a module containing a class named ``EosType`` derived from our EOS interface:

.. autoclass::
   despasito.equations_of_state.interface.EosTemplate
   :members:


