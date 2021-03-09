
FAQs
====

Thermodynamics
###############
**Q: My bubble point calculations aren't converging near the critical point, what do I do?**

**A:** Try adding the ``Pmin`` keyword with a pressure value just under the expected final value. As the calculations approach the critical point, it's more difficult for the algorithm to discern the lower limit of feasible pressure values.


Parameter Fitting
##################
**Q: After fitting the parameters there are several suitable parameter sets, which do I pick?**

**A:** Consider including additional experimental data, such as liquid density or second derivative properties.


SAFT-:math:`\gamma`-Mie
#########################
**Q: How can I use parameters from a SAFT equation of state in simulations?**

**A:** There are several references that have shown how to implement various versions of SAFT. An attractive option is the SAFT-:math:`\gamma` EOS since the group contribution approach is comparable to coarse-graining methods. There is a difficulty in applying this EOS as the shape factor, :math:`S_k`, does not seem directly applicable to the method. From the first order expansion of the monomer Helmholtz contribution, Yaroson `[1]`_ showed that the shape factor scales the energy parameter, providing basis for the assertion that:

:math:`\epsilon^{Sim} = S^2\epsilon^{EOS}`

Testing this relationship with simulations showed approximate agreement. Although is established that the shape factor characterizes the extent to which a group contributes toward the thermo-physical properties of a chain, it also provides a link to the geometry. This factor is needed for the EOS to properly represent a molecule's surface area and volume, otherwise this group contribution method suffers from unrealistic representations.



**Q: For an EOS with the Mie potential, are all four parameters (energy and size parameters, with both exponents) independent?**

**A:** A corresponding-states framework has been identified for the Mie family of intermolecular potentials that connects the repulsive and attractive exponents `[2]`_. A van der Waals like attraction parameter can be derived from the first-order perturbation term of the Helmholtz free energy:

:math:`\alpha^{Mie}=C [\frac{1}{\lambda_a-3} - \frac{1}{\lambda_r-3}]`

Where C is the prefactor of the Mie potential. This connection suggests that setting the attractive exponent to the physically meaningful value of six, does not limit the the Mie potential's ability to conform, and in fact not setting it would introduce degeneracy. After the fitting process the exponents can then we adjusted without altering the thermodynamic properties at this level of approximation as long as :math:`\alpha^{Mie}` is kept constant.



**Q: For an EOS with the Mie potential, how can the repulsive exponent be meaningfully swayed? Is there a way to prevent premature freezing in simulations?**

**A:** Yes, Lobanova et al. `[3]`_ successfully modeled water without the common issues with premature freezing in simulations. They reference a connection between :math:`\alpha^{Mie}` and the ratio between the critical temperature and triple point. If the exponents don't represent a high enough van der Waals - like attraction parameter, then a choice between a proper description of the critical point or triple point must be made.



**References**

_`[1]` Yaroson, O. H. E. PhD, Imperial College London, 2014.

_`[2]` Ramrattan, N. S.; Avendaño, C.; Müller, E. A.; Galindo, A. Mol. Phys. 2015, 113 (9–10), 932–947. https://doi.org/10.1080/00268976.2015.1025112.

_`[3]` Lobanova, O.; Avendaño, C.; Lafitte, T.; Müller, E. A.; Jackson, G. Mol. Phys. 2015, 113 (9–10), 1228–1249. https://doi.org/10.1080/00268976.2015.1004804.




