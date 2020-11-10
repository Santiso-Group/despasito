
Peng-Robinson
=======================

EOS type: ``cubic.peng-robinson``

Since its publication in 1976, the Peng & Robinson equation of state (PR EOS) has become one of the most useful and successfully applied models for thermodynamic and volumetric calculations in both industrial and academic fields. Although several variations exist, the traditional form in terms of density is:

:math:`P=\frac{R T \rho}{1-b_{ij} \rho}-\frac{a_{ij} \rho^2}{(1+b_{ij} \rho)+\rho b_{ij} (1-b_{ij} \rho)}`

Where:

:math:`a_{i}=0.45723553 \frac{R^2 T_{C}^2}{P_{C}}` 

:math:`b_{i}=0.07779607 \frac{R T_{C}}{P_{C}}`

:math:`\alpha_{i}=[1+(0.37464+1.54226 \omega-0.26992 \omega^2) (1-\sqrt{T_{R}^2})]^2`

We used the following mixing rules:

:math:`a_{ij}=\sum{\sum{x_{i} x_{j} \sqrt{a_{i} \alpha_{i} a_{j} \alpha_{j}} (1-k_{ij})}}`

:math:`b_{ij}=\sum{x_{i} b{i}}`


.. autoclass::
   despasito.equations_of_state.cubic.peng_robinson.EosType
   :members:

