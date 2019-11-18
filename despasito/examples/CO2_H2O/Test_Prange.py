import sys
import numpy as np
import matplotlib.pyplot as plt

import despasito.thermodynamics.calc as calc
import despasito.thermodynamics as thermo
import despasito.equations_of_state

T = 351.1
xi = np.array([0.008, 0.992])

beads_co2_h2o = ['CO2', '2H2O']
nui_co2_h2o = np.array([[1., 0.],[0., 1.]])
beadlibrary_co2_h2o = {'CO2': {'epsilon': 361.69, 'l_a': 6.66, 'l_r': 23.0, 'sigma': 3.741e-10, 'Sk': 1.0, 'Vks': 1, 'mass': 0.04401},'2H2O': {'epsilon': 400.0, 'l_a': 6.0, 'l_r': 8.00, 'sigma': 3.7467e-10, 'Sk': 1.0, 'Vks': 1, 'mass': 0.03603}}
crosslibrary_co2_h2o = {'CO2': {'2H2O': {'epsilon':179.44146}}}
eos = despasito.equations_of_state.eos(eos="saft.gamma_mie",xi=xi,beads=beads_co2_h2o,nui=nui_co2_h2o,beadlibrary=beadlibrary_co2_h2o,crosslibrary=crosslibrary_co2_h2o)
yi = [9.99305059e-01, 6.94941027e-04]

Pmin = 2627641.9
Pmax = 1.5*267772686.9

Prange = np.linspace(Pmin,Pmax,100)

obj = []
for p in Prange:
    phil, rhol, flagl = calc.calc_phil(p, T, xi, eos)
    yi, phiv, flagv = calc.solve_yi_xiT(yi, xi, phil, p, T, eos)
    print("\n    xi, phil, flag",xi,phil,flagl)
    print("    yi, phiv, flag",yi,phiv,flagv)
    print("Obj: ",np.sum(xi * phil / phiv) - 1.0)
    obj.append(np.sum(xi * phil / phiv) - 1.0)

plt.plot(Prange,obj,"-o")
plt.show()
