import sys
import numpy as np
import matplotlib.pyplot as plt

import despasito.thermodynamics.calc as calc
import despasito.thermodynamics as thermo
import despasito.equations_of_state

T = 323.2
xi = np.array([0.78988277, 0.21011723])
rho = np.array([21146.16997993])
P = np.array([1713500.67089664])

beads_co2_h2o = ['CO2', 'H2O']
nui_co2_h2o = np.array([[1., 0.],[0., 1.]])
beadlibrary_co2_h2o = {'CO2': {'epsilon': 207.89, 'l_a': 5.055, 'l_r': 26.408, 'sigma': 3.05e-10, 'Sk': 0.8468, 'Vks': 2, 'mass': 0.04401, 'NkH': 1, 'Nka1': 1},'H2O': {'epsilon': 266.68, 'l_a': 6.0, 'l_r': 17.02, 'sigma': 3.0063e-10, 'Sk': 1.0, 'Vks': 1, 'mass': 0.018015, 'NkH': 2, 'Nke1': 2, 'epsilonHe1': 1985.4, 'KHe1': 1.0169e-28}}
crosslibrary_co2_h2o = {'CO2': {'H2O': {'epsilon': 226.38, 'epsilonHe1': 2200.0, 'KHe1': 9.1419e-29}}}
sitenames_co2_h2o = ['H', 'e1', 'a1']
epsilonHB_co2_h2o = np.array([[[[   0.,     0.,     0. ], \
                       [   0.,     0.,     0. ], \
                       [   0.,     0.,     0. ]], \
                      [[   0.,  2200.,     0. ], \
                       [   0.,     0.,     0. ], \
                       [   0.,     0.,     0. ]]], \
                     [[[   0.,     0.,     0. ], \
                       [2200.,     0.,     0. ], \
                       [   0.,     0.,     0. ]], \
                      [[   0.,  1985.4,    0. ], \
                       [1985.4,    0.,     0. ], \
                       [   0.,     0.,     0. ]]]])
eos = despasito.equations_of_state.eos(eos="saft.gamma_mie",xi=xi,beads=beads_co2_h2o,nui=nui_co2_h2o,beadlibrary=beadlibrary_co2_h2o,crosslibrary=crosslibrary_co2_h2o,sitenames=sitenames_co2_h2o)

dy_array = [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
dy_array = [1e-4]
x1_array = np.linspace(0.,0.01,100)

for dy in dy_array:
    phi = []
    for x1 in x1_array:
        xi = np.array([x1,1.-x1])
        try:
            phi.append(eos.fugacity_coefficient(P,rho,xi,T,dy=dy))
        except:
            phi.append(np.array([0.,0.]))
    phi = np.array(phi).T
    plt.figure(1)
    plt.plot(x1_array,phi[0],label="dy={}".format(dy))
    plt.figure(2)
    plt.plot(x1_array,phi[1],label="dy={}".format(dy))
plt.legend(loc="best")
plt.show()
