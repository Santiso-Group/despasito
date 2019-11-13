import sys
import numpy as np
import matplotlib.pyplot as plt

import despasito.thermodynamics.calc as calc
import despasito.thermodynamics as thermo
import despasito.equations_of_state

T = 323.2
xi = np.array([0.78988277, 0.21011723])
P = 4492927.45
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

Nmol = [1, 10, 100, 1000, 10000, 100000]
DNmol = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
ratio = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#for i,nmol in enumerate(Nmol):
#    mui = []
#    for j,dnmol in enumerate(DNmol):
#        mui.append(eos.chemicalpotential(rho,xi,T,nmol=nmol,dnmol=dnmol))
#    mui = np.array(mui).T
#    plt.figure(1)
#    plt.plot(range(1,len(DNmol)+1),mui[0],"-.",label=str(nmol))
#    plt.figure(2)
#    plt.plot(range(1,len(DNmol)+1),mui[1],"-.",label=str(nmol))
#
#plt.figure(1)
##plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend(loc="best")
#plt.figure(2)
##plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend(loc="best")

######### 
for i,nmol in enumerate(Nmol):
    mui = []
    for r in ratio:
        mui.append(eos.chemicalpotential(rho,xi,T,nmol=nmol,dnmol=nmol/10**r))
    mui = np.array(mui).T
    plt.figure(1)
    plt.plot(range(1,len(ratio)+1),mui[0],"-o",label="".format(nmol,))
    plt.figure(2)
    plt.plot(range(1,len(ratio)+1),mui[1],"-o",label=str(nmol))

######### 
for i,dnmol in enumerate(DNmol):
    mui = []
    for r in ratio:
        mui.append(eos.chemicalpotential(rho,xi,T,nmol=dnmol*10**r,dnmol=dnmol))
    mui = np.array(mui).T
    plt.figure(3)
    plt.plot(range(1,len(ratio)+1),mui[0],"-o",label=str(dnmol))
    plt.figure(4)
    plt.plot(range(1,len(ratio)+1),mui[1],"-o",label=str(dnmol))

plt.figure(1)
plt.legend(loc="best")
plt.figure(2)
plt.legend(loc="best")
plt.figure(3)
plt.legend(loc="best")
plt.figure(4)
plt.legend(loc="best")

plt.show()

