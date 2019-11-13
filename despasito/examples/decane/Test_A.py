import sys
import numpy as np
import matplotlib.pyplot as plt

import despasito.thermodynamics.calc as calc
import despasito.thermodynamics as thermo
import despasito.equations_of_state

T = 700.0
P = 2307225.081
n = 100
V = 300*1e-27
xi = np.array([1.0])
rho = np.array([n/V])/6.02214086e23
print("rho ",rho)
print("P ",P)

beads  = ['CH3', 'CH2']
nui  = np.array([[2., 8.]])
beadlibrary  = {'CH3': {'epsilon': 256.7662, 'l_a': 6.0, 'l_r': 15.04982, 'sigma': 4.077257e-10, 'Sk': 0.5725512, 'Vks': 1, 'mass': 0.015035},'CH2': {'epsilon': 473.3893, 'l_a': 6.0, 'l_r': 19.87107, 'sigma': 4.880081e-10, 'Sk': 0.2293202, 'Vks': 1, 'mass': 0.014027}}
crosslibrary  = {'CH3': {'CH2': {'epsilon': 350.770}}}
eos = despasito.equations_of_state.eos(eos="saft.gamma_mie",xi=xi,beads=beads ,nui=nui ,beadlibrary=beadlibrary ,crosslibrary=crosslibrary )

eos.fugacity_coefficient(P, rho,xi,T)
