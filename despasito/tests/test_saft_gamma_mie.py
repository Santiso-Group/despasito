"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.equations_of_state.saft.gamma_mie
import despasito.equations_of_state.saft.gamma_mie_funcs

import despasito.equations_of_state.saft.solv_assoc as solv_assoc

import pytest
import sys
import numpy as np

xi_co2_ben = np.array([0.2, 0.2])
beads_co2_ben = ['CO2', 'benzene']
nui_co2_ben = np.array([[1., 0.],[0., 1.]])
beadlibrary_co2_ben = {'CO2': {'epsilon': 361.69, 'l_a': 6.66, 'l_r': 23.0, 'sigma': 3.741e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.04401}, 'baCH': {'epsilon': 371.53, 'l_a': 6.0, 'l_r': 14.756, 'sigma': 4.0578e-10, 'Sk': 0.32184, 'Vks': 1.0, 'mass': 0.01302}, 'baCHCH': {'epsilon': 243.88, 'l_a': 6.0, 'l_r': 11.58, 'sigma': 3.482e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.02604}, 'baCHCHCH': {'epsilon': 353.93, 'l_a': 6.0, 'l_r': 14.23, 'sigma': 3.978e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.03905}, 'benzene': {'epsilon': 658.17, 'l_a': 6.0, 'l_r': 32.0, 'sigma': 3.842e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.07811}}

xi_co2_h2o = np.array([0.78988277, 0.21011723])
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
eos_co2_h2o = despasito.equations_of_state.eos(eos="saft.gamma_mie",xi=xi_co2_h2o,beads=beads_co2_h2o,nui=nui_co2_h2o,beadlibrary=beadlibrary_co2_h2o,crosslibrary=crosslibrary_co2_h2o,sitenames=sitenames_co2_h2o)
T = 323.2 
rho_co2_h2o = np.array([21146.16997993]) 
P = np.array([1713500.67089664])

def test_saft_gamma_mie_imported():
#    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.equations_of_state.saft.gamma_mie" in sys.modules

def test_saft_gamma_mie_class_noassoc(xi=xi_co2_ben,beads=beads_co2_ben,nui=nui_co2_ben,beadlibrary=beadlibrary_co2_ben):    
#   """Test ability to create EOS object without association sites"""
    eos_class = despasito.equations_of_state.eos(eos="saft.gamma_mie",xi=xi,beads=beads,nui=nui,beadlibrary=beadlibrary)
    assert (eos_class._massi==np.array([0.04401, 0.07811])).all()

def test_saft_gamma_mie_class_assoc(xi=xi_co2_h2o,beads=beads_co2_h2o,nui=nui_co2_h2o,beadlibrary=beadlibrary_co2_h2o,crosslibrary=crosslibrary_co2_h2o,sitenames=sitenames_co2_h2o,epsilonHB=epsilonHB_co2_h2o):
#   """Test ability to create EOS object with association sites"""
    eos_class = despasito.equations_of_state.eos(eos="saft.gamma_mie",xi=xi,beads=beads,nui=nui,beadlibrary=beadlibrary,crosslibrary=crosslibrary,sitenames=sitenames)
    assert (eos_class._epsilonHB==epsilonHB).all()

def test_saft_gamma_mie_class_assoc_P(xi=xi_co2_h2o,T=T,eos=eos_co2_h2o,rho=rho_co2_h2o):
#   """Test ability to predict P with association sites"""
    P = eos.P(rho,T,xi)[0]
    assert P == pytest.approx(1713511.5399049097,abs=1e-1)

def test_saft_gamma_mie_class_assoc_mu(P=P,xi=xi_co2_h2o,T=T,eos=eos_co2_h2o,rho=rho_co2_h2o):
#   """Test ability to predict P with association sites"""
    mui = eos.chemicalpotential(rho,xi,T)
#    assert mui == pytest.approx(np.array([1.61884825, -4.09022886]),abs=1e-4)
    assert mui == pytest.approx(np.array([-12.27665471, -17.97013912]),abs=1e-4)


