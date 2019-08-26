"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.equations_of_state.saft
import pytest
import sys

#xi = [0.2 0.2]
#beads = ['CO2', 'benzene']
#nui = [[1. 0.],[0. 1.]]
#beadlibrary = {'CO2': {'epsilon': 361.69, 'l_a': 6.66, 'l_r': 23.0, 'sigma': 3.741e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.04401}, 'baCH': {'epsilon': 371.53, 'l_a': 6.0, 'l_r': 14.756, 'sigma': 4.0578e-10, 'Sk': 0.32184, 'Vks': 1.0, 'mass': 0.01302}, 'baCHCH': {'epsilon': 243.88, 'l_a': 6.0, 'l_r': 11.58, 'sigma': 3.482e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.02604}, 'baCHCHCH': {'epsilon': 353.93, 'l_a': 6.0, 'l_r': 14.23, 'sigma': 3.978e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.03905}, 'benzene': {'epsilon': 658.17, 'l_a': 6.0, 'l_r': 32.0, 'sigma': 3.842e-10, 'Sk': 1.0, 'Vks': 1.0, 'mass': 0.07811}}

#def test_saft_gamma_mie_imported():
#    """Sample test, will always pass so long as import statement worked"""
#    assert "despasito.equations_of_state.saft.gamma_mie" in sys.modules

#def test_saft_gamma_mie_class():
    
#    eos_class = despasito.equations_of_state.eos("saft.gamma_mie",)
    
#    assert "saft.gamma_mie" in sys.modules
