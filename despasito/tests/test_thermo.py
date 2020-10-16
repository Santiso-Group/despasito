"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.thermodynamics.calc as calc
import despasito.thermodynamics as thermo
import despasito.equations_of_state
import pytest
import sys
import numpy as np

Tlist = [353.0]
xilist = [[0.002064,0.997935]]
yilist = [[0.987755,0.012245]]
Plist = [7000000.0]
rho_co2_h2o = np.array([19986.78358869])

beads_co2_h2o = ['CO2', 'H2O353']
nui_co2_h2o = np.array([[1., 0.],[0., 1.]])
beadlibrary_co2_h2o = {'H2O353': {'epsilon': 479.56, 'l_a': 6.0, 'l_r': 8.0, 'sigma': 3.0029e-1, 'Sk': 1.0, 'Vks': 1, 'mass': 0.018015}, 'CO2': {'epsilon': 353.55, 'l_a': 6.66, 'l_r': 23.0, 'sigma': 3.741e-1, 'Sk': 1.0, 'Vks': 1, 'mass': 0.04401}}
crosslibrary_co2_h2o = {'CO2': {'H2O353': {'epsilon': 432.69}}}
eos_co2_h2o = despasito.equations_of_state.eos(eos="saft.gamma_mie",beads=beads_co2_h2o,nui=nui_co2_h2o,beadlibrary=beadlibrary_co2_h2o,crosslibrary=crosslibrary_co2_h2o, jit=True)

def test_thermo_import():
#    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.thermodynamics" in sys.modules

def test_phase_xiT(eos=eos_co2_h2o,Tlist=Tlist,xilist=xilist):
    output = thermo.thermo(eos,{"calculation_type":"phase_xiT","Tlist":Tlist,"xilist":xilist, "Pmin": [6900000], "Pmax":[7100000]})
    assert output["P"][0]==pytest.approx(7000914.5,abs=1e+1) and output["yi"][0]==pytest.approx([0.98779049, 0.01220951],abs=1e-4)

#def test_phase_yiT(eos=eos_co2_h2o,Tlist=Tlist,yilist=yilist):
#    output = thermo.thermo(eos,{"calculation_type":"phase_yiT","Tlist":Tlist,"yilist":yilist})
#    assert output["P"][0]==pytest.approx(6926411.28,abs=1e+1) and output["xi"][0]==pytest.approx([0.26451786, 0.73548214],abs=1e-4)

def test_saturation_properties(eos=eos_co2_h2o,Tlist=Tlist):

    output = thermo.thermo(eos,{"calculation_type":"saturation_properties","Tlist":Tlist,"xilist":[np.array([0.0, 1.0])]})

    assert output["Psat"][0]==pytest.approx(46266.2,abs=1e+1) and output["rhol"][0]==pytest.approx(53883.63,abs=1e-1), output["rhol"][0]==pytest.approx(2371.38970066,abs=1e-1)

def test_liquid_properties(eos=eos_co2_h2o,Tlist=Tlist,xilist=xilist,Plist=Plist):

    output = thermo.thermo(eos,{"calculation_type":"liquid_properties","Tlist":Tlist,"Plist":Plist,"xilist":xilist})

#    assert output["rhol"][0]==pytest.approx(54072.87630577754,abs=1e-1) and output["phil"][0]==pytest.approx(np.array([2646.44010, 0.120295122]),abs=1e-1)
    assert output["rhol"][0]==pytest.approx(53831.7,abs=1e-1) and output["phil"][0]==pytest.approx(np.array([403.98, 6.8846e-03]),abs=1e-1)

def test_vapor_properties(eos=eos_co2_h2o,Tlist=Tlist,yilist=yilist,Plist=Plist):

    output = thermo.thermo(eos,{"calculation_type":"vapor_properties","Tlist":Tlist,"Plist":Plist,"yilist":yilist})

#    assert output["rhov"][0]==pytest.approx(37.85937201,abs=1e-1) and output["phiv"][0]==pytest.approx(np.array([2.45619145, 0.37836741]),abs=1e-1)
    assert output["rhov"][0]==pytest.approx(2938.3,abs=1e-1) and output["phiv"][0]==pytest.approx(np.array([0.865397, 0.63848]),abs=1e-1)
    
