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
xilist = [[0.002065, 0.997935]]
yilist = [[0.98779049, 0.01220951]]
Plist = [7000000.0]
rho_co2_h2o = np.array([19986.78358869])

bead_library = {
    "H2O353": {
        "epsilon": 479.56,
        "lambdaa": 6.0,
        "lambdar": 8.0,
        "sigma": 3.0029e-1,
        "Sk": 1.0,
        "Vks": 1,
        "mass": 0.018015,
    },
    "CO2": {
        "epsilon": 353.55,
        "lambdaa": 6.66,
        "lambdar": 23.0,
        "sigma": 3.741e-1,
        "Sk": 1.0,
        "Vks": 1,
        "mass": 0.04401,
    },
    "CH3CH2CH2-": {
        "Sk": 1.0,
        "Vks": 1.0,
        "epsilon": 342.0,
        "lambdaa": 6.0,
        "lambdar": 15.0,
        "mass": 0.04309,
        "sigma": 0.45089,
    },
}
cross_library = {
    "CO2": {"H2O353": {"epsilon": 432.69}},
    "H2O353": {"CH3CH2CH2-": {"epsilon": 315.7604284}},
}

beads_co2_h2o = ["CO2", "H2O353"]
molecular_composition_co2_h2o = np.array([[1.0, 0.0], [0.0, 1.0]])
Eos_co2_h2o = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=beads_co2_h2o,
    molecular_composition=molecular_composition_co2_h2o,
    bead_library=bead_library,
    cross_library=cross_library,
    numba=True,
)

beads_co2_h2o = ["H2O353", "CH3CH2CH2-"]
molecular_composition_co2_h2o = np.array([[1.0, 0.0], [0.0, 2.0]])
Eos_h2o_hexane = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=beads_co2_h2o,
    molecular_composition=molecular_composition_co2_h2o,
    bead_library=bead_library,
    cross_library=cross_library,
)


def test_thermo_import():
    #    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.thermodynamics" in sys.modules


def test_bubble_pressure(Eos=Eos_co2_h2o, Tlist=Tlist, xilist=xilist):
    output = thermo.thermo(
        Eos,
        calculation_type="bubble_pressure",
        **{"Tlist": Tlist, "xilist": xilist, "Pmin": [6900000], "Pmax": [7100000]}
    )
    assert output["P"][0] == pytest.approx(7005198.6, abs=5e1) and output["yi"][
        0
    ] == pytest.approx([0.98779049, 0.01220951], abs=1e-4)


def test_saturation_properties(Eos=Eos_co2_h2o, Tlist=Tlist):

    output = thermo.thermo(
        Eos,
        calculation_type="saturation_properties",
        **{"Tlist": Tlist, "xilist": [np.array([0.0, 1.0])]}
    )

    assert output["Psat"][0] == pytest.approx(46266.2, abs=1e1) and output["rhol"][
        0
    ] == pytest.approx(53883.63, abs=1e-1), output["rhol"][0] == pytest.approx(
        2371.38970066, abs=1e-1
    )


def test_liquid_properties(Eos=Eos_co2_h2o, Tlist=Tlist, xilist=xilist, Plist=Plist):

    output = thermo.thermo(
        Eos,
        calculation_type="liquid_properties",
        **{"Tlist": Tlist, "Plist": Plist, "xilist": xilist}
    )

    assert output["rhol"][0] == pytest.approx(53831.6, abs=1e-1) and output["phil"][
        0
    ] == pytest.approx(np.array([403.98, 6.8846e-03]), abs=1e-1)


def test_vapor_properties(Eos=Eos_co2_h2o, Tlist=Tlist, yilist=yilist, Plist=Plist):

    output = thermo.thermo(
        Eos,
        calculation_type="vapor_properties",
        **{"Tlist": Tlist, "Plist": Plist, "yilist": yilist}
    )

    assert output["rhov"][0] == pytest.approx(2938.3, abs=1e-1) and output["phiv"][
        0
    ] == pytest.approx(np.array([0.865397, 0.63848]), abs=1e-1)


def test_activity_coefficient(
    Eos=Eos_h2o_hexane, Tlist=Tlist, xilist=xilist, yilist=yilist, Plist=Plist
):

    output = thermo.thermo(
        Eos,
        calculation_type="activity_coefficient",
        **{"Tlist": Tlist, "Plist": Plist, "yilist": yilist, "xilist": xilist}
    )

    print(output["gamma"])
    assert output["gamma"][0] == pytest.approx(
        np.array([7.23733364e04, 6.30243983e-01]), abs=1e-2
    )
