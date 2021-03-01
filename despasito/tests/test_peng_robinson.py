"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.equations_of_state.cubic.peng_robinson

import pytest
import sys
import numpy as np

xi = np.array([0.827, 0.173])
molecular_composition = np.array([[1.0, 0.0], [0.0, 1.0]])
beads = ["acetone", "chloroform"]
bead_library = {
    "acetone": {"Tc": 508.1, "Pc": 4690000.0, "omega": 0.304},
    "chloroform": {"Tc": 536.4, "Pc": 5471550.0, "omega": 0.221902},
}
cross_library = {"acetone": {"chloroform": {"kij": -0.0605}}}

Eos = despasito.equations_of_state.initiate_eos(
    eos="cubic.peng_robinson",
    xi=xi,
    beads=beads,
    molecular_composition=molecular_composition,
    bead_library=bead_library,
    cross_library=cross_library,
)
T = 332.15
P = np.array([101330.0])
yi = np.array([0.89, 0.11])
rho = np.array([12546.22])


def test_peng_robinson_imported():
    #    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.equations_of_state.cubic.peng_robinson" in sys.modules


def test_PR_coefficients(
    xi=xi,
    beads=beads,
    molecular_composition=molecular_composition,
    bead_library=bead_library,
):  #   """Test ability to create EOS object without association sites"""
    Eos_class = despasito.equations_of_state.initiate_eos(
        eos="cubic.peng_robinson",
        xi=xi,
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=bead_library,
    )
    tmp = [
        Eos_class.bead_library[beads[0]]["kappa"],
        Eos_class.bead_library[beads[1]]["kappa"],
    ]
    assert (
        Eos_class.eos_dict["ai"]
        == pytest.approx(np.array([1.73993846, 1.66217026]), abs=1e-4)
        and Eos_class.eos_dict["bi"]
        == pytest.approx(np.array([7.00758212e-05, 6.34118233e-05]), abs=1e-9)
        and tmp == pytest.approx(np.array([0.81854211, 0.70357958]), abs=1e-4)
    )


def test_peng_robinson_pressure(xi=xi, T=T, Eos=Eos, rho=rho):
    #   """Test ability to predict P with association sites"""
    P = Eos.pressure(rho, T, xi)[0]
    assert P == pytest.approx(69904905.698, abs=1e-1)


def test_peng_robinson_class_assoc_mu(P=P, xi=xi, T=T, Eos=Eos, rho=rho):
    #   """Test ability to predict P with association sites"""
    phi = Eos.fugacity_coefficient(P, rho, xi, T)
    #    assert mui == pytest.approx(np.array([1.61884825, -4.09022886]),abs=1e-4)
    assert phi == pytest.approx(np.array([1.12643785, 0.55712584]), abs=1e-4)
