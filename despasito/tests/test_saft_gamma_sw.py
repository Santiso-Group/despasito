"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.equations_of_state

# import despasito.equations_of_state.saft.solv_assoc as solv_assoc
import copy

import pytest
import sys
import numpy as np

bead = ["H2O"]
molecular_composition = np.array([[1.0]])
bead_library = {
    "H2O": {
        "epsilon": 250.0,
        "lambda": 1.7889,
        "sigma": 3.0342e-1,
        "Sk": 1.0,
        "Vks": 1.0,
        "Nk-e": 2,
        "Nk-H": 2,
        "epsilonHB-e-H": 1400.0,
        "rc-e-H": 0.210822,
        "mass": 0.018015,
    }
}

Kklab = np.array([[[[[[0.0, 0.00106673], [0.00106673, 0.0]]]]]])

Eos = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_sw",
    beads=bead,
    molecular_composition=molecular_composition,
    bead_library=copy.deepcopy(bead_library),
)
T = 580.0
density = np.array([40004.84463113798])
P = np.array([9447510.360679299])


def test_saft_gamma_sw_class_bonding_volume(
    T=T,
    beads=bead,
    molecular_composition=molecular_composition,
    bead_library=bead_library,
    Kklab=Kklab,
):
    #   """Test ability to create EOS object with association sites"""
    Eos_class = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_sw",
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=copy.deepcopy(bead_library),
    )

    rc_klab = Eos_class.eos_dict["rc_klab"]
    Kklab_new = Eos_class.saft_source.calc_Kijklab(T, rc_klab)

    assert Kklab_new == pytest.approx(Kklab, abs=1e-7)


def test_saft_gamma_sw_class_assoc_P(T=T, xi=[1.0], Eos=Eos, density=density):
    #   """Test ability to predict P with association sites"""
    P = Eos.pressure(density, T, xi)[0]
    assert P == pytest.approx(9447510.360679299, abs=1e3)


def test_saft_gamma_sw_class_assoc_mu(P=P, xi=[1.0], T=T, Eos=Eos, density=density):
    #   """Test ability to predict P with association sites"""
    phi = Eos.fugacity_coefficient(P, density, xi, T)
    assert phi == pytest.approx(np.array([0.8293442]), abs=1e-4)

Eos = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_sw",
    beads=bead,
    molecular_composition=molecular_composition,
    bead_library=copy.deepcopy(bead_library),
    numba=True
)

def test_saft_gamma_sw_class_assoc_P_numba(T=T, xi=np.array([1.0]), Eos=Eos, density=density):
    #   """Test ability to predict P with association sites"""
    P = Eos.pressure(density, T, xi)[0]
    assert P == pytest.approx(9447510.360679299, abs=1e3)

