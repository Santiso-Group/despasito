"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import copy
import os
import pytest
import sys
import numpy as np

import despasito.equations_of_state

xi_co2_ben = np.array([0.2, 0.2])
beads_co2_ben = ["CO2", "benzene"]
molecular_composition_co2_ben = np.array([[1.0, 0.0], [0.0, 1.0]])
bead_library_co2_ben = {
    "CO2": {
        "epsilon": 361.69,
        "lambdaa": 6.66,
        "lambdar": 23.0,
        "sigma": 3.741e-1,
        "Sk": 1.0,
        "Vks": 1.0,
        "mass": 0.04401,
    },
    "baCH": {
        "epsilon": 371.53,
        "lambdaa": 6.0,
        "lambdar": 14.756,
        "sigma": 4.0578e-1,
        "Sk": 0.32184,
        "Vks": 1.0,
        "mass": 0.01302,
    },
    "baCHCH": {
        "epsilon": 243.88,
        "lambdaa": 6.0,
        "lambdar": 11.58,
        "sigma": 3.482e-1,
        "Sk": 1.0,
        "Vks": 1.0,
        "mass": 0.02604,
    },
    "baCHCHCH": {
        "epsilon": 353.93,
        "lambdaa": 6.0,
        "lambdar": 14.23,
        "sigma": 3.978e-1,
        "Sk": 1.0,
        "Vks": 1.0,
        "mass": 0.03905,
    },
    "benzene": {
        "epsilon": 658.17,
        "lambdaa": 6.0,
        "lambdar": 32.0,
        "sigma": 3.842e-1,
        "Sk": 1.0,
        "Vks": 1.0,
        "mass": 0.07811,
    },
}

xi_co2_h2o = np.array([0.78988277, 0.21011723])
beads_co2_h2o = ["CO2", "H2O"]
molecular_composition_co2_h2o = np.array([[1.0, 0.0], [0.0, 1.0]])
bead_library_co2_h2o = {
    "H2O": {
        "epsilon": 266.68,
        "lambdaa": 6.0,
        "lambdar": 17.02,
        "sigma": 3.0063e-1,
        "Sk": 1.0,
        "Vks": 1,
        "mass": 0.018015,
        "Nk-H": 2,
        "Nk-e1": 2,
        "epsilonHB-H-e1": 1985.4,
        "K-H-e1": 1.0169e-1,
    },
    "CO2": {
        "epsilon": 207.89,
        "lambdaa": 5.055,
        "lambdar": 26.408,
        "sigma": 3.05e-1,
        "Sk": 0.8468,
        "Vks": 2,
        "mass": 0.04401,
        "Nk-H": 1,
        "Nk-a1": 1,
    },
}
cross_library_co2_h2o = {
    "CO2": {"H2O": {"epsilon": 226.38, "epsilonHB-H-e1": 2200.0, "K-H-e1": 9.1419e-2}}
}
epsilonHB_co2_h2o = np.array(
    [
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 2200.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2200.0, 0.0, 0.0]],
            [[0.0, 0.0, 1985.4], [0.0, 0.0, 0.0], [1985.4, 0.0, 0.0]],
        ],
    ]
)

T = 323.2
rho_co2_h2o = np.array([21146.16997993])
P = np.array([15727315.77])

def test_saft_gamma_mie_imported():
    #    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.equations_of_state" in sys.modules


def test_saft_gamma_mie_class_noassoc(
    beads=beads_co2_ben,
    molecular_composition=molecular_composition_co2_ben,
    bead_library=bead_library_co2_ben,
):
    #   """Test ability to create EOS object without association sites"""
    Eos_class = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_mie",
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=copy.deepcopy(bead_library),
    )
    assert (Eos_class.eos_dict["massi"] == np.array([0.04401, 0.07811])).all()

@pytest.mark.skipif(hasattr(sys, 'getwindowsversion'), reason="Issue with f2py Fortran modules on Windows")
def test_fortran_available():
    from despasito.equations_of_state.saft.compiled_modules import ext_Aassoc_fortran
    try:
        from despasito.equations_of_state.saft.compiled_modules import ext_Aassoc_fortran
        flag = True
    except Exception:
        flag = False

    assert flag


def test_saft_gamma_mie_class_assoc(
    beads=beads_co2_h2o,
    molecular_composition=molecular_composition_co2_h2o,
    bead_library=bead_library_co2_h2o,
    cross_library=cross_library_co2_h2o,
    epsilonHB=epsilonHB_co2_h2o,
):
    #   """Test ability to create EOS object with association sites"""
    Eos_class = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_mie",
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=copy.deepcopy(bead_library),
        cross_library=copy.deepcopy(cross_library),
    )
    assert (Eos_class.eos_dict["epsilonHB"] == epsilonHB).all()


def test_saft_gamma_mie_class_assoc_P(
    T=T, 
    xi=xi_co2_h2o, 
    rho=rho_co2_h2o,
    beads=beads_co2_h2o,
    molecular_composition=molecular_composition_co2_h2o,
    bead_library=bead_library_co2_h2o,
    cross_library=cross_library_co2_h2o,
):
    #   """Test ability to create EOS object with association sites"""
    Eos_class = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_mie",
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=copy.deepcopy(bead_library),
        cross_library=copy.deepcopy(cross_library),
    )
    #   """Test ability to predict P with association sites"""
    P = Eos_class.pressure(rho, T, xi)[0]
    assert P == pytest.approx(15727315.77, abs=1e3)


def test_saft_gamma_mie_class_assoc_fugacity_coeff(
    P=P,
    T=T, 
    xi=xi_co2_h2o, 
    rho=rho_co2_h2o,
    beads=beads_co2_h2o,
    molecular_composition=molecular_composition_co2_h2o,
    bead_library=bead_library_co2_h2o,
    cross_library=cross_library_co2_h2o,
):
    #   """Test ability to create EOS object with association sites"""
    Eos_class = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_mie",
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=copy.deepcopy(bead_library),
        cross_library=copy.deepcopy(cross_library),
    )
    #   """Test ability to predict P with association sites"""
    phi = Eos_class.fugacity_coefficient(P, rho, xi, T)
    assert phi == pytest.approx(np.array([0.48972481, 0.00281112]), abs=1e-4)

def test_numba_available():

    try:
        from despasito.equations_of_state.saft.compiled_modules.ext_Aassoc_numba import (
            calc_Xika,
        )
        from despasito.equations_of_state.saft.compiled_modules.ext_gamma_mie_numba import (
            calc_a1s,
            calc_Bkl,
            calc_a1ii,
            calc_a1s_eff,
            calc_Bkl_eff,
            calc_da1iidrhos,
            calc_da2ii_1pchi_drhos,
        )

        flag = True
    except Exception:
        flag = False

    assert flag

def test_saft_gamma_mie_class_assoc_P_numba(
    T=T,
    xi=xi_co2_h2o,
    rho=rho_co2_h2o,
    beads=beads_co2_h2o,
    molecular_composition=molecular_composition_co2_h2o,
    bead_library=bead_library_co2_h2o,
    cross_library=cross_library_co2_h2o,
):
#   """Test ability to predict P with association sites"""
    Eos = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_mie",
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=copy.deepcopy(bead_library),
        cross_library=copy.deepcopy(cross_library),
        numba=True
    )

#   """Test ability to predict P with association sites"""
    P = Eos.pressure(rho,T,xi)[0]
    assert P == pytest.approx(15727315.77,abs=1e+3)

def test_cython_available():

    from despasito.equations_of_state.saft.compiled_modules.ext_Aassoc_cython import (
        calc_Xika,
    )
    from despasito.equations_of_state.saft.compiled_modules.ext_gamma_mie_cython import (
        calc_a1s,
        calc_Bkl,
        calc_a1ii,
        calc_a1s_eff,
        calc_Bkl_eff,
        calc_da1iidrhos,
        calc_da2ii_1pchi_drhos,
    )

    try:
        from despasito.equations_of_state.saft.compiled_modules.ext_Aassoc_cython import (
            calc_Xika,
        )
        from despasito.equations_of_state.saft.compiled_modules.ext_gamma_mie_cython import (
            calc_a1s,
            calc_Bkl,
            calc_a1ii,
            calc_a1s_eff,
            calc_Bkl_eff,
            calc_da1iidrhos,
            calc_da2ii_1pchi_drhos,
        )

        flag = True
    except Exception:
        flag = False

    assert flag

@pytest.mark.skip(reason="Cython does not produce the correct result with pytest, allow will in examples")
def test_saft_gamma_mie_class_assoc_P_cython(
    T=T,
    xi=xi_co2_h2o,
    rho=rho_co2_h2o,
    beads=beads_co2_h2o,
    molecular_composition=molecular_composition_co2_h2o,
    bead_library=bead_library_co2_h2o,
    cross_library=cross_library_co2_h2o,
):
#   """Test ability to predict P with association sites"""
    Eos = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_mie",
        beads=beads,
        molecular_composition=molecular_composition,
        bead_library=copy.deepcopy(bead_library),
        cross_library=copy.deepcopy(cross_library),
        cython=True
    )
    P = Eos.pressure(rho,T,xi)[0]
    assert P == pytest.approx(15727315.77,abs=1e+3)

