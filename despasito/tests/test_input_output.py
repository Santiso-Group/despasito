"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.input_output as d_io
import pytest
import sys

@pytest.mark.parametrize('key, answer', [("rhoinc",2.0),("minrhofrac",2.5e-06)])
def test_file2paramdict(key,answer):
    """Test conversion of txt file to dictionary"""
    rho_dict = d_io.readwrite_input.file2paramdict("examples/dens_params.txt")
    assert rho_dict[key] == pytest.approx(answer,abs=1e-7)


#def test_extract_calc_data(fname):
#    """Unit test of extracting all data"""
#    calctype, eos_dict, thermo_dict = extract_calc_data(input_fname)
#    assert 

