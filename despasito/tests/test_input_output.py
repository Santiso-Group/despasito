"""
Regression tests for the despasito input/output package.

The following functions are not tested because they require importing a file:
    file2paramdict, extract_calc_data, process_commandline, make_xi_matrix

The following functions are not tested because they require exporting a file:
    write_EOSparameters

The following functions are tested:
    process_bead_data
"""

# Import package, test suite, and other packages as needed
import despasito.input_output.read_input as ri
import pytest
import numpy as np

# Not Used because we shouldn't reference an external file
@pytest.mark.parametrize("data", [([[["CH4_2", 1]], [["eCH3", 2]]])])
def test_process_bead_data(data):
    """Test extraction of system component information"""

    beads, molecular_composition = ri.process_bead_data(data)

    errors = []
    if not set(beads) == set(["CH4_2", "eCH3"]):
        errors.append("beads: %s is not %s" % (str(beads), str(["CH4_2", "eCH3"])))
    elif not np.array_equal(molecular_composition, np.array([[1.0, 0.0], [0.0, 2.0]])):
        errors.append(
            "molecular_composition: %s is not %s"
            % (str(molecular_composition), str(np.array([[1.0, 0.0], [0.0, 2.0]])))
        )

    assert not errors, "errors occured:\n{}".format("\n".join(errors))


# Not Used because we shouldn't reference an external file
# @pytest.mark.parametrize('key, answer', [("density_increment",2.0),("min_density_fraction",2.5e-06)])
# def test_file2paramdict(key,answer):
#    """Test conversion of txt file to dictionary"""
#    rho_dict = ri.file2paramdict("example/dens_params.txt")
#    assert rho_dict[key] == pytest.approx(answer,abs=1e-7)
