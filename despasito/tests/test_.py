
import pytest
from pip._internal.operations.freeze import FrozenRequirement
import pkg_resources

# Taken from https://stackoverflow.com/questions/40530000/check-if-my-application-runs-in-development-editable-mode

def test_editable():

    distributions = {v.key: v for v in pkg_resources.working_set}
    for key, value in distributions.items():
        print(key, value)
    distribution = distributions['despasito']
    frozen_requirement = FrozenRequirement.from_dist(distribution)

    if not frozen_requirement.editable:
        pytest.exit("Tests should only be run when installed as editable: `pip install -e .`, consider cloning the repository on GitHub")
    else:
        assert True
