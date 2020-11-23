"""
DESPASITO
DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output
"""

# Add imports here
from .main import run
#from .main import method_stat

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
