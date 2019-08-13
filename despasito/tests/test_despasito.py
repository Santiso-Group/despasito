"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito
import pytest
import sys

def test_despasito_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "despasito" in sys.modules
