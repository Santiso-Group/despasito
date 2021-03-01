"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito
import pytest
import logging
import random
import os

logger = logging.getLogger(__name__)


def test_despasito_log_file():
    """Test enabling of logging"""

    fname = "despasito_{}.log".format(random.randint(1, 10))
    despasito.initiate_logger(log_file=fname, verbose=10)
    logger.info("test")

    if os.path.isfile(fname):
        flag = True
        despasito.initiate_logger(log_file=False)
        try:
            os.remove(fname)
        except:
            print("Error removing log file")
    else:
        flag = False

    assert flag


def test_despasito_log_console(capsys):
    """Test enabling of logging"""

    despasito.initiate_logger(console=True, verbose=10)
    logger.info("test")

    _, err = capsys.readouterr()

    despasito.initiate_logger(console=False)

    assert "[INFO](despasito.tests.test_logging): test" in err
