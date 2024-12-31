"""
DESPASITO
DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT,
Intended for Thermodynamic Output
"""

import os
import logging
import logging.handlers
from ._version import __version__ as __version__

logger = logging.getLogger("despasito")
logger.setLevel(30)


def initiate_logger(console=None, log_file=None, verbose=30):
    """
    Initiate a logging handler if more detail on the calculations is desired.

    This function is useful when DESPASITO is used as an imported package. If a handler
    of the given type is already present, nothing is done, as is the case when the
    input file schema is used. If either handler is given a value of False, the handler
    of that type is removed.

    Parameters
    ----------
    console : bool, Optional, default=None
        Initiates a stream handler to print to a console. If True, this handler is
        initiated. If it is False, then any StreamHandler is removed.
    log_file : bool/str, Optional, default=None
        If log output should be recorded in a file, set this keyword to either True or
        to a name for the log file. If True, the file name 'despasito.log' is used.
        Note that if a file with the same name already exists, it will be deleted.
    verbose : int, Optional, default=30
        The verbosity of logging information can be set to any supported representation
        of the
        `logging level <LOGGER>`_.

    .. _LOGGER: https://docs.python.org/3/library/logging.html#logging-levels
    """

    logger.setLevel(verbose)

    # Check for existing handlers
    handler_console = None
    handler_logfile = None
    for tmp in logger.handlers:
        if "RotatingFileHandler" in str(tmp):
            handler_logfile = tmp
        if "StreamHandler" in str(tmp):
            handler_console = tmp

    # Set up logging to console
    if console and handler_console is None:
        console_handler = logging.StreamHandler()  # sys.stderr
        console_handler.setFormatter(logging.Formatter("[%(levelname)s](%(name)s): %(message)s"))
        console_handler.setLevel(verbose)
        logger.addHandler(console_handler)
    elif console:
        logger.warning("StreamHandler already exists")
    elif handler_console is not None:
        handler_console.close()
        logger.removeHandler(handler_console)

    # Rotating File Handler
    if log_file is not None and handler_logfile is None:

        if not isinstance(log_file, str):
            log_file = "despasito.log"

        if os.path.isfile(log_file):
            os.remove(log_file)

        log_file_handler = logging.handlers.RotatingFileHandler(log_file)
        log_file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d): " + "%(message)s")
        )
        log_file_handler.setLevel(verbose)
        logger.addHandler(log_file_handler)
    elif log_file:
        logger.warning("RotatingFileHandler already exists")
    elif not handler_logfile and handler_logfile is not None:
        handler_logfile.close()
        logger.removeHandler(handler_logfile)
