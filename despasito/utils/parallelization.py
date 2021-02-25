"""

Parallelization class to handle processing threads and logging.

"""

import sys
import numpy as np
import multiprocessing
import logging
import logging.handlers
import os
import glob

logger = logging.getLogger(__name__)


class MultiprocessingJob:

    """
    This object initiates the pool for multiprocessing jobs.

    Parameters
    ----------
    ncores : int, Optional, default=-1
        Number of processes used. If the default value of -1, the system cpu count is used.

    Attributes
    ----------
    pool : function
        This pool is used parallelize jobs
    """

    def __init__(self, ncores=-1):

        self.flag_use_mp = True
        if ncores == -1:
            ncores = multiprocessing.cpu_count()  # includes logical cores!
            logger.info("Detected {} cores".format(ncores))
        elif ncores > 1:
            logger.info("Number of cores set to {}".format(ncores))
        elif ncores == 1:
            self.flag_use_mp = False
            logger.info(
                "Number of cores set to 1, bypassing mp and using serial methods"
            )
        else:
            raise ValueError("Number of cores cannot be zero or negative.")

        self.ncores = ncores

        if self.flag_use_mp:

            # Remove old mp logs
            self._extract_root_logging()

            # Initiate multiprocessing
            ctx = multiprocessing.get_context("spawn")
            self._pool = ctx.Pool(
                ncores,
                initializer=self._initialize_mp_handler,
                initargs=(self._level, self._logformat),
            )

            self.logfiles = []
            for worker in self._pool._pool:
                filename = "mp-handler-{0}.log".format(worker.pid)
                self.logfiles.append(filename)
            logger.info("MP log files: {}".format(", ".join(self.logfiles)))

    def _extract_root_logging(self):
        """ Swap root handlers defined in despasito.__main__ with process specific log handlers
        """
        for handler in logging.root.handlers:
            if "baseFilename" in handler.__dict__:
                self._logformat = handler.formatter._fmt
                self._level = handler.level

        if not hasattr(self, "_logformat"):
            self._logformat = None
            self._level = None

    @staticmethod
    def _initialize_mp_handler(level, logformat):
        """Wraps the handlers in the given Logger with an MultiProcessingHandler.

        Parameters
        ----------
        level : int
            The verbosity level of logging information can be set to any supported representation of the `logging level <https://docs.python.org/3/library/logging.html#logging-levels>`_. 
        logformat : str
            Formating of logging information can be set to any supported representation of the `formatting class <https://docs.python.org/3/library/logging.html#logging.Formatter>`_. 
        """

        logger = logging.getLogger()

        pid = os.getpid()
        filename = "mp-handler-{0}.log".format(pid)
        handler = logging.handlers.RotatingFileHandler(filename)
        if level is not None:
            logger.setLevel(level)
            handler.setLevel(level)
        if logformat is not None:
            handler.setFormatter(logging.Formatter(logformat))

        logger.addHandler(handler)

    def pool_job(self, func, inputs):
        """
        This function will setup and dispatch thermodynamic or parameter fitting jobs.

        Parameters
        ----------
        func : function
            Function used in job
        inputs : list
            Each entry of this list contains the input arguments for each job

        Returns
        -------
        output : tuple
            This structure contains the outputs of the jobs given

        """

        if self.flag_use_mp:
            output = zip(*self._pool.map(func, inputs))
            self._consolidate_mp_logs()
        else:
            logger.info("Performing task serially")
            output = self.serial_job(func, inputs)

        return output

    @staticmethod
    def serial_job(func, inputs):
        """
        This function will serially perform thermodynamic jobs.

        Parameters
        ----------
        func : function
            Function used in job
        inputs : list
            Each entry of this list contains the input arguments for each job

        Returns
        -------
        output : tuple
            This structure contains the outputs of the jobs given

        """

        output = []
        for i, finput in enumerate(inputs):
            foutput = func(finput)
            output.append(foutput)

        return np.transpose(output)

    def _consolidate_mp_logs(self):
        """ Consolidate multiprocessing logs into main log
        """
        for i, fn in enumerate(self.logfiles):
            with open(fn) as f:
                logger.info("Log from thread {0}:\n{1}".format(i, f.read()))
            open(fn, "w").write("")

    def _remove_mp_logs(self):
        """ Ensure all previous mp logs are removed
        """
        for i, fn in enumerate(self.logfiles):
            os.remove(fn)

    def end_pool(self):
        """ Close multiprocessing pool
        """
        if self.flag_use_mp:
            self._pool.close()
            self._pool.join()
            self._remove_mp_logs()


def initialize_mp_handler(level, logformat):
    """ Wraps the handlers in the given Logger with an MultiProcessingHandler.

    Parameters
    ----------
    level : int
        The verbosity level of logging information can be set to any supported representation of the `logging level <https://docs.python.org/3/library/logging.html#logging-levels>`_. 
    logformat : str
        Formating of logging information can be set to any supported representation of the `formatting class <https://docs.python.org/3/library/logging.html#logging.Formatter>`_. 
    """

    logger = logging.getLogger()

    pid = os.getpid()
    filename = "mp-handler-{0}.log".format(pid)
    handler = logging.handlers.RotatingFileHandler(filename)

    handler.setFormatter(logging.Formatter(logformat))
    handler.setLevel(level)

    logger.addHandler(handler)


def batch_jobs(func, inputs, ncores=1, logger=None):
    """
    This function will setup and dispatch thermodynamic jobs.

    Parameters
    ----------
    func : function
        Function used in job
    inputs : list
        Each entry of this list contains the input arguments for each job
    ncores : int, Optional, default=1
        Number of processes used.
    logger : class, Optional, default=None
        The logger object used.

    Returns
    -------
    output : tuple
        This structure contains the outputs of the jobs given

    """

    if logger is None:
        logger = logging.getLogger()

    root_handlers = logging.root.handlers
    for handler in root_handlers:
        if "baseFilename" in handler.__dict__:
            logformat = handler.formatter._fmt
            level = handler.level
    logging.root.handlers = []

    pool = multiprocessing.Pool(
        ncores, initializer=initialize_mp_handler, initargs=(level, logformat)
    )

    output = zip(*pool.map(func, inputs))

    logging.root.handlers = root_handlers

    for i, fn in enumerate(glob.glob("./mp-handler-*.log")):
        with open(fn) as f:
            logger.info("Log from thread {0}:\n{1}".format(i, f.read()))
        os.remove(fn)

    return output
