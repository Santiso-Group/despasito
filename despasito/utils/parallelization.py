
import multiprocessing
import logging
import os
import glob

logger = logging.getLogger(__name__)

#class MultiprocessingJob:
#
#    """
#    This object initiates the pool for multiprocessing jobs.
#
#    Parameters
#    ----------
#    ncores : int, Optional, default: -1
#        Number of processes used. If the default value of -1, the system cpu count is used.
#    logger : class, Optional, default: None
#        The logger object used.
#
#    Attributes
#    ----------
#    pool : function
#        This pool is used parallelize jobs
#    """
#
#    def __init__(self, ncores=-1, logger=None, logname="despasito.log"):
#
#        if ncores == -1:
#            ncores = multiprocessing.cpu_count() # includes logical cores!
#            logger.info(f'Detected {ncores} cores')
#        else:
#            logger.info(f'Number of cores set to {ncores}')
#
#        self.ncores = ncores
#        self.pool = multiprocessing.Pool(ncores)
#
#    def pool_job(self, func, inputs):
#        """
#        This function will setup and dispatch thermodynamic jobs.
#
#        Parameters
#        ----------
#        func : function
#            Function used in job
#        inputs : list
#            Each entry of this list contains the input arguements for each job
#
#        Returns
#        -------
#        output : tuple
#            This structure contains the outputs of the jobs given
#
#        """
#
#        initialize_workers()
#        output, log_msg = zip(*self.pool.map(worker_process(func), inputs))
#
#        return output
#
#    def initialize_mp_handler():
#        """Wraps the handlers in the given Logger with an MultiProcessingHandler.
#
#        :param logger: whose handlers to wrap. By default, the root logger.
#        """
#        if logger is None:
#            logger = logging.getLogger()
#
#        for i, orig_handler in enumerate(list(logger.handlers)):
#
#            handler = MultiProcessingHandler(
#                'mp-handler-{0}'.format(i), sub_handler=orig_handler)
#
#            logger.removeHandler(orig_handler)
#            logger.addHandler(handler)
# 
#
#    def consolidate_logs():
#
#        for i in range(ncores):
#            # append lagname with multiprocess log i
#            # delete that log
#
#    def end_pool():
#        """
#        Close multiprocessing pool
#        """
#        self.pool.close()
#        self.pool.join()

def initialize_MY_mp_handler():
    """Wraps the handlers in the given Logger with an MultiProcessingHandler.

    :param logger: whose handlers to wrap. By default, the root logger.
    """
    logger = logging.getLogger()
    pid = os.getpid()
    filename = 'mp-handler-{0}.log'.format(pid)
    handler = logging.handlers.RotatingFileHandler(filename)
    logger.addHandler(handler)

def batch_jobs( func, inputs, ncores=1, logger=None):
    """
    This function will setup and dispatch thermodynamic jobs.

    Parameters
    ----------
    func : function
        Function used in job
    inputs : list
        Each entry of this list contains the input arguements for each job
    ncores : int, Optional, default: 1
        Number of processes used.
    logger : class, Optional, default: None
        The logger object used.

    Returns
    -------
    output : tuple
        This structure contains the outputs of the jobs given

    """

    if logger == None:
        logger = logging.getLogger()

    ## New rotating handling (In CH3OH case, we don't need to worry about deleting during a run)
    pool = multiprocessing.Pool(ncores, initializer=initialize_MY_mp_handler) 

    ## Run Multiprocessing
    output = zip(*pool.map(func, inputs))

    for i, fn in enumerate(glob.glob('./mp-handler-*.log')):
        with open(fn) as f:
           logger.info("Log from thread {0}:\n{1}".format(i, f.read()))
        os.remove(fn)

    return output

