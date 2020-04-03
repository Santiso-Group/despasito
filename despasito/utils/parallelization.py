
import multiprocessing
from multiprocessing_logging import install_mp_handler

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
        Number of processes used
    logger : class, Optional, default: None
        The logger object used.

    Returns
    -------
    output : tuple
        This structure contains the outputs of the jobs given

    """

    install_mp_handler(logger=logger)
    pool = multiprocessing.Pool(ncores, initializer=install_mp_handler) 
    output = zip(*pool.map(func, inputs))

    return output

