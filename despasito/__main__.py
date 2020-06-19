"""
Process command line argparse and initiate logging settings.
"""

from .main import run, commandline_parser
from .utils.parallelization import MultiprocessingJob
import os
import logging
import logging.handlers

quiet = False

args = commandline_parser()

## Extract arguments
if args.verbose == 0:
    quiet = True
    args.verbose = 20
elif args.verbose < 4:
    args.verbose = (4-args.verbose)*10
else:
    args.verbose = 10

# Logging
logger = logging.getLogger()
logger.setLevel(args.verbose)

# Set up rotating log files
if os.path.isfile(args.logFile):
    os.remove(args.logFile)
log_file_handler = logging.handlers.RotatingFileHandler(args.logFile)
log_file_handler.setFormatter( logging.Formatter('%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d): %(message)s') )
log_file_handler.setLevel(args.verbose)
logger.addHandler(log_file_handler)

if quiet == False:
    # Set up logging to console
    console_handler = logging.StreamHandler() # sys.stderr
    console_handler.setFormatter( logging.Formatter('[%(levelname)s](%(name)s): %(message)s') )
    console_handler.setLevel(args.verbose)
    logger.addHandler(console_handler)

logging.info("Input args: {}".format(args))

# Update flags for optimization methods 
logging.info("Use Numba JIT: {}".format(args.numba))
logging.info("Use Cython: {}".format(args.cython))
logging.info("Pure Python (no fortran): {}".format(args.python))

# Run program
if args.input:
    kwargs = {"filename":args.input}
else:
    kwargs = {}

kwargs["mpObj"] = MultiprocessingJob(ncores=args.ncores)
kwargs["ncores"] = args.ncores
kwargs["path"] = args.path

run(**kwargs)
