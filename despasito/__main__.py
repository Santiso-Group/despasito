"""
Process command line argparse and initiate logging settings.
"""

from .main import run, commandline_parser
import os
import logging
import logging.handlers

parser = commandline_parser()

## Extract arguments
quiet = False
args = parser.parse_args()

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
logging.info("JIT compilation: {}".format(args.jit))
logging.info("Use Cython: {}".format(args.cython))

# Threads
# if args.threads != None:
#     threadcount = args.threads
# else:
#     threadcount = 1

# Run program
if args.input:
    kwargs = {"filename":args.input}
else:
    kwargs = {}
#kwargs["logFile"] = args.logFile
kwargs["threads"] = args.threads
kwargs["path"] = args.path
kwargs["jit" ] = args.jit
kwargs["cython" ] = args.cython

run(**kwargs)
