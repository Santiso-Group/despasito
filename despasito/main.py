"""
Handles the primary functions

In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

"""

from .input_output import readwrite_input
from .equations_of_state import eos as eos_mod
from .thermodynamics import thermo
from .fit_parameters import fit

def run(*args):
	""" Main function for running despasito calculations. All inputs and settings should be in the supplied JSON file(s).
	"""
	# Settings that should be replaced
	meth = "brent"

	#read input file (need to add command line specification)
	eos_dict, thermo_dict = readwrite_input.process_commandline(*args)

	try:
		eos = eos_mod("saft.gamma_mie", **eos_dict)
	except:
		raise

	# Run either parameterization or thermodynamic calculation
	if "opt_params" in list(thermo_dict.keys()):
    		output = fit(eos, thermo_dict)
    		print(output)
	else:
    		output_dict = thermo(eos, thermo_dict)
		# readwrite_input.writeout_dict(output_dict)
