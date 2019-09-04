import numpy as np
import time
import json
import timeit
import copy
from scipy import integrate
from scipy.misc import derivative
import scipy.optimize as spo
from scipy import interpolate
from scipy.optimize import minimize_scalar
from multiprocessing import Pool
import argparse

######## Will eventually move to main ##
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threads", type=int, help="set the number of theads used")
args = parser.parse_args()

if args.threads != None:
    threadcount = args.threads
else:
    threadcount = 1
########

####### Substitute with input_output ##
with open('NCH2-2CH3_bead_fit/SAFTgroup.json', 'r') as f:
    output = f.read()
beadlibrary = json.loads(output)

with open('NCH2-2CH3_bead_fit/SAFTcross.json', 'r') as f:
    output = f.read()
crosslibrary = json.loads(output)

#crosslibrary['CH3'].pop('CH2',None)
#crosslibrary['CH2'].pop('CH3',None)

#print crosslibrary.keys()
#if 'CH3' in keys.crosslibrary:
#    if 'CH2OCH2' in keys.crosslibrary['CH3']

#    crosslibrary['CH3']['CH2OCH2'] = {'epsilon':500.0}

input_file = open('NCH2-2CH3_bead_fit/input.json', 'r').read()
opt_input = json.loads(input_file)

opt_params = opt_input[0]
molecule_params = opt_input[1]
### return opt_params molecule_params

### Process parameter fit input

# Read experimental data
# Molecule is the parameter being fit
for molecule in list(molecule_params.keys()):
    # data is the beadconfig, and the source of exp. data
    for data in list(molecule_params[molecule].keys()):
        if data != "beadconfig":
            molecule_params[molecule][data] = np.loadtxt(molecule_params[molecule][data], delimiter=',').T
            #convert from kmol/m3 to mol
            if data == 'erhol':
                molecule_params[molecule][data][1] *= 1000.0
        else:
            #read bead config for molecule
            xi, beads, nui = readwrite_input.process_bead_data(molecule_params[molecule]['beadconfig'])
            molecule_params[molecule]['xi'] = xi
            molecule_params[molecule]['beads'] = beads
            molecule_params[molecule]['nui'] = nui
    # Note make eos object here, remove nned for next 5 lines, need sitenames?
            massi = np.zeros_like(xi)
            for i in range(np.size(xi)):
                for k in range(np.size(beads)):
                    massi[i] += nui[i, k] * beadlibrary[beads[k]]["mass"]
            molecule_params[molecule]['massi'] = massi
##### Here molecule_params will have an eos object and exp. data

beadparams0 = np.zeros(np.size(opt_params["fit_params"]))
bounds = []

for i, param in enumerate(opt_params["fit_params"]):
    try:
        beadparams0[i] = beadlibrary[opt_params["fit_bead"]][param]
        bounds.append(tuple(opt_params[param + '_bounds']))
    except KeyError:
        if param.startswith('epsilon_'):

            #compute mixing rule as initial guess for cross terms.
            bead_a = param.split('_')[1]
            bead_b = opt_params["fit_bead"]
            sigmaa = beadlibrary[bead_a]["sigma"]
            sigmab = beadlibrary[bead_b]["sigma"]

            sigmaab = (sigmaa + sigmab) / 2.0
            epsilona = beadlibrary[bead_a]["epsilon"]
            epsilonb = beadlibrary[bead_b]["epsilon"]
            beadparams0[i] = np.sqrt(epsilona * epsilonb) * np.sqrt((sigmaa**3) * (sigmab**3)) / (sigmaab**3)

            bounds.append(tuple(opt_params['epsilon_bounds']))
        elif param.startswith('epsilon') and param != 'epsilon':
            bounds.append(tuple(opt_params[param + '_bounds']))
            beadparams0[i] = 1000.0
        elif param.startswith('K'):
            bounds.append(tuple(opt_params[param + '_bounds']))
            beadparams0[i] = 100.0e-30

##### Here bounds and beadparams0 (initial guess) is defined from opt_params and beadlibrary
   # Change opt_params {fit_bead, fit params, beadparams0, bounds}

######### End input/output

####### Fit parameters: need opt_params{ now with beadparams0, bounds}, molecule params
custombasinstep = BasinStep(np.array([550.0, 26.0, 4.0e-10, 0.45, 500.0, 150.0e-30, 550.0]), stepsize=0.1)

res = spo.basinhopping(compute_SAFT_obj,
                       beadparams0,
                       niter=500,
                       T=0.2,
                       stepsize=0.1,
                       minimizer_kwargs={
                           "args": (opt_params["fit_bead"], opt_params["fit_params"], molecule_params, beadlibrary, bounds, crosslibrary, threadcount),"method": 'nelder-mead', "options": {'maxiter': 200}},
                       take_step=custombasinstep,
                       disp=True)
