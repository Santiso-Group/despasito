import numpy as np
import calchelmholtz
import readwrite_input
import constants
import calc_phase
import time
import json
import timeit
import copy
from scipy import integrate
#import cmath
from scipy.misc import derivative
import scipy.optimize as spo
from scipy import interpolate
from scipy.optimize import minimize_scalar
#import Achain
#import density_vector_test
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threads", type=int, help="set the number of theads used")
args = parser.parse_args()

if args.threads != None:
    threadcount = args.threads
else:
    threadcount = 1

with open('CH2OH-CO2_fit/SAFTgroup.json', 'r') as f:
    output = f.read()
beadlibrary = json.loads(output)

with open('CH2OH-CO2_fit/SAFTcross.json', 'r') as f:
    output = f.read()
crosslibrary = json.loads(output)

#crosslibrary['CH3'].pop('CH2',None)
#crosslibrary['CH2'].pop('CH3',None)

#print crosslibrary.keys()
#if 'CH3' in keys.crosslibrary:
#    if 'CH2OCH2' in keys.crosslibrary['CH3']

#    crosslibrary['CH3']['CH2OCH2'] = {'epsilon':500.0}

input_file = open('CH2OH-CO2_fit/input.json', 'r').read()
opt_input = json.loads(input_file)
opt_params = opt_input[0]
molecule_params = opt_input[1]

nmoleculetypes = np.size(list(molecule_params.keys()))

#read experimental data
for molecule in list(molecule_params.keys()):
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
            massi = np.zeros_like(xi)
            for i in range(np.size(xi)):
                for k in range(np.size(beads)):
                    massi[i] += nui[i, k] * beadlibrary[beads[k]]["mass"]
            molecule_params[molecule]['massi'] = massi

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
            bounds.append(tuple(opt_params[param + '_bounds']))
        elif param.startswith('epsilon') and param != 'epsilon':
            bounds.append(tuple(opt_params[param + '_bounds']))
            beadparams0[i] = 1000.0
        elif param.startswith('K'):
            bounds.append(tuple(opt_params[param + '_bounds']))
            beadparams0[i] = 100.0e-30
        elif param.startswith('l_r'):
            bounds.append(tuple(opt_params[param + '_bounds']))
            bead_a = param.split('_')[2]
            bead_b = opt_params["fit_bead"]
            lra = beadlibrary[bead_a]["l_r"]
            lrb = beadlibrary[bead_b]["l_r"]
            beadparams0[i] = np.sqrt((lra - 3.0) * (lrb - 3.0)) + 3.0

print(beadparams0)
beadparams0 = np.array([200.0, 9.5077])
#t1=time.time()
#test=compute_SAFT_obj(beadparams0,opt_params["fit_bead"],opt_params["fit_params"],molecule_params,beadlibrary,crosslibrary=crosslibrary,threads=threadcount)
#t2=time.time()
#print t2-t1
#res = spo.minimize(compute_SAFT_obj, beadparams0, args=(opt_params["fit_bead"],opt_params["fit_params"],molecule_params,beadlibrary,threadcount), method='nelder-mead',options={'maxiter': 1000})
#print res
#res = spo.differential_evolution(compute_SAFT_obj, bounds, args=(opt_params["fit_bead"],opt_params["fit_params"],molecule_params,beadlibrary,crosslibrary,threadcount), polish=False,disp=True)

custombasinstep = BasinStep(np.array([220.86002155, 11.15625049]), stepsize=0.1)

res = spo.basinhopping(compute_SAFT_obj,
                       beadparams0,
                       niter=500,
                       T=0.2,
                       stepsize=0.1,
                       minimizer_kwargs={
                           "args": (opt_params["fit_bead"], opt_params["fit_params"], molecule_params, beadlibrary,
                                    bounds, crosslibrary, threadcount),
                           "method":
                           'nelder-mead',
                           "options": {
                               'maxiter': 50
                           }
                       },
                       take_step=custombasinstep,
                       disp=True)
