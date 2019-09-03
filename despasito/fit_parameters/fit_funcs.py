
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

from despasito.thermodynamics import calc

class BasinStep(object):
    """
    Attributes:
    -----------

    stepmag : list
        List of step magnitudes
    stepsize : float
        default 0.05
    """
    def __init__(self, stepmag, stepsize=0.05):
        self._stepsize = stepsize
        self._stepmag = stepmag

    def __call__(self, x):

        # Save intital guess in array
        xold = np.copy(x)

        # Loop for number of times to start over
        for j in range(1000):
            # reset array x
            x = np.copy(xold)
            breakloop = True
            # Iterate through array of step magnitudes
            for i, mag in enumerate(self._stepmag):
                # Add or subtract a random number within distribution of +- mag*stepsize
                x[i] += np.random.uniform(-mag * self._stepsize, mag * self._stepsize)
                # If a value of x is negative break cycle
                if x[i] < 0.0:
                    breakloop = False
            if breakloop: break
            print(x, j)
        return x


def print_fun(x, f, accepted):
    print('test')
    print(("at minimum %.4f accepted %d" % (f, int(accepted))))


def calc_phase_wrapper(calctype,T,xi,eos,rhodict={}):
    if calctype == "sat_props":
        try:
            Psat, rholsat, rhovsat = calc.calc_Psat(T, xi, eos, rhodict=rhodict)
        except:
            Psat = 100000.0
            rholsat = 100000.0
            rhovsat = 100000.0
        return Psat, rholsat, rhovsat

    elif calctype == "liquid_properties":
        try:
            rhol, flagl = calc.calc_rhol(101325.0, T, xi, eos, rhodict=rhodict)
        except:
            rhol = 100000.0
        print(rhol)
        return 100000.0, rhol, 100000.0

    elif calctype == "phase_xiT":
        try:
            P, yi = calc.calc_xT_phase(xi, T, eos, rhodict=rhodict)
        except:
            P = 10000000.0
            yi = np.ones_like(xi) * 100.0
        return P, yi

def compute_SAFT_obj(beadparams,
                     fit_bead,
                     fit_params,
                     molecule_params,
                     beadlibrary,
                     bounds,
                     crosslibrary={},
                     threads=1):

    for i, boundval in enumerate(bounds):
        if (beadparams[i] > boundval[1]) or (beadparams[i] < boundval[0]):
            return 10000.0 # for sat, liquid, vapor properties
#            return 100000000.0 # for calc_phase_xiT

    #update beadlibrary with test paramters
    for i, param in enumerate(fit_params):
        if param.startswith('epsilon_'):
            if fit_bead in list(crosslibrary[param.split('_')[1]].keys()):
                crosslibrary[param.split('_')[1]][fit_bead].update({'epsilon': beadparams[i]})
            else:
                crosslibrary[param.split('_')[1]][fit_bead] = {'epsilon': beadparams[i]}
        elif param.startswith('epsilon') and param != 'epsilon':
            #print param.split('_')[0],beadparams[i]
            #print param.split('_')[1]
            if fit_bead in list(crosslibrary[param.split('_')[1]].keys()):
                crosslibrary[param.split('_')[1]][fit_bead].update({param.split('_')[0]: beadparams[i]})
            else:
                crosslibrary[param.split('_')[1]][fit_bead] = {param.split('_')[0]: beadparams[i]}
        elif param.startswith('K'):
            if fit_bead in list(crosslibrary[param.split('_')[1]].keys()):
                crosslibrary[param.split('_')[1]][fit_bead].update({param.split('_')[0]: beadparams[i]})
            else:
                crosslibrary[param.split('_')[1]][fit_bead] = {param.split('_')[0]: beadparams[i]}
        elif param.startswith('l_r') and param != 'l_r':
            if fit_bead in list(crosslibrary[param.split('_')[2]].keys()):
                crosslibrary[param.split('_')[2]][fit_bead].update({'l_r': beadparams[i]})
            else:
                crosslibrary[param.split('_')[2]][fit_bead] = {'l_r': beadparams[i]}
        else:
            beadlibrary[fit_bead][param] = beadparams[i]

    #generate input list
    input_list = []
    for molecule in molecule_params:
        # calctype Saturation Properties
        for T in molecule_params[molecule]["ePsat"][0]:
            rhodict = {"minrhofrac":(1.0 / 80000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
            input_list.append((T, molecule_params[molecule]["xi"], eos, rhodict, "sat_props"))
        # calctype liquid_properties
        for T in molecule_params[molecule]["erhol"][0]:
            rhodict = {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
            input_list.append((T, molecule_params[molecule]["xi"], eos, rhodict, "liquid_properties"))
        # calctype phase_xiT
        for i, T in enumerate(molecule_params[molecule]["TLVE"][0]):
            rhodict = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
            input_list.append((np.array([molecule_params[molecule]["TLVE"][3][i],
                           molecule_params[molecule]["TLVE"][4][i]]), T, eos, rhodict, "phase_xiT"))

    #Tlist_rhol=exp_rhol[0]
    #Tlist_Psat=exp_Psat[0]
    #Tlist=np.append(Tlist_rhol,Tlist_Psat)
    #Psat=np.zeros_like(Tlist_Psat)
    #rholsat=np.zeros_like(Tlist_rhol)

    if __name__ == '__main__':
        p = Pool(threads)
        #input_list = [(T,xi,massi,nui,beads,beadlibrary,(1.0/30000.0),10.0,1.0E-4,1000.0*101325.0) for T in Tlist]
        phase_list = p.map(calc_phase_wrapper, input_list, 1)
        p.close()
        p.join()

    # Arrange data
    type_phase_list = [type(x) for x in phase_list[0]]
    if (type_phase_list not in [list,np.ndarray]).all():
        phase_list = np.array(phase_list).T
    else:
        len_phase_list = []
        for i,typ in enumerate(type_phase_list):
            if typ in [list,np.ndarray]:
                len_phase_list.append(len(phase_list[i]))
            else:
                len_phase_list.append(1)
        phase_array = np.zeros([len(phase_list), sum(len_phase_list)])
      
        for i, val in enumerate(phase_list):
            ind = 0
            for j,l in enumerate(len_phase_list):
                if l == 1
                    phase_array[i, ind] = val[j] 
                else:
                    phase_array[i, ind:ind+l+1] = val[j]
                ind += l
        phase_list = np.array(phase_array).T
                
    # Compute obj_function
    obj_function = 0.0
    index = 0

    # Dataset_array
    dataset_array = ["ePsat", "erhol"]
    dataset_array = ["TLVE"]

    #loop over each molecule with data
    for molecule in molecule_params:

        #compare for Psat and rhol i=0 for Psat and i=1 for rhol
        for i, dataset in enumerate(dataset_array):
            if dataset in ["ePsat", "erhol"]:
                if dataset == "ePsat": ind = 0
                elif dataset == "erhol": ind = 1
                obj_function += np.sum(((phase_list[ind, index:(index+np.size(molecule_params[molecule][dataset][1]))]-molecule_params[molecule][dataset][1]) / molecule_params[molecule][dataset][1])**2)
            elif dataset == "TLVE":
                obj_function += np.sum(((phase_list[0][index:(index+np.size(molecule_params[molecule][dataset][5]))] - molecule_params[molecule][dataset][5]) / molecule_params[molecule][dataset][5])**2)
                # This fitting assumes a binary fluid, so only the composition of the first component is used in the objective function
                obj_function += np.sum(((phase_list[1][index:(index+np.size(molecule_params[molecule][dataset][5]))] - molecule_params[molecule][dataset][1]))**2)

            index += np.size(molecule_params[molecule][dataset][1])

    print(phase_list)
    print(beadparams)
    print(obj_function)
    #for i in range(np.size(Psat)):
    #    print exp_P_sat[0][i],Psat[i],exp_P_sat[1][i],rholsat[i],exp_rho_liq[1][i]
    return obj_function

