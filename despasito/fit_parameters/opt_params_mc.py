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


class BasinStep(object):
    def __init__(self, stepmag, stepsize=0.05):
        self.stepsize = stepsize
        self.stepmag = stepmag

    def __call__(self, x):
        s = self.stepsize
        smag = self.stepmag
        xold = np.copy(x)
        for j in range(1000):
            x = np.copy(xold)
            breakloop = True
            for i, mag in enumerate(smag):
                x[i] += np.random.uniform(-mag * s, mag * s)
                if x[i] < 0.0:
                    breakloop = False
            if breakloop: break
            print(x, j)
        return x


def print_fun(x, f, accepted):
    print('test')
    print(("at minimum %.4f accepted %d" % (f, int(accepted))))


def calc_phase_wrapper(xxx_todo_changeme):
    (xi, T, massi, nui, beads, beadlibrary, crosslibrary, minrhofracval, rhoincval, vspacemaxval, Pmaxval,
     calctype) = xxx_todo_changeme
    if calctype == 0:
        try:
            #print yi,T,massi,nui,beads,beadlibrary,["H","e1"],crosslibrary,minrhofracval,rhoincval,vspacemaxval,Pmaxval,calctype
            #P,yi=calc_phase.Calc_xT_phase(xi,T,massi,nui,beads,beadlibrary,["H","e1","a1"],crosslibrary,minrhofracval,rhoincval,vspacemaxval,Pmaxval,0.65,Pguess)
            P, yi = calc_phase.Calc_xT_phase(xi, T, massi, nui, beads, beadlibrary, ["H", "e1"], crosslibrary,
                                             minrhofracval, rhoincval, vspacemaxval, Pmaxval, 0.65)
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
            return 100000000.0

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
        for i, T in enumerate(molecule_params[molecule]["TLVE"][0]):
            input_list.append(
                (np.array([molecule_params[molecule]["TLVE"][3][i],
                           molecule_params[molecule]["TLVE"][4][i]]), T, molecule_params[molecule]["massi"],
                 molecule_params[molecule]["nui"], molecule_params[molecule]["beads"], beadlibrary, crosslibrary,
                 (1.0 / 300000.0), 10.0, 1.0E-4, 1000.0 * 101325.0, 0))

            #input_list.append((np.array([molecule_params[molecule]["TLVE"][1][i],molecule_params[molecule]["TLVE"][2][i],molecule_params[molecule]["TLVE"][3][i]]),T,molecule_params[molecule]["massi"],molecule_params[molecule]["nui"],molecule_params[molecule]["beads"],beadlibrary,crosslibrary,(1.0/300000.0),10.0,1.0E-4,1000.0*101325.0,molecule_params[molecule]["TLVE"][4][i]*2.0,0))
        #for T in molecule_params[molecule]["ePvap"][0]:
        #input_list.append((T,molecule_params[molecule]["xi"],molecule_params[molecule]["massi"],molecule_params[molecule]["nui"],molecule_params[molecule]["beads"],beadlibrary,crosslibrary,(1.0/80000.0),10.0,1.0E-4,1000.0*101325.0,0))
        #for T in molecule_params[molecule]["erhol"][0]:
        #input_list.append((T,molecule_params[molecule]["xi"],molecule_params[molecule]["massi"],molecule_params[molecule]["nui"],molecule_params[molecule]["beads"],beadlibrary,crosslibrary,(1.0/60000.0),10.0,1.0E-4,1000.0*101325.0,1))

        #calc_phase.Calc_yT_phase(np.array([0.5328,0.4672]),363.3,massi,nui,beads,beadlibrary,["H","e1"],crosslibrary,1/60000.0,10.0,1.0E-4,1000.0*101325.0,0.65)

    #Tlist_rhol=exp_rhol[0]
    #Tlist_Pvap=exp_Pvap[0]
    #Tlist=np.append(Tlist_rhol,Tlist_Pvap)
    #Pvap=np.zeros_like(Tlist_Pvap)
    #rholsat=np.zeros_like(Tlist_rhol)

    if __name__ == '__main__':
        p = Pool(threads)
        #input_list = [(T,xi,massi,nui,beads,beadlibrary,(1.0/30000.0),10.0,1.0E-4,1000.0*101325.0) for T in Tlist]
        phase_list = p.map(calc_phase_wrapper, input_list, 1)
        p.close()
        p.join()

        #phase_array=np.zeros([)
        #for i in phase_list:
    phase_array = np.zeros([len(phase_list), 1 + np.size(phase_list[0][1])])
    for i, val in enumerate(phase_list):
        phase_array[i, 0] = val[0]
        phase_array[i, 1:] = val[1]

    phase_list = np.array(phase_array).T
    #print phase_list[1][0:2]
    #print phase_list.ravel()
    #compute obj_function
    obj_function = 0.0
    index = 0
    dataset = "TLVE"
    #loop over each molecule with data
    for molecule in molecule_params:
        print(phase_list[0][index:index + np.size(molecule_params[molecule][dataset][5])],
              molecule_params[molecule][dataset][5])
        print(phase_list[1][index:index + np.size(molecule_params[molecule][dataset][5])],
              molecule_params[molecule][dataset][1])
        obj_function += np.sum(((phase_list[0][index:index + np.size(molecule_params[molecule][dataset][5])] -
                                 molecule_params[molecule][dataset][5]) / molecule_params[molecule][dataset][5])**2)
        obj_function += np.sum(((phase_list[1][index:index + np.size(molecule_params[molecule][dataset][5])] -
                                 molecule_params[molecule][dataset][1]))**2)
        index += np.size(molecule_params[molecule][dataset][1])

        #obj_function+=np.sum(np.abs(((phase_list[0][index:index+np.size(molecule_params[molecule][dataset])]*phase_list[1][index:index+np.size(molecule_params[molecule][dataset])])-molecule_params[molecule][dataset][4])))
        #print molecule_params[molecule][dataset][5]
        #print phase_list[1:][index:index+np.size(molecule_params[molecule][dataset])]
        #print molecule_params[molecule][dataset][3:5]

        #print molecule_params[molecule][dataset][1]

        #obj_function+=np.sum(((phase_list[i,index:index+np.size(molecule_params[molecule][dataset][1])] - molecule_params[molecule][dataset][1])/molecule_params[molecule][dataset][1])**2)

        #for z in phase_list[i,index:index+np.size(molecule_params[molecule][dataset][1])]:
        #    print z
        #print molecule_params[molecule][dataset][1]

        #index+=np.size(molecule_params[molecule][dataset][1])

    #rholsat=phase_list[1,0:np.size(Tlist_rhol)]
    #Pvap=phase_list[0,np.size(Tlist_rhol):]

    #obj_function=np.sum(((Pvap-exp_Pvap[1])/exp_Pvap[1])**2)+np.sum(((rholsat-exp_rhol[1])/exp_rhol[1])**2)

    print(phase_list)
    print(beadparams)
    print(obj_function)
    #for i in range(np.size(Psat)):
    #    print exp_P_sat[0][i],Psat[i],exp_P_sat[1][i],rholsat[i],exp_rho_liq[1][i]
    return obj_function


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
