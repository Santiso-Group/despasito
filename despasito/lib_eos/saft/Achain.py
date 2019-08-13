"""
    despasito
    DESPASITO: Determining Equilibrium State and Parameters Applied to SAFT, Intended for Thermodynamic Output
    
    This file contains specific calculations to compute the chain contribution for the SAFT-Gamma-Mie EOS.
    
"""

import numpy as np
from scipy import misc
from . import constants
from . import gamma_mie_funcs as funcs


def calc_a1sii(rhos_var, epsilonii, dii3, l_ii, etax):
    """
    """
    #initialize values
    #print l_ii
    etaii = np.zeros_like(epsilonii)
    cii = np.zeros(4)
    c_coef = np.array([[0.81096, 1.7888, -37.578, 92.284], [1.0205, -19.341, 151.26, -463.50],
                       [-1.9057, 22.845, -228.14, 973.92], [1.0885, -6.1962, 106.98, -677.64]])
    l_iipow = np.array([1.0, l_ii**-1.0, l_ii**-2.0, l_ii**-3.0])

    cii = np.inner(c_coef, l_iipow.T)
    etaii = (cii[0] * etax) + (cii[1] * etax**2.0) + (cii[2] * etax**3.0) + (cii[3] * etax**4.0)

    # a1sii = -2.0 * np.pi * rhos * ((epsilonii*dii3)/(l_ii-3.0)) * (1.0 - (etaii/2.0)) / ((1.0 - etaii)**3)

    return -2.0 * np.pi * rhos_var * ((epsilonii * dii3) / (l_ii - 3.0)) * (1.0 - (etaii / 2.0)) / ((1.0 - etaii)**3)


def calc_Bii(rhos_var, epsilonii, l_ii, dii3, etax, x0ii):
    """
    """

    I_ii = (1.0 - x0ii**(3.0 - l_ii)) / (l_ii - 3.0)
    J_ii = (1.0 - (x0ii**(4.0 - l_ii)) * (l_ii - 3.0) + (x0ii**(3.0 - l_ii)) * (l_ii - 4.0)) / ((l_ii - 3.0) *
                                                                                                (l_ii - 4.0))

    Bii = (2.0 * np.pi * rhos_var * dii3 * epsilonii) * (((1.0 - (etax / 2.0)) / ((1.0 - etax)**3) * I_ii) -
                                                         ((9.0 * etax * (1.0 + etax)) / (2.0 *
                                                                                         (1.0 - etax)**3)) * J_ii)

    return Bii


def calc_a1ii(rhos_var, epsilonii, l_rii, l_aii, dii3, dkl, xskl, x0ii):
    """
    """
    etax = (rhos_var * np.pi / 6.0) * np.sum(xskl * (dkl**3))
    Cii = C(l_rii, l_aii)
    a1sii_laii = funcs.calc_a1s(rhos_var, 1.0, l_aii, etax, epsilonii, dii3**(1.0 / 3.0))
    a1sii_lrii = funcs.calc_a1s(rhos_var, 1.0, l_rii, etax, epsilonii, dii3**(1.0 / 3.0))
    #a1sii_laii = calc_a1sii(rhos_var,epsilonii,dii3,l_aii,etax)
    #a1sii_lrii = calc_a1sii(rhos_var,epsilonii,dii3,l_rii,etax)

    Bii_laii = funcs.calc_Bkl(rhos_var, l_aii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)
    Bii_lrii = funcs.calc_Bkl(rhos_var, l_rii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)

    #Bii_laii = calc_Bii(rhos_var,epsilonii,l_aii,dii3,etax,x0ii)
    #Bii_lrii = calc_Bii(rhos_var,epsilonii,l_rii,dii3,etax,x0ii)
    #test1=((x0ii**l_rii)*(a1sii_lrii + Bii_lrii))

    a1ii = Cii * (((x0ii**l_aii) * (a1sii_laii + Bii_laii)) - ((x0ii**l_rii) * (a1sii_lrii + Bii_lrii)))
    #print rhos,a1ii
    return a1ii


def C(l_r, l_a):
    """ Returns C, the Mie potential prefactor, defined in eq. 2
        Inputs:
            l_r: repulsive exponent of the Mie potential
            l_a: attractive exponent of the Mie potential
    """

    return (l_r / (l_r - l_a)) * (l_r / l_a)**(l_a / (l_r - l_a))


def calc_a2ii_1pchi(rhos_var, epsilonii, dii3, dkl, xskl, x0ii, l_rii, l_aii):

    etax = (rhos_var * np.pi / 6.0) * np.sum(xskl * (dkl**3))
    KHS = ((1.0 - etax)**4) / (1.0 + (4.0 * etax) + (4.0 * (etax**2)) - (4.0 * (etax**3)) + (etax**4))
    Cii = C(l_rii, l_aii)

    a1sii_2l_aii = funcs.calc_a1s(rhos_var, 1.0, 2.0 * l_aii, etax, epsilonii, dii3**(1.0 / 3.0))
    a1sii_2l_rii = funcs.calc_a1s(rhos_var, 1.0, 2.0 * l_rii, etax, epsilonii, dii3**(1.0 / 3.0))
    a1sii_l_riil_aii = funcs.calc_a1s(rhos_var, 1.0, l_aii + l_rii, etax, epsilonii, dii3**(1.0 / 3.0))

    #a1sii_2l_aii=calc_a1sii(rhos_var,epsilonii,dii3,2.0*l_aii,etax)
    #a1sii_2l_rii=calc_a1sii(rhos_var,epsilonii,dii3,2.0*l_rii,etax)
    #a1sii_l_riil_aii=calc_a1sii(rhos_var,epsilonii,dii3,l_aii+l_rii,etax)

    Bii_2l_aii = funcs.calc_Bkl(rhos_var, 2.0 * l_aii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)
    Bii_2l_rii = funcs.calc_Bkl(rhos_var, 2.0 * l_rii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)
    Bii_l_aiil_rii = funcs.calc_Bkl(rhos_var, l_aii + l_rii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)

    #Bii_2l_aii=calc_Bii(rhos_var,epsilonii,2.0*l_aii,dii3,etax,x0ii)
    #Bii_2l_rii=calc_Bii(rhos_var,epsilonii,2.0*l_rii,dii3,etax,x0ii)
    #Bii_l_aiil_rii=calc_Bii(rhos_var,epsilonii,l_aii+l_rii,dii3,etax,x0ii)

    #print 'a1sii_2l_rii',a1sii_2l_rii,calc_a1sii(rhos,epsilonii,dii3,l_rii,etax)
    #print 'l',l_rii,2.0*l_rii
    #test = 0.5*KHS*epsilonii*(Cii**2)*( (x0ii**(2.0*l_aii))*(a1sii_2l_aii+Bii_2l_aii) - (2.0*(x0ii**(l_aii+l_rii))) * (a1sii_l_riil_aii + Bii_l_aiil_rii) + (x0ii**(2.0*l_rii)) * (a1sii_2l_rii+Bii_2l_rii) )
    #print 'rhos',rhos_var,test
    a2ii_1pchi = 0.5 * epsilonii * (Cii**2) * ((x0ii**(2.0 * l_aii)) * (a1sii_2l_aii + Bii_2l_aii) -
                                               (2.0 * (x0ii**(l_aii + l_rii))) * (a1sii_l_riil_aii + Bii_l_aiil_rii) +
                                               (x0ii**(2.0 * l_rii)) * (a1sii_2l_rii + Bii_2l_rii))

    a2ii_1pchi = np.einsum("i,ij->ij", KHS, a2ii_1pchi)
    return a2ii_1pchi

    #return 0.5*KHS*epsilonii*(Cii**2)*( (x0ii**(2.0*l_aii))*(a1sii_2l_aii+Bii_2l_aii) - (2.0*(x0ii**(l_aii+l_rii))) * (a1sii_l_riil_aii + Bii_l_aiil_rii) + (x0ii**(2.0*l_rii)) * (a1sii_2l_rii+Bii_2l_rii) )

    # a1s_2la=calc_a1s(rho,Amonopre,2.0*l_akl,etax,epsilonkl,dkl)
    #a1s_2lr=calc_a1s(rho,Amonopre,2.0*l_rkl,etax,epsilonkl,dkl)
    #a1s_lalr=calc_a1s(rho,Amonopre,l_akl+l_rkl,etax,epsilonkl,dkl)
    #B_2la=calc_Bkl(rho,2.0*l_akl,Amonopre,dkl,epsilonkl,x0kl,etax)
    #B_2lr=calc_Bkl(rho,2.0*l_rkl,Amonopre,dkl,epsilonkl,x0kl,etax)
    #B_lalr=calc_Bkl(rho,l_akl+l_rkl,Amonopre,dkl,epsilonkl,x0kl,etax)

    #a2kl=(x0kl**(2.0*l_akl))*(a1s_2la+B_2la)-((2.0*x0kl**(l_akl+l_rkl))*(a1s_lalr+B_lalr))+((x0kl**(2.0*l_rkl))*(a1s_2lr+B_2lr))
    #a2kl*=(1.0+chikl)*epsilonkl*(Ckl**2)*(KHS/2.0)


def calc_da1iidrhos(rhos, epsilonii, l_rii, l_aii, dii3, dkl, xskl, x0ii, stepmult=1.0):
    """
    computer derivative
    """

    step = np.sqrt(np.finfo(float).eps) * rhos * stepmult
    a1ii_p = calc_a1ii(rhos + step, epsilonii, l_rii, l_aii, dii3, dkl, xskl, x0ii)
    a1ii_m = calc_a1ii(rhos - step, epsilonii, l_rii, l_aii, dii3, dkl, xskl, x0ii)

    return np.einsum("ij,i->ij", (a1ii_p - a1ii_m), 0.5 / step)


def calc_da2ii_1pchi_drhos(rhos, epsilonii, dii3, dkl, xskl, x0ii, l_rii, l_aii, stepmult=1.0):
    """
    computer derivative
    """

    step = np.sqrt(np.finfo(float).eps) * rhos * stepmult
    a2ii_1pchi_p = calc_a2ii_1pchi(rhos + step, epsilonii, dii3, dkl, xskl, x0ii, l_rii, l_aii)
    a2ii_1pchi_m = calc_a2ii_1pchi(rhos - step, epsilonii, dii3, dkl, xskl, x0ii, l_rii, l_aii)

    return np.einsum("ij,i->ij", (a2ii_1pchi_p - a2ii_1pchi_m), 0.5 / step)


def calc_Achain(rho, Amonopre, xi, nui, etax, sigmakl, epsilonkl, dkl, xskl, l_rkl, l_akl, beads, beadlibrary, T,
                etaxstar, KHS):
    """
    nui: (number of components,number of group types) numpy array, list of group quantites in each component 
                    defined for eq. 11. Note that indicies are flipped from definition in reference.
    etax:
    sigmakl: Mie segment diameter in groups k and l
    dkl: Hard sphere radius for segments in groups k and l
    epsilonkl
    l_rkl: Mie potential repulsive exponent for segment in group k and l
    l_akl Mie potential attractive exponent for segment in group k and l

    """
    #initialize values
    ngroups = len(beads)
    ncomp = np.size(xi)
    zki = np.zeros((ncomp, ngroups), float)
    zkinorm = np.zeros(ncomp, float)
    sigmaii3 = np.zeros(ncomp, float)
    dii3 = np.zeros_like(sigmaii3)
    epsilonii = np.zeros_like(sigmaii3)
    l_rii = np.zeros_like(sigmaii3)
    l_aii = np.zeros_like(sigmaii3)
    x0ii = np.zeros_like(sigmaii3)
    km = np.zeros((np.size(rho), 4))
    gdHS = np.zeros((np.size(rho), ncomp))

    rhos = rho * Amonopre
    kT = T * constants.kb

    stepmult = 100

    #compute zki
    for i in range(ncomp):
        for k in range(ngroups):
            zki[i, k] = nui[i, k] * beadlibrary[beads[k]]["Vks"] * beadlibrary[beads[k]]["Sk"]
            zkinorm[i] += zki[i, k]

    for i in range(ncomp):
        for k in range(ngroups):
            zki[i, k] = zki[i, k] / zkinorm[i]

    # compute average molecular segment size ^ 3: sigmaii3
    # compute effective hard sphere diameter ^ 3: dii3
    # compute average interaction energy epsilonii
    #compute average repulsive and attractive exponenets l_rkl, l_akl
    for i in range(ncomp):
        for k in range(ngroups):
            for l in range(ngroups):
                sigmaii3[i] += zki[i, k] * zki[i, l] * sigmakl[k, l]**3
                dii3[i] += zki[i, k] * zki[i, l] * dkl[k, l]**3
                epsilonii[i] += zki[i, k] * zki[i, l] * epsilonkl[k, l] * constants.kb
                l_rii[i] += zki[i, k] * zki[i, l] * l_rkl[k, l]
                l_aii[i] += zki[i, k] * zki[i, l] * l_akl[k, l]

    #compute x0ii
    x0ii = (sigmaii3**(1.0 / 3.0)) / (dii3**(1.0 / 3.0))

    km[:, 0] = -np.log(1.0 - etax) + (42.0 * etax - 39.0 * etax**2 + 9.0 * etax**3 - 2.0 * etax**4) / (6.0 *
                                                                                                       (1.0 - etax)**3)
    km[:, 1] = (etax**4 + 6.0 * etax**2 - 12.0 * etax) / (2.0 * (1.0 - etax)**3)
    km[:, 2] = -3.0 * etax**2 / (8.0 * (1.0 - etax)**2)
    km[:, 3] = (-etax**4 + 3.0 * etax**2 + 3.0 * etax) / (6.0 * (1.0 - etax)**3)

    for i in range(ncomp):
        gdHS[:, i] = np.exp(km[:, 0] + km[:, 1] * x0ii[i] + km[:, 2] * x0ii[i]**2 + km[:, 3] * x0ii[i]**3)

    da1iidrhos = calc_da1iidrhos(rhos, epsilonii, l_rii, l_aii, dii3, dkl, xskl, x0ii, stepmult=stepmult)

    a1sii_l_aii = funcs.calc_a1s(rhos, 1.0, l_aii, etax, epsilonii, dii3**(1.0 / 3.0))
    a1sii_l_rii = funcs.calc_a1s(rhos, 1.0, l_rii, etax, epsilonii, dii3**(1.0 / 3.0))

    Bii_l_aii = funcs.calc_Bkl(rhos, l_aii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)
    Bii_l_rii = funcs.calc_Bkl(rhos, l_rii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)

    Cii = C(l_rii, l_aii)

    g1 = (1.0 /
          (2.0 * np.pi * epsilonii * dii3)) * (3.0 * da1iidrhos - Cii * l_aii *
                                               (x0ii**l_aii) * np.einsum("ij,i->ij",
                                                                         (a1sii_l_aii + Bii_l_aii), 1.0 / rhos) +
                                               (Cii * l_rii *
                                                (x0ii**l_rii)) * np.einsum("ij,i->ij",
                                                                           (a1sii_l_rii + Bii_l_rii), 1.0 / rhos))

    #compute g2

    phi7 = np.array([10.0, 10.0, 0.57, -6.7, -8.0])
    alphaii = Cii * ((1.0 / (l_aii - 3.0)) - (1.0 / (l_rii - 3.0)))
    theta = np.exp(epsilonii / kT) - 1.0

    gammacii = np.zeros_like(gdHS)
    for i in range(ncomp):
        gammacii[:, i] = phi7[0] * (-np.tanh(phi7[1] * (phi7[2] - alphaii[i])) +
                                    1.0) * etaxstar * theta[i] * np.exp(phi7[3] * etaxstar + phi7[4] * (etaxstar**2))

    a2iidchi = calc_a2ii_1pchi(rhos, epsilonii, dii3, dkl, xskl, x0ii, l_rii, l_aii)

    da2iidrhos = calc_da2ii_1pchi_drhos(rhos, epsilonii, dii3, dkl, xskl, x0ii, l_rii, l_aii, stepmult=stepmult)

    a1sii_2l_aii = funcs.calc_a1s(rhos, 1.0, 2.0 * l_aii, etax, epsilonii, dii3**(1.0 / 3.0))
    a1sii_2l_rii = funcs.calc_a1s(rhos, 1.0, 2.0 * l_rii, etax, epsilonii, dii3**(1.0 / 3.0))
    a1sii_l_riil_aii = funcs.calc_a1s(rhos, 1.0, l_aii + l_rii, etax, epsilonii, dii3**(1.0 / 3.0))

    Bii_2l_aii = funcs.calc_Bkl(rhos, 2.0 * l_aii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)
    Bii_2l_rii = funcs.calc_Bkl(rhos, 2.0 * l_rii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)
    Bii_l_aiil_rii = funcs.calc_Bkl(rhos, l_aii + l_rii, 1.0, dii3**(1.0 / 3.0), epsilonii, x0ii, etax)

    eKC2 = np.einsum("i,j->ij", KHS / rhos, epsilonii * (Cii**2))

    g2MCA = (1.0 /
             (2.0 * np.pi *
              (epsilonii**2) * dii3)) * ((3.0 * da2iidrhos) - (eKC2 * l_rii * (x0ii**(2.0 * l_rii))) *
                                         (a1sii_2l_rii + Bii_2l_rii) + eKC2 * (l_rii + l_aii) *
                                         (x0ii**(l_rii + l_aii)) * (a1sii_l_riil_aii + Bii_l_aiil_rii) - eKC2 * l_aii *
                                         (x0ii**(2.0 * l_aii)) * (a1sii_2l_aii + Bii_2l_aii))

    #print np.size(g2MCA,axis=0),np.size(g2MCA,axis=1)
    #print np.size(gammacii,axis=0),np.size(gammacii,axis=1)
    g2 = (1.0 + gammacii) * g2MCA
    #g2=np.einsum("i,ij->ij",1.0+gammacii,g2MCA)

    #print np.exp((epsilonii*g1/(kT*gdHS))+(((epsilonii/kT)**2)*g2/gdHS))
    #try:
    gii = gdHS * np.exp((epsilonii * g1 / (kT * gdHS)) + (((epsilonii / kT)**2) * g2 / gdHS))
    tmp = [(epsilonii * g1 / (kT * gdHS)), (((epsilonii / kT)**2) * g2 / gdHS)]
    #except:
    #    print gdHS,epsilonii,g1,kT,gdHS,epsilonii,kT,g2,gdHS
    Achain = 0.0
    tmp_A = [0, 0]
    for i in range(ncomp):
        beadsum = -1.0

        for k in range(ngroups):
            beadsum += (nui[i, k] * beadlibrary[beads[k]]["Vks"] * beadlibrary[beads[k]]["Sk"])

        Achain -= xi[i] * beadsum * np.log(gii[:, i])
        tmp_A[0] -= tmp[0][:, i]
        tmp_A[1] -= tmp[1][:, i]

    #return Achain,sigmaii3,epsilonii,np.array(tmp_A)
    return Achain, sigmaii3, epsilonii
