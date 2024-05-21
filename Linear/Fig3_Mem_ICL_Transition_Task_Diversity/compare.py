import numpy as np
from bayes_estimator import *

d=100
alphas = [0.25,0.5,0.75,1,5,10,50];
kappas = np.logspace(np.log10(0.1),np.log10(100),50); Ks = np.int64(kappas*d);

icl_ridge = np.zeros((len(alphas),len(kappas))); icl_dmmse = np.zeros((len(alphas),len(kappas)));
idg_ridge = np.zeros((len(alphas),len(kappas))); idg_dmmse = np.zeros((len(alphas),len(kappas)));

for a in range(len(alphas)):
    N = np.int64(alphas[a]*d);
    icl, idg = bayes_estimator(d,Ks,N,1,0.5,1000)
    icl_ridge[a,:] = icl[0,:]; icl_dmmse[a,:] = icl[1,:]; 
    idg_ridge[a,:] = idg[0,:]; idg_dmmse[a,:] = idg[1,:]; 

np.savetxt('icl_ridge.csv', icl_ridge, delimiter=',')
np.savetxt('icl_dmmse.csv', icl_dmmse, delimiter=',')
np.savetxt('idg_ridge.csv', icl_ridge, delimiter=',')
np.savetxt('idg_dmmse.csv', icl_dmmse, delimiter=',')