import numpy as np
from bayes_estimator.py import *

d=100
alphas = [0.25,0.5,0.75,1,5,10,50];
kappas = np.logspace(np.log(0.1),np.log(100),50); Ks = np.int64(kappas*d);

ICL = np.zeros(len(alphas),2,len(kappas)); IDG = np.zeros(len(alphas),2,len(kappas));
for a in range(len(alphas)):
    N = np.int64(alphas[a]*d);
    icl, idg = bayes_estimator(d,Ks,N,1,0.5,1000)
    ICL[a,:,:] = icl; IDG[a,:,:] = icl;

icl_ridge = icl[:,0,:]; icl_dmmse= icl[:,1,:]; 
idg_ridge = idg[:,0,:]; idg_dmmse = icl[:,1,:]; 

np.savetxt('icl_ridge.csv', icl_ridge, delimiter=',')
np.savetxt('icl_dmmse.csv', icl_dmmse, delimiter=',')
np.savetxt('idg_ridge.csv', icl_ridge, delimiter=',')
np.savetxt('idg_dmmse.csv', icl_dmmse, delimiter=',')