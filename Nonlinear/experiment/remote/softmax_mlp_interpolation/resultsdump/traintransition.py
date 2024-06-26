import pickle
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

mydir = sys.argv[1]
myd = int(sys.argv[2])

trainvals = []
testvals = []

#mydir='3l20_22143820'
for i in range(21):
    file_path = f'./{mydir}/pickles/train-{i}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())
# cutoff = 0.001
# overparam = [i for i in range(len(trainvals)) if trainvals[i] < cutoff]
# print("Tau Inflection at tau = ",overparam[-1])

taus = np.linspace(2.5,4.5,21) #np.linspace(2.5,4.5,21) #np.linspace(0.1,2.1,21)
trainvals = np.array(trainvals)
myarr = np.array(trainvals[0]/trainvals)
# sums = [np.sum(myarr[0:i+1]) for i in range(len(myarr))]
# threshold = 0.9995; myval = threshold*sums[-1]
# tind = np.where(sums > myval)[0][0]
# tstar = taus[tind]
# print("tstar",tstar)
plt.scatter(taus,trainvals,c='black',label='Final Training Error')
#plt.axvline(x=tstar,c='red',label=f'tau* = {tstar} (cummulative sum w threshold {threshold})')
plt.legend()
plt.title(f'Train Error Transition - d = {myd}')
plt.savefig(f'{mydir}/1L-transition-{myd}.png')