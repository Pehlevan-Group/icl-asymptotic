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
for i in range(26):
    file_path = f'./{mydir}/pickles/train-{i}-0.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())
# cutoff = 0.001
# overparam = [i for i in range(len(trainvals)) if trainvals[i] < cutoff]
# print("Tau Inflection at tau = ",overparam[-1])

taus = np.linspace(0.1,6.1,61)[:26]
# 40 np.array(list(np.linspace(0.1,3.1,31)) + list(np.linspace(3.5,6,6)))[:37]
#np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5])[:21]

trainvals = np.array(trainvals)
myarr = np.array(trainvals[0]/trainvals)
plt.scatter(taus,trainvals,c='black',label='Final Training Error')
plt.legend()
plt.title(f'Train Error Transition - d = {myd}')
plt.savefig(f'{mydir}/1L-transition-{myd}.png')