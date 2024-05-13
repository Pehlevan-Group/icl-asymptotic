import pickle
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import optax
import os

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from train import train, create_train_state

mydir = sys.argv[1]
myiter = int(sys.argv[2])

trainvals = []
testvals = []

for i in range(25):
    file_path = f'./{mydir}/pickles/train-{i}-{myiter}.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fp:
            loaded = pickle.load(fp)
        trainloss = [Metrics.loss for Metrics in loaded['train']]
        loss_array = trainloss[-1]
        trainvals.append(loss_array.item())
    else:
        trainvals.append(trainvals[-1])

#taus = np.linspace(0.1,50.1,26)
taus = np.linspace(2,50,25)

# def paramrelu(xs, a, b, c):
#     return [a*max(b,x)+c for x in xs]
# def lossrelu(params, xs,ys):
#     a,b,c = params
#     yhat = np.array(paramrelu(xs,a,b,c))
#     return np.mean((yhat - ys)**2)
# def learnrelu(xs, ys):
#     result = minimize(lossrelu,[1,20,-20],args=(xs,ys),method='BFGS')
#     return result.x[0],result.x[1],result.x[2]

# plt.scatter(taus,trainvals,c='black',label='Final Training Error')
# a,b,c = learnrelu(taus, trainvals)
# print("slope", a, "elbow", b, "offset", c)
# plt.plot(taus,paramrelu(taus,a,b,c),c='blue',label='Relu Fit')
# plt.axvline(x=b,c='red',label='RELU Predicted Transition')
# plt.title(f'Train Error Regime Transition')
# plt.legend()
# plt.savefig(f'../plots/transition-relu-{myd}.png')

trainvals = np.array(trainvals)
myarr = np.array(trainvals[0]/trainvals)
sums = [np.sum(myarr[0:i+1]) for i in range(len(myarr))]
threshold = 0.9995; myval = threshold*sums[-1]
tind = np.where(sums > myval)[0][0]
tstar = taus[tind]
print("tstar",tstar)
plt.scatter(taus,trainvals,c='black',label='Final Training Error')
plt.axvline(x=tstar,c='red',label=f'tau* = {tstar} (cummulative sum w threshold {threshold})')
plt.legend()
plt.title(f'Train Error Transition for {mydir} iteration {myiter}')
plt.savefig(f'{mydir}/transition-{myiter}.png')