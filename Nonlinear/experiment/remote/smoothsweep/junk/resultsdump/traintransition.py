import pickle
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

mydir = sys.argv[1]
myiter = int(sys.argv[2])

early = 10;
trainvals = []
trainvals_early = []
testvals = []
testvals_early = []

for i in range(3,25):
    file_path = f'./{mydir}/pickles/train-{i}-{myiter}.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fp:
            loaded = pickle.load(fp)
        trainloss = [Metrics.loss for Metrics in loaded['train']]
        trainvals.append(trainloss[-1].item())
        trainvals_early.append(trainloss[early].item())
        trainloss = [Metrics.loss for Metrics in loaded['test']]
        testvals.append(trainloss[-1].item())
        testvals_early.append(trainloss[early].item())
    else:
        trainvals.append(trainvals[-1])
        testvals.append(testvals[-1])

#taus = np.linspace(0.1,50.1,26)
taus = range(1,81)[3:25]


trainvals = np.array(trainvals); trainvals_early=np.array(trainvals_early)
testvals = np.array(testvals)
myarr = np.array(trainvals[0]/trainvals)
def relumodel(x, a, b,c):
    return np.array([-a*min(b,x_i)+c for x_i in x])

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


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sums = [np.sum(myarr[0:i+1]) for i in range(len(myarr))]
threshold = 0.999; myval = threshold*sums[-1]
tind = np.where(sums > myval)[0][0]
tstar = taus[tind]
print("tstar",tstar)
ax1.scatter(taus,myarr,c='black',label='inverse Final Training Error')
#ax1.axvline(x=tstar,c='red',label=f'tau* = {tstar} (cummulative sum w threshold {threshold})')
ax1.scatter(taus,np.array(trainvals_early[0]/trainvals_early),label=f'earlier stop {early}')
a0, b0, c0 = [0.1,20,0]
popt, pcov = curve_fit(relumodel, taus[1:], myarr[1:],[a0,b0,c0])
# Get the parameters a and c
a, b,c = popt
print(a,b,c)
ax1.plot(taus,relumodel(taus,a,b,c),'-',label='fit')
#ax1.plot(taus,relumodel(taus,a0,b0,c0),'-',label='original wtf')
ax1.legend()
ax1.set_title(f'Train Error Transition for {mydir} iteration {myiter}')

ax2.scatter(taus,testvals,c='black',label='Final test Error')
ax2.scatter(taus,testvals_early,label=f'earlier stop {early}')

ax2.legend()
ax2.set_title(f'Test Error Transition for {mydir} iteration {myiter}')

fig.savefig(f'{mydir}/transition-{myiter}.png')
