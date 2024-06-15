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
    file_path = f'./{mydir}/pickles/train-{i}-0.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())
# cutoff = 0.001
# overparam = [i for i in range(len(trainvals)) if trainvals[i] < cutoff]
# print("Tau Inflection at tau = ",overparam[-1])

taus = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5])[:21]

# def growth(myarr):
#     answ = []
#     for i in range(len(myarr)):
#         if i == 0:
#             answ.append(myarr[1]/myarr[0])
#         else:
#             answ.append(myarr[i]/myarr[i-1])
#     return answ

def paramrelu(xs, a, b, c):
    return [a*max(b,x)+c for x in xs]
def lossrelu(params, xs,ys):
    a,b,c = params
    yhat = np.array(paramrelu(xs,a,b,c))
    return np.mean((yhat - ys)**2)
def learnrelu(xs, ys):
    result = minimize(lossrelu,[1,20,-20],args=(xs,ys),method='BFGS')
    return result.x[0],result.x[1],result.x[2]

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
#plt.axvline(x=tstar,c='red',label=f'tau* = {tstar} (cummulative sum w threshold {threshold})')
plt.legend()
plt.title(f'Train Error Transition - d = {myd}')
plt.savefig(f'{mydir}/1L-transition-{myd}.png')