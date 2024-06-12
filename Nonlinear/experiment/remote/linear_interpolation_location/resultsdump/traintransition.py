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
for i in range(11):
    file_path = f'./{mydir}/pickles/train-{i}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())
# cutoff = 0.001
# overparam = [i for i in range(len(trainvals)) if trainvals[i] < cutoff]
# print("Tau Inflection at tau = ",overparam[-1])

taus = np.linspace(0.95,1.15,11) # np.linspace(0.1,2.1,21)[:13]

# def growth(myarr):
#     answ = []
#     for i in range(len(myarr)):
#         if i == 0:
#             answ.append(myarr[1]/myarr[0])
#         else:
#             answ.append(myarr[i]/myarr[i-1])
#     return answ

# def paramrelu(xs, a, b, c):
#     return [a*max(b,x)+c for x in xs]
# def lossrelu(params, xs,ys):
#     a,b,c = params
#     yhat = np.array(paramrelu(xs,a,b,c))
#     return np.mean((yhat - ys)**2)
# def learnrelu(xs, ys):
#     result = minimize(lossrelu,[1,20,-20],args=(xs,ys),method='BFGS')
#     return result.x[0],result.x[1],result.x[2]

# # plt.scatter(taus,trainvals,c='black',label='Final Training Error')
# # a,b,c = learnrelu(taus, trainvals)
# # print("slope", a, "elbow", b, "offset", c)
# # plt.plot(taus,paramrelu(taus,a,b,c),c='blue',label='Relu Fit')
# # plt.axvline(x=b,c='red',label='RELU Predicted Transition')
# # plt.title(f'Train Error Regime Transition')
# # plt.legend()
# # plt.savefig(f'../plots/transition-relu-{myd}.png')

trainvals = np.array(trainvals)
myarr = np.array(trainvals[0]/trainvals)
sums = [np.sum(myarr[0:i+1]) for i in range(len(myarr))]
threshold = 0.9995; myval = threshold*sums[-1]
tind = np.where(sums > myval)[0][0]
tstar = taus[tind]
print("tstar",tstar)


plt.scatter(taus,trainvals,c='black',label='Final Training Error')
more = [3.407228723517619e-05, 3.0324617910082452e-05, 2.3542552298749797e-05, 2.3064530978444964e-05, 2.7070831492892466e-05, 3.093613850069232e-05, 4.410104884300381e-05, 4.042880027554929e-05, 6.266944546950981e-05, 0.00034977233735844493, 0.02390388213098049, 0.0966218039393425, 0.14506091177463531, 0.15114298462867737, 0.24264347553253174]
plt.scatter(np.linspace(0.1,2.1,21)[:15],more)
#plt.axvline(x=tstar,c='red',label=f'tau* = {tstar} (cummulative sum w threshold {threshold})')
plt.legend()
plt.title(f'Train Error Transition - d = {myd}')
plt.savefig(f'{mydir}/1L-transition-{myd}.png')