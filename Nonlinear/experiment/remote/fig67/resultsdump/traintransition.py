import pickle
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import optax

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from train import train, create_train_state

mydir = sys.argv[1]
myk = sys.argv[2]

trainvals = []
testvals = []

#mydir='3l20_22143820'
for i in range(20):
    file_path = f'./{mydir}/pickles/train-{myk}-{i}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['test']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())
# cutoff = 0.001
# overparam = [i for i in range(len(trainvals)) if trainvals[i] < cutoff]
# print("Tau Inflection at tau = ",overparam[-1])

taus = np.linspace(10,50,20)

# def growth(myarr):
#     answ = []
#     for i in range(len(myarr)):
#         if i == 0:
#             answ.append(myarr[1]/myarr[0])
#         else:
#             answ.append(myarr[i]/myarr[i-1])
#     return answ
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
plt.title(f'Train Error Transition - 2 Layers, 10*d hidden dim, kappa = {myk}')
plt.savefig(f'{mydir}/fig6test-{myk}.png')