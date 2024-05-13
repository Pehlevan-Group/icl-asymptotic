import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import optax

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from train import train, create_train_state

mydir = 'linestar5_19923005'
trainvals = []
testvals = []
for i in range(40):
    file_path = f'./{mydir}/pickles/train-{i}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())
mydir = 'linestar5_19951324'
for i in range(40):
    file_path = f'./{mydir}/pickles/train-{i}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())

cutoff = 0.001
overparam = [i for i in range(len(trainvals)) if trainvals[i] < cutoff]
print("Tau Inflection at tau = ",overparam[-1])
taus = range(1,81)

#plt.plot(taus,trainvals,label='Final Training Error')
#plt.axvline(x=overparam[-1],label='tau inflection')
# deltas = growth(trainvals)
# plt.plot(taus[0:30],deltas[0:30],label='delta Training Error')
#plt.axvline(x=25,label='empirical inflection')
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.set_yscale('log')
axes.plot(taus,trainvals,label='Final Training Error')
plt.title(f'Train Error Regime Transition')
plt.legend()
plt.savefig(f'./transition-d5-log.png')

# def computeslope(myarr):
#     answ=[]
#     for i in range(len(myarr)):
#         if i == 0:
#             answ.append(myarr[1]-myarr[0])
#         elif i == len(myarr)-1:
#             answ.append(myarr[i]-myarr[i-1])
#         else:
#             answ.append((myarr[i+1]-myarr[i-1])/2)
#     return answ


# trainslopes = computeslope(trainvals)
# plt.plot(range(40),trainslopes[0:40],label='slopes approx')
# plt.axvline(x=24,label='approx max')
# plt.legend()
# plt.savefig(f'./{mydir}/testplot-{myiteration}.png')
