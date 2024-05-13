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
trainvals = []
testvals = []

for i in range(30):
    file_path = f'./{mydir}/pickles/train-{i}-0.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())

taus = range(11,41)

trainvals = np.array(trainvals)
myarr = np.array(trainvals[0]/trainvals)
sums = [np.sum(myarr[0:i+1]) for i in range(len(myarr))]
threshold = 0.999; myval = threshold*sums[-1]
tstar = np.where(sums > myval)[0][0]
print(tstar)
plt.plot(taus,(1/trainvals)*trainvals[0],label='Final Training Error')
plt.axvline(x=tstar,label='Predicted Transition')
plt.legend()
plt.title(f'Train Error Transition - Quad')
plt.savefig(f'../plots/transitionquad.png')