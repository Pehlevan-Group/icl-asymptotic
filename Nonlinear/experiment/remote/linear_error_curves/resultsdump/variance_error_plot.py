import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

def find_closest_index(arr, target):
    # Check if the array is empty
    if not arr:
        return None
    # Initialize variables to store the index and the smallest difference
    closest_index = 0
    smallest_difference = abs(arr[0] - target)
    # Iterate through the array
    for i in range(1, len(arr)):
        current_difference = abs(arr[i] - target)
        if current_difference < smallest_difference:
            smallest_difference = current_difference
            closest_index = i
    return closest_index

taus_so_far = 24
taus = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5,7,10])[:taus_so_far]
num_iters = 5

mydir = sys.argv[1]
total_train = []
total_test = []
total_var = []
for tauindex in range(taus_so_far):
    partial_train = []; partial_test = []; partial_var = []
    for iter in range(num_iters):
        # Load the data for this file 
        trainvals = []
        testvals = []
        file_path = f'./{mydir}/pickles/train-{tauindex}-{iter}.pkl'
        with open(file_path, 'rb') as fp:
            loaded = pickle.load(fp)
        trainloss = [Metrics.loss for Metrics in loaded['train']]
        testloss = [Metrics.loss for Metrics in loaded['test']]
        for loss_array in trainloss:
            trainvals.append(loss_array.item())
        for loss_array in testloss:
            testvals.append(loss_array.item())
        trainvals=np.array(trainvals)
        testvals=np.array(testvals)
        # Compute variance profile
        partial_train.append(trainvals); partial_test.append(testvals); partial_var.append([np.var(trainvals[j-5:j+1]) for j in range(10,len(trainvals))])
    total_train.append(partial_train); total_test.append(partial_test); total_var.append(partial_var);

plotvals = np.zeros((taus_so_far,num_iters))
plotvals_orig = np.zeros((taus_so_far,num_iters))
for tauindex in range(taus_so_far):
    for iter in range(num_iters):
        idx = find_closest_index(total_var[tauindex][iter], total_var[-1][iter][-1])
        plotvals[tauindex,iter] = total_test[tauindex][iter][idx+7];
        plotvals_orig[tauindex,iter] = total_test[tauindex][iter][-1];

plt.plot(taus, np.mean(plotvals,axis=1),label='variance matching')
plt.plot(taus, np.mean(plotvals_orig,axis=1),label='fixed gradient steps')
plt.legend()
plt.savefig("pleasework.png")


