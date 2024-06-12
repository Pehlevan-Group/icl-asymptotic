import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

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

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

tau = int(sys.argv[1])
iter = 0

dirs = ['job_linear20',  'job_linear40' , 'job_linear80']

full_train = []
full_test = []

for idx, mydir in enumerate(dirs):
    trainvals = []
    testvals = []
    file_path = f'./{mydir}/pickles/train-{tau}-{iter}.pkl'
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
    full_train.append(trainvals)
    full_test.append(testvals)

vars_train = []
vars_test = []
for i in range(len(dirs)):
    vars_train.append([np.var(full_train[i][j-5:j+1]) for j in range(10,len(full_train[i]))])
    vars_test.append([np.var(full_test[i][j-5:j+1]) for j in range(10,len(full_test[i]))])

match_indices = []
for i in range(len(dirs)):
    match_indices.append(find_closest_index(vars_train[i],vars_train[-1][-1]))
    print(vars_train[i][match_indices[-1]])
    
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
for i in range(len(dirs)):
    # ax1.plot(range(len(full_train[i])), full_train[i],label=f'train, {dirs[i]}')
    # ax2.plot(range(len(full_test[i])), full_test[i],label=f'test, {dirs[i]}')
    ax1.plot(range(len(vars_train[i])), vars_train[i],label=f'train, {dirs[i]}')
    ax1.axvline(x=match_indices[i])
    ax2.plot(range(len(full_test[i])), full_test[i],label=f'test, {dirs[i]}')
    ax2.axvline(x=match_indices[i]+10)
ax1.legend()
ax2.legend()
fig.savefig(f'compare-{tau}-{iter}.png')