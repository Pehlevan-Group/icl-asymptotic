import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm as tqdm

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

mydir = sys.argv[1]
indexrange = range(61,120)
exprange = range(5)

def find_increase_start_index(arr, tolerance=1, consistency=3):
    # Tolerance: maximum allowed increase for it to be considered "not too much"
    # Consistency: number of consecutive increases needed to consider it a trend
    increasing_streak = 0
    
    for i in range(1, len(arr)):
        difference = arr[i] - arr[i - 1]
        
        if difference > tolerance:
            increasing_streak += 1
        else:
            increasing_streak = 0
        
        if increasing_streak >= consistency:
            return i - consistency + 1  # Return the starting index of the consistent increase
    
    return -1  # Return -1 if no consistent increase is found

early_errs = []
usual_errs = []
for tauind in tqdm(indexrange):
    early_temp = []
    usual_temp = []
    for expind in exprange:
        file_path = f'./{mydir}/pickles/train-{tauind}-{expind}.pkl'
        testvals = []
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fp:
                loaded = pickle.load(fp)
            testloss = [Metrics.loss for Metrics in loaded['test']]
            for loss_array in testloss:
                testvals.append(loss_array.item())
            usual_temp.append(testvals[-1])
            early_temp.append(testvals[find_increase_start_index(testvals,0.1,10)])
    usual_errs.append(usual_temp)
    early_errs.append(early_temp)

taus = np.linspace(0.1,6.1,61)
plt.plot(taus[range(len(indexrange))],[np.mean(usual_errs[i]) for i in range(len(usual_errs))],label='usual stopping')
plt.plot(taus[range(len(indexrange))],[np.mean(early_errs[i]) for i in range(len(early_errs))],label='early stopping')
plt.legend()
plt.savefig(f'./{mydir}/pickles/earlyplot.png')



