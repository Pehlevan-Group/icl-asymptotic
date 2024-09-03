import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm as tqdm

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *


exprange = range(10)

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

indexrange = range(0,45)
mydir = 'job_20d_1a_10t'
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
            early_temp.append(testvals[-1])
    usual_errs.append(usual_temp)
    early_errs.append(early_temp)
u20 = usual_errs
e20 = early_errs

indexrange = range(0,35)
mydir = 'job_40d_1a_10t'
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
            print(len(testloss))
            for loss_array in testloss:
                testvals.append(loss_array.item())
            usual_temp.append(testvals[-1])
            early_temp.append(testvals[int(len(testvals)/2)])#testvals[find_increase_start_index(testvals,0.05,5)])
    usual_errs.append(usual_temp)
    early_errs.append(early_temp)
u40 = usual_errs
e40 = early_errs

indexrange = range(0,35)
mydir = 'job_80d_1a_10t'
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
            early_temp.append(testvals[int(len(testvals)/4)])#testvals[find_increase_start_index(testvals,0.05,5)])
    usual_errs.append(usual_temp)
    early_errs.append(early_temp)
u80 = usual_errs
e80 = early_errs

d=20
Ks20soft = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 28, 35, 45, 57, 69, 82, 96, 109, 123, 136, 150, 163, 177, 190, 204, 217, 231, 244, 258, 271, 285, 298, 312, 325, 339, 352, 366, 379, 393, 406, 420, 433, 447, 460])/d
d=40
Ks40soft = np.array(list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(10*d),30))))/d
d=80
K80soft = np.array(list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(5*d),15))))/d

plt.plot(Ks20soft, [np.mean(u20[i]) for i in range(len(u20))],':',color='red',label='usual stopping 20')
plt.plot(Ks40soft[range(len(u40))], [np.mean(u40[i]) for i in range(len(u40))],':',color='orange',label='usual stopping 40')
plt.plot(K80soft, [np.mean(u80[i]) for i in range(len(u80))],':',color='blue',label='usual stopping 80')
plt.plot(Ks20soft,[np.mean(e20[i]) for i in range(len(e20))],color = 'red', label='early stopping 20')
plt.plot(Ks40soft[range(len(e40))],[np.mean(e40[i]) for i in range(len(e40))],color = 'orange', label='early stopping 40')
plt.plot(K80soft,[np.mean(e80[i]) for i in range(len(e80))],color='blue',label='early stopping 80')
plt.xscale('log')
plt.legend()
plt.savefig(f'./earlyplot.png')

print([np.mean(e40[i]) for i in range(len(e40))])
print([np.mean(e80[i]) for i in range(len(e80))])



