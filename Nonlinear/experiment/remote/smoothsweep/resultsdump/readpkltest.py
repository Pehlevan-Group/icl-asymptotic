import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

print("dumbfuck please run")

mydir = sys.argv[1]
testvals = []
for t in range(100):
    print(t)
    dummy = []
    for i in range(10):
        file_path = f'./{mydir}/pickles/train-{t}-{i}.pkl'
        with open(file_path, 'rb') as fp:
            loaded = pickle.load(fp)
        testloss = [Metrics.loss for Metrics in loaded['test']][300];
        dummy.append(testloss.item())
    dummy = np.array(dummy); testvals.append(dummy);

testvals = np.array(testvals);
print([list(i) for i in testvals])

plt.plot(range(1,101),np.mean(testvals,axis=1),label='Test at 300')
plt.title(f'pls work')
plt.legend()
plt.savefig(f'./300fixplot.png')