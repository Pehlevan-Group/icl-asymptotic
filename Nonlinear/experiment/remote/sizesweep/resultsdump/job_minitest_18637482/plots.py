import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import optax

sys.path.append('../../../../')
sys.path.append('../../../../../')

from common import *
from train import train, create_train_state

trainvals = []
testvals = []
for i in range(100):
    file_path = f'./train-{i}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']][-1]
    testloss = [Metrics.loss for Metrics in loaded['test']][-1]
    trainvals.append(trainloss.item())
    testvals.append(testloss.item())

trainvals=np.array(trainvals)
testvals=np.array(testvals)

plt.plot(np.linspace(0.1,100,100),trainvals)
plt.xlabel('tau')
plt.title(f'Training Error')
plt.legend()
plt.savefig(f'duringtraining.png')