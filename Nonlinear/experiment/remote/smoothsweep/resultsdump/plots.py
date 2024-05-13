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

mydir = sys.argv[1]
myfile = sys.argv[2]

trainvals = []
testvals = []
file_path = f'./{mydir}/pickles/{myfile}.pkl'
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

plt.plot(range(len(trainvals)),trainvals,label='Train')
plt.plot(range(len(testvals)),testvals,label='Test')
plt.title(f'{myfile} train and test')
plt.legend()
plt.savefig(f'./{mydir}/pickles/{myfile}plot.png')