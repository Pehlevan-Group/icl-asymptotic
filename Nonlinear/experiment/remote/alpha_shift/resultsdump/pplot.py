import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

mydir = sys.argv[1]
myfile = sys.argv[2]

trainvals = []
idgvals = []
iclvals = []
file_path = f'./{mydir}/pickles/{myfile}.pkl'
with open(file_path, 'rb') as fp:
    loaded = pickle.load(fp)
trainloss = [Metrics.loss for Metrics in loaded['train']]
idgloss = [Metrics.loss for Metrics in loaded['test']]
iclloss = [Metrics.loss for Metrics in loaded['true_test']]
for loss_array in trainloss:
    trainvals.append(loss_array.item())
for loss_array in idgloss:
    idgvals.append(loss_array.item())
for loss_array in iclloss:
    iclvals.append(loss_array.item())
trainvals=np.array(trainvals)
idgvals=np.array(idgvals)
iclvals=np.array(iclvals)

plt.plot(range(len(trainvals)),trainvals,label='Train')
plt.plot(range(len(idgvals)),idgvals,label='IDG testing')
plt.plot(range(len(iclvals)),iclvals,label='ICL testing')
plt.title(f'{myfile} train and tests')
plt.legend()
plt.savefig(f'./{mydir}/pickles/{myfile}plot.png')