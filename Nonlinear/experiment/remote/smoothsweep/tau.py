import jax
from flax.serialization import from_state_dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from traintheory import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect, FiniteSampler

d=20;
tvals = np.linspace(0.1,50.1,26)
alpha = 1; N = int(alpha*d);
h = 10*d;

sigma = 0.25;
psi = 1;

myname = sys.argv[1] # grab value of $mydir to add results
tauind = int(sys.argv[2]); # grab value of $SLURM_ARRAY_TASK_ID to index over taus 
avgind = int(sys.argv[3]); # grab value of $SLURM_ARRAY_TASK_ID to index over experiment repeats 
mykappa = float(sys.argv[4]); myK = int(mykappa*d);
P = int(tvals[tauind]*(d**2));

trainobject = FiniteSampler(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, diversity = myK, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2)

state, hist = train(config, data_iter=iter(trainobject), loss='mse', test_every=1000, train_iters=500000, optim=optax.adamw,lr=1e-4)

testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
avgerr = 0;
loss_func = optax.squared_error
numsamples = 5000
for _ in range(numsamples):
  xs, labels = next(testobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;

file_path = f'./{myname}/error-{tauind}-{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}')
file_path = f'./{myname}/pickles/train-{tauind}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)