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
from task.regression import FiniteSampler
from task.regression import LinearRegressionCorrect

d=7;
tvals = range(1,61)
alpha = 1; N = int(alpha*d);
h = 100;

sigma = 0.25;
psi = 1;

myname = sys.argv[1] # grab value of $mydir to add results
tauind = int(sys.argv[2]) - 1; # grab value of $SLURM_ARRAY_TASK_ID to index over taus 
P = int(tvals[tauind]*(d**2));

trainobject = FiniteSampler(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, diversity=P, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2)

state, hist = train(config, data_iter=iter(trainobject), loss='mse', test_every=1000, train_iters=500000, optim=optax.adamw,lr=1e-4)

avgerr = 0;
loss_func = optax.squared_error
numsamples = 10000
testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);

for _ in range(numsamples):
  xs, labels = next(testobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;

file_path = f'./{myname}/errors/error-{tauind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}')
file_path = f'./{myname}/pickles/train-{tauind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
