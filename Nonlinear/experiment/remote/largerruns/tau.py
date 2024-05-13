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
from trainmini import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression
from task.regression import LinearRegressionCorrect

d=40;
tvals = range(1,51)
alpha = 1; N = int(alpha*d);
h = 200;

sigma = 0.25;
psi = 1;

myname = sys.argv[1] # grab value of $mydir to add results
tauind = int(sys.argv[2]); # grab value of $SLURM_ARRAY_TASK_ID to index over taus 
avgind = int(sys.argv[3]); # grab value of $SLURM_ARRAY_TASK_ID to index over experiment repeats 
P = int(tvals[tauind]*(d**2));

linobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2)

state, hist = train(config, data_iter=iter(linobject), batch_size=np.min([10000,P]), loss='mse', test_every=1000, train_iters=500000, optim=optax.adamw,lr=1e-4)

avgerr = 0;
loss_func = optax.squared_error
numsamples = 10000
for _ in range(numsamples):
  xs, labels = next(linobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;
print(avgerr)

file_path = f'./{myname}/errors/error-{tauind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}\n')
file_path = f'./{myname}/pickles/train-{tauind}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
# # code for saving K,Q,V matrices in each layer??
# file_path = f'./{myname}/kqv/train-{i}.pkl'