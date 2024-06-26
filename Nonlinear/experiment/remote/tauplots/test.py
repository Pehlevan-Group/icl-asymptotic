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
from traintheory import train
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect

d=2;
t = 10;
P = int(t*(d**2));
alpha = 5; N = int(alpha*d);

sigma = 0.25;
psi = 1;

linobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=128)

state, hist = train(config, data_iter=iter(linobject), loss='mse', test_every=1000, train_iters=5000, lr=1e-4, l2_weight = 0.1)

avgerr = 0;
loss_func = optax.squared_error
for _ in range(50):
  xs, labels = next(linobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()

avgerr = avgerr/50;
file_path = f'../../../../resultstheory/mixedreg10d{i}.txt'
with open(file_path, 'w') as file:
    file.write(f'{avgerr}')
file_path = f'../../../../resultstheory/mixedreg10d{i}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)

