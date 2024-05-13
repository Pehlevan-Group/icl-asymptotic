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
from task.regression import LinearRegressionCorrect, FiniteSampler

sigma = 0.1;
psi = 1;
alpha = 1;
taus = np.linspace(10,50,20); 
d = 20;
N = int(alpha*d);

h = 10*d;

myname = sys.argv[1] # grab value of $mydir to add results
tauindex = int(sys.argv[2]); # grab value of $SLURM_ARRAY_TASK_ID to index over kappas 
avgind = int(sys.argv[3]); # grab value of $SLURM_ARRAY_TASK_ID to index over different experiements 
P = int(taus[tauindex]*(d**2));
kappa = float(sys.argv[4]) # kappa value specified in program

if kappa == 0:
  trainobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
  mylabel = 'diverse';
else:
  K = int(kappa*d);
  trainobject = FiniteSampler(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, diversity=K, batch_size = P, seed=None);
  mylabel = f'{kappa}';
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2, n_mlp_layers=1, pure_linear_self_att=False)

state, hist = train(config, data_iter=iter(trainobject), batch_size=np.min([10000,P]), loss='mse', test_every=1000, train_iters=30000, optim=optax.adamw,lr=1e-4)

avgerr = 0;
loss_func = optax.squared_error
numsamples = 1000

testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
for _ in range(numsamples):
  xs, labels = next(testobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;
print(f'tau is {taus[tauindex]}, kappa is {kappa}, error is {avgerr}')
file_path = f'./{myname}/errors/error-{mylabel}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr} iteration {avgind}\n')
    
file_path = f'./{myname}/pickles/train-{mylabel}-{tauindex}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
