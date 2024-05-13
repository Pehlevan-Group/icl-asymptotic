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

sigma = 0.25;
psi = 1;
alpha = 1;
d = 20;
N = int(alpha*d);

h = 10*d;

taus = [1,10,20,30,40];

myname = sys.argv[1] # grab value of $mydir to add results
tauind = int(sys.argv[2]) - 1; # grab value of $SLURM_ARRAY_TASK_ID to index over taus 
P = int(taus[tauind]*(d**2));

trainobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, data_cov = 1/np.sqrt(alpha), batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2, n_mlp_layers=1, pure_linear_self_att=False)

state, hist = train(config, data_iter=iter(trainobject), batch_size=np.min([10000,P]), loss='mse', test_every=1000, train_iters=30000, optim=optax.adamw,lr=1e-4)

avgerr = 0;
loss_func = optax.squared_error
numsamples = 10000

alphatests = np.linspace(0.1,2,20);
for alphatest in alphatests:
    Ntest = int(alphatest*d);
    testobject = LinearRegressionCorrect(n_points = Ntest+1, n_dims= d, eta_scale = sigma, w_scale = psi, data_cov = 1/np.sqrt(alphatest), batch_size = P, seed=None);
    avgerr = 0;
    for _ in range(numsamples):
      xs, labels = next(testobject); # generates data
      logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
      avgerr = avgerr + loss_func(logits, labels).mean()
    avgerr = avgerr/numsamples;
    file_path = f'./{myname}/errors/error-{tauind}.txt'
    with open(file_path, 'a') as file:
        file.write(f'{avgerr}\n')
    
file_path = f'./{myname}/pickles/train-{tauind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
