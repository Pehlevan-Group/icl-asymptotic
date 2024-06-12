import numpy as np
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainmini import train
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect

tvals = np.linspace(0.95,1.15,11) #np.linspace(0.1,2.1,21)

sigma = 0.1;
psi = 1;
alpha = 1; 

myname = sys.argv[1] # grab value of $mydir to add results
d = int(sys.argv[2])
tauind = int(sys.argv[3]) - 1; # grab value of $SLURM_ARRAY_TASK_ID to index over taus 

N = int(alpha*d);
P = int(tvals[tauind]*(d**2));
h = d;

trainobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=0,pure_linear_self_att=True)
print("initialisation fine")
state, hist = train(config, data_iter=iter(trainobject), batch_size=int(0.1*P), loss='mse', test_every=1000, train_iters=10000, optim=optax.adamw,lr=1e-4)

# avgerr = 0;
# loss_func = optax.squared_error
# numsamples = 10000
# testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);

# for _ in range(numsamples):
#   xs, labels = next(testobject); # generates data
#   logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#   avgerr = avgerr + loss_func(logits, labels).mean()
# avgerr = avgerr/numsamples;

# file_path = f'./{myname}/errors/error-{tauind}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'{avgerr}')

file_path = f'./{myname}/pickles/train-{tauind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
