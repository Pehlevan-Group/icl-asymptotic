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

sigma = 0.1;
psi = 1;

myname = sys.argv[1] # grab value of $mydir to add results
d = int(sys.argv[2])
tauind = int(sys.argv[3]); # grab value of $SLURM_ARRAY_TASK_ID to index over taus 
avgind = int(sys.argv[4]); # grab value of $SLURM_ARRAY_TASK_ID to index over experiment repeats 
tvals = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5]); P = int(tvals[tauind]*(d**2)); #np.array([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2,2.5,3,5,7,10]); 
alpha = 1; N = int(alpha*d);
h = d;

trainobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=0, pure_linear_self_att=True)
print("start training")
state, hist = train(config, data_iter=iter(trainobject), loss='mse', batch_size=int(0.1*P), test_every=1000, train_iters=20000, optim=optax.adamw,lr=1e-4)

testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
avgerr = 0;
loss_func = optax.squared_error
numsamples = 1000
for _ in range(numsamples):
  xs, labels = next(testobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;

print("tau = ", tvals[tauind], " and error = ", avgerr)

file_path = f'./{myname}/error-{tauind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}\n')
file_path = f'./{myname}/pickles/train-{tauind}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)