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
#);
tval = 10; P = int(tval*d**2)
bvals = [0.01,0.05,0.1,0.5,1]
#np.array(list(np.linspace(0.1,3.1,31)) + list(np.linspace(3.5,6,6))); 
B = bvals[tauind]
#np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5]); 
alpha = 1; N = int(alpha*d);
h = d;

trainobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=0, pure_linear_self_att=False)
print("start training")
state, hist = train(config, data_iter=iter(trainobject), loss='mse', batch_size=int(B*P), test_every=100, train_iters=5000, optim=optax.adamw,lr=1e-4)

testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
avgerr = 0;
loss_func = optax.squared_error

numsamples = 1000
for _ in range(numsamples):
  xs, labels = next(testobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;

file_path = f'./{myname}/error-{tauind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}\n')
file_path = f'./{myname}/pickles/train-{tauind}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)