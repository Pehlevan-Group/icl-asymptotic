import numpy as np
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainminiold import train
from model.transformerold import TransformerConfig
from task.regression import FiniteSampler
from task.regression import LinearRegressionCorrect

sigma = 0.1;
psi = 1;
alpha = 0.5; tau = 0.5;

myname = sys.argv[1] # grab value of $mydir to add results
d = int(sys.argv[2])
N = int(alpha*d); P = int(tau*(d**2));

#Ks20_original = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(10*d),30)));
Ks80 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(5*d),15)))
Ks40 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(100*d),20)))
Ks20 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 28, 35, 45, 57, 69, 82, 96, 109, 123, 136, 150, 163, 177, 190, 204, 217, 231, 244, 258, 271, 285, 298, 312, 325, 339, 352, 366, 379, 393, 406, 420, 433, 447, 460];
if d == 20:
   Ks = np.array(Ks20);
if d == 40:
   Ks = np.array(Ks40);
if d == 80:
   Ks = np.array(Ks80);

kappaind = int(sys.argv[3]); # grab value of $SLURM_ARRAY_TASK_ID to index over taus 
avgind = int(sys.argv[4]); # grab value of $SLURM_ARRAY_TASK_ID to index over experiment repeats 
#);

K = Ks[kappaind]

h = d;

trainobject = FiniteSampler(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, diversity=K, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2, n_mlp_layers=1, pure_linear_self_att=False)
print("start training")
state, hist = train(config, data_iter=iter(trainobject), loss='mse', batch_size=int(0.1*P), test_every=1000, train_iters=100000, optim=optax.adamw,lr=1e-4)

testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
avgerr = 0;
loss_func = optax.squared_error
numsamples = 1000
for _ in range(numsamples):
  xs, labels = next(testobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;
print("kappa = ", K/d, " and icl error = ", avgerr)
file_path = f'./{myname}/icl-{kappaind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}\n')

testobject = trainobject
avgerr = 0;
loss_func = optax.squared_error
numsamples = 1000
for _ in range(numsamples):
  xs, labels = next(testobject); # generates data
  logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
  avgerr = avgerr + loss_func(logits, labels).mean()
avgerr = avgerr/numsamples;
print("kappa = ", K/d, " and idg error = ", avgerr)
file_path = f'./{myname}/idg-{kappaind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}\n')

file_path = f'./{myname}/pickles/train-{kappaind}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)