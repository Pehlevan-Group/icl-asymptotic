import numpy as np
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainmini import train
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect, FiniteSampler

sigma = 0.1;
psi = 1;

myname = sys.argv[1] # grab value of $mydir to add results
d = int(sys.argv[2])
alpha = float(sys.argv[3]); N = int(alpha*d);
kappa = float(sys.argv[4]); K = int(kappa*d);
tau = float(sys.argv[5]); P = int(tau*(d**2));
#avgind = int(sys.argv[5]); # grab value of $SLURM_ARRAY_TASK_ID to index over experiment repeats 
h = d;

trainobject = FiniteSampler(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, diversity = K, seed=None);
testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=0, pure_linear_self_att=False)
print("start training")
state, hist = train(config, data_iter=iter(trainobject), test_iter=iter(testobject), loss='mse', batch_size=int(0.1*P), test_every=500, train_iters=100000, optim=optax.adamw,lr=1e-4)

# Alphas = np.linspace(0.1,5,50)
# tests = []
# for Alpha in Alphas:
#   testobject = LinearRegressionCorrect(n_points = int(Alpha*d)+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
#   avgerr = 0;
#   loss_func = optax.squared_error
#   numsamples = 1000
#   for _ in range(numsamples):
#     xs, labels = next(testobject); # generates data
#     logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#     avgerr = avgerr + loss_func(logits, labels).mean()
#   avgerr = avgerr/numsamples;
#   tests.append(avgerr)

# print(tests)

# file_path = f'./{myname}/error-alpha{alpha}-kappa{kappa}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'{tests}\n')
file_path = f'./{myname}/pickles/train.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)