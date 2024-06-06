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
from trainmini import train
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect, FiniteSampler

sigma = 0.1;
psi = 1;
d = 30;
alpha = 5; N = int(alpha*d);
tau = 50; P = int(tau*(d**2));
kappas = [ 0.1,  0.2,  0.3,  0.4,  0.6,  0.7,  1. ,  1.3,  1.7,  2.2,  2.8,
        3.7,  4.8,  6.2,  8.1, 10.5, 13.6, 17.7, 22.9, 29.7, 38.5, 49.9]; #np.exp(np.linspace(np.log(0.1),np.log(50),25));
h = 10*d;

myname = sys.argv[1] # grab value of $mydir to add results
kappaind = int(sys.argv[2]) - 1 # kappa index specified in program
kappa = kappas[kappaind]; K = int(kappa*d);

totalavg = 0;
for avgind in range(6):
    trainobject = FiniteSampler(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, diversity=K, batch_size = P, seed=None);
    config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2, n_mlp_layers=1, pure_linear_self_att=False)

    state, hist = train(config, data_iter=iter(trainobject), batch_size=np.min([10000,P]), loss='mse', test_every=1000, train_iters=300000, optim=optax.adamw,lr=1e-4)

    avgerr = 0;
    loss_func = optax.squared_error
    numsamples = 1000

    testobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
    for _ in range(numsamples):
        xs, labels = next(testobject); # generates data
        logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
        avgerr = avgerr + loss_func(logits, labels).mean()
    avgerr = avgerr/numsamples;
    totalavg = totalavg + avgerr;
    file_path = f'./{myname}/pickles/train-{kappaind}-{avgind}.pkl'
    with open(file_path, 'wb') as fp:
        pickle.dump(hist, fp)
    file_path = f'./{myname}/errors/error-{kappaind}.txt'
    with open(file_path, 'a') as file:
        file.write(f'{avgerr}\n')

totalavg = totalavg/10;
file_path = f'./{myname}/errors/error-{kappaind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{totalavg}\n')
