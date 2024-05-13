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
from trainmini import train, create_train_state, get_random_batch, compute_metrics
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect, FiniteSampler

sigma = 0.1;
psi = 1;
alpha = 9;
d = 20;
N = int(alpha*d);
tau = 50;
P = int(tau*(d**2));

h = 10*d;

myname = sys.argv[1] # grab value of $mydir to add results

devices = jax.devices()
if any(device.device_kind == 'gpu' for device in devices):
    print("Main code is running on a GPU.")
else:
    print("Main code is not running on a GPU.")
trainobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, data_cov = 1, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2, n_mlp_layers=1, pure_linear_self_att=False)

init_rng = jax.random.key(new_seed())
model = config.to_model()

data_iter = iter(trainobject);
samp = next(data_iter); 
mini_samp_x, _ = get_random_batch(samp, np.min([10000,P]));

loss_func = optax.squared_error
state = create_train_state(init_rng, model, mini_samp_x, optim=optax.adamw)
state = compute_metrics(state, samp) 
xs, labels = next(trainobject); # generates data
devices = jax.devices()
if any(device.device_kind == 'gpu' for device in devices):
    print("inference call is running on a GPU.")
else:
    print("inference call is not running on a GPU.")
logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
avgerr = loss_func(logits, labels).mean()

print("success!!")