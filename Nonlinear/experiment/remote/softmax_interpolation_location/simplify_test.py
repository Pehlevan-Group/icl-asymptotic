import numpy as np
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainminiold import train
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect

tvals = [1.5]#np.linspace(1,1.5,11)

sigma = 0.1;
psi = 1;
alpha = 1; 

myname = sys.argv[1] # grab value of $mydir to add results
d = int(sys.argv[2])
tauind = int(sys.argv[3]) - 1; # grab value of $SLURM_ARRAY_TASK_ID to index over taus 

N = int(alpha*d);
P = int(tvals[tauind]*(d**2));
h = d+1;

trainobject = LinearRegressionCorrect(n_points = N+1, n_dims= d, eta_scale = sigma, w_scale = psi, batch_size = P, seed=None);
config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=0,pure_linear_self_att=True)
state, hist = train(config, data_iter=iter(trainobject), batch_size=int(0.1*P), loss='mse', test_every=100, train_iters=5000, optim=optax.sgd,lr=1e-4)
myparams = state.params
# key = np.array(myparams['TransformerBlock_0']['SingleHeadSelfAttention_0']['key']['kernel'])
# query = np.array(myparams['TransformerBlock_0']['SingleHeadSelfAttention_0']['query']['kernel'])
# value = np.array(myparams['TransformerBlock_0']['SingleHeadSelfAttention_0']['value']['kernel'])
key = np.array(myparams['LinearSelfAttentionBlock_0']['key']['kernel'])
query = np.array(myparams['LinearSelfAttentionBlock_0']['query']['kernel'])
value = np.array(myparams['LinearSelfAttentionBlock_0']['value']['kernel'])
attention = key.T @ query;
V_11 = value[:d,:d]
v_12 = value[:d,d]
v_21 = value[d,:d]
v = value[d,d]
M_11 = attention[:d,:d]
m_12 = attention[:d,d]
m_21 = attention[d,:d]
m = attention[d,d]

print("v21 should be 0", v_21)
print("m21 should be 0??", m_21)