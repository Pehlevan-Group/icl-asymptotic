import matplotlib.pyplot as plt
import numpy as np
import sys
import tqdm
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainmini import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegressionCorrect, FiniteSampler


d = 20;
alpha = 5; N = int(alpha*d);
tau = 50; P = int(tau*d*d);
sigma = 0.1;

def finitebayes(d,N,P,K,sigma,psi=1):
    mydata = FiniteSampler(N+1,d,sigma,psi,K,P)
    myvectors = mydata.E;
    xs,ys = next(mydata);
    allys = xs[:,:,-1];

    numerator = [];
    for k in range(K):
        yj = xs[:,:,0:-1] @ myvectors[:,k]; 
        numerator.append(np.sum(np.exp(-np.sum((allys - yj)*(allys - yj),axis = 1)/(2*sigma**2)))/P);
    numerator = np.array(numerator); 
    estimator = myvectors @ numerator; estimator = estimator/sum(numerator);
    return estimator;

kappalist = np.exp(np.linspace(np.log(0.1),np.log(50),25));
diverseobject = LinearRegressionCorrect(N+1,d,sigma,1,1,P);
errors = np.zeros(len(kappalist));
numsamples = 10;
for _ in range(numsamples):
    xsdiverse, ysdiverse = next(diverseobject)
    for k in tqdm(range(len(kappalist))):
        kappa = kappalist[k]; K = int(kappa*d);
        estimator = finitebayes(d,N,P,K,sigma,1);
        errors[k] = errors[k] + np.sum((xsdiverse[:,-1,0:-1] @ estimator - ysdiverse)*(xsdiverse[:,-1,0:-1] @ estimator - ysdiverse))/P;

errors = errors/numsamples;
print(errors)
