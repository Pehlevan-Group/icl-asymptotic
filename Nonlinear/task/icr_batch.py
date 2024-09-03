from regression import *
import numpy as np
import random

samples = 1000
d = 100;
tau = 1; P = np.int64(tau*d**2)

kapparange = np.logspace(np.log10(0.01),np.log10(100),50)
rng = np.random.default_rng()
icrs = np.zeros(len(kapparange),samples)

for kind, kappa in enumerate(kapparange):
    for i in range(samples):
        K = np.int(kappa*d); 
        E = rng.normal(loc=0, scale = 1, size=(d, K))
        uniform_ps = np.array([random.randrange(K) for _ in range(P)])
        ws = np.array([E[:,uniform_ps[i]] for i in range(len(uniform_ps))]) #shape P x d
        H = ws@ws.T # P x P hessian matrix
        eigenvalues, _ = np.linalg.eig(H); lmax = np.max(eigenvalues); lmin = np.min(eigenvalues)
        khat = np.mean(eigenvalues)/lmin;
        k = lmax/lmin;
        icrs[kind,i] = khat/np.sqrt(k);




