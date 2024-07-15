import numpy as np
import sys
from common import *

d = sys.argv[1]
alpha = sys.argv[2]
d = int(d)
alpha = float(alpha)

sigma_noise = 0.1; sigma_beta = 1; rho = (sigma_noise/sigma_beta)**2
lam = 0.00000001
kappa = 0.5; K = np.int64(kappa * d)

numavg = 10;
tau_values = [0.2, 0.5, 0.85] #[1.2,1.5,2]
icl_sim_ary = []; idg_sim_ary = []

print("d ", d)
print("alpha", alpha)
print("kappa",kappa)
print("lambda",lam)
for tau in tau_values:
  rp = np.int64(np.round(tau * d**2 / K))
  n = rp * K
  icl_dummy = []; idg_dummy = [];
  for dummy in range(numavg):
    print("iteration ",dummy)
    B = np.random.randn(K, d)*sigma_beta;
    norms = np.linalg.norm(B, axis=1)
    B = B / norms[:, np.newaxis] * np.sqrt(d)

    beta = np.repeat(B[np.newaxis, :, :], rp, axis=0).reshape(n, d)
    tau_max = 3
    Gamma = learn_Gamma_fast_NEW(beta, alpha, sigma_noise, lam, tau_max)
    icl_dummy.append(gen_err_analytical_NEW(Gamma, alpha, np.zeros(d), sigma_beta**2 * np.eye(d), (sigma_noise/sigma_beta)**2))
    print("iteration ",dummy," icl ",icl_dummy)
    idg_dummy.append(gen_err_analytical_NEW(Gamma, alpha, np.mean(B,axis=0), (B.T @ B)/K, (sigma_noise/sigma_beta)**2))
    print("iteration ",dummy," idg ",idg_dummy)

  icl_sim_ary.append(np.array(icl_dummy));
  idg_sim_ary.append(np.array(idg_dummy));

icl_sim_ary = np.array(icl_sim_ary)
idg_sim_ary = np.array(idg_sim_ary)
print("iclmean",np.mean(icl_sim_ary,axis=1))
print("iclstd",np.std(icl_sim_ary,axis=1))
print("idgmean",np.mean(idg_sim_ary,axis=1))
print("idgstd",np.std(idg_sim_ary,axis=1))