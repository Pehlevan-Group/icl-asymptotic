import numpy as np
import sys
from common import *

# Example parameters

d = sys.argv[1]
alpha = sys.argv[2]

d = int(d)
alpha = float(alpha)
l = int(alpha * d)  # Context length

lambda_val = 0.00000001
kappa = 0.5; k = np.int64(kappa * d)
rho = 0.01  # Example value
sigma = 0.1
n_test = 40000  # Number of test samples
n_MC = 10  # Number of Monte Carlo runs

tau_values = [0.2, 0.5, 0.85]

# Initialize empty lists to store the e_ICL_g_tr and e_ICL_g_th values
e_icl_g_tr_values = []
e_icl_g_tr_std = []
idg_save = []; idg_std = []

print("d is ",d)
print("alpha is", alpha)
print("kappa is", kappa)
print("lambda", lambda_val)
# Calculate e_ICL_g_tr and e_ICL_g_th for each kappa value
for tau in tau_values:
    print(tau, alpha, kappa)
    n = int(tau * (d**2))  # Number of samples

    # average over n_MC Monte Carlo runs
    e_icl_g_values = []
    e_idg_g_values = []
    for i in range(n_MC):
        print("iteration ",i)
        x, y, w, B = draw_pretraining_data(n, d, l, k, rho)
        H_Z = construct_H_NEW(x, y, l, d)
        y_l1 = y[:, l]
        Gamma_star = (n, d, H_Z, y_l1, lambda_val)
        e_icl_g_values.append(gen_err_analytical_NEW(Gamma_star, alpha, np.zeros(d), np.eye(d), rho))
        print("iteration ",i, " icl vals ", e_icl_g_values)
        e_idg_g_values.append(gen_err_analytical_NEW(Gamma_star, alpha, np.mean(B,axis=0), (B.T @ B)/k, rho))
        print("iteration ",i, " idg vals ", e_idg_g_values)

    e_icl_g_tr_values.append(np.mean(e_icl_g_values))
    e_icl_g_tr_std.append(np.std(e_icl_g_values))
    idg_save.append(np.mean(e_idg_g_values))
    idg_std.append(np.std(e_idg_g_values))

print("iclmean",e_icl_g_tr_values)
print("iclstd",e_icl_g_tr_std)
print("idgmean",idg_save)
print("idgstd",idg_std)