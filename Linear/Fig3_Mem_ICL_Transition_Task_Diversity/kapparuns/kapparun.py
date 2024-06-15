import numpy as np
import matplotlib.pyplot as plt
from helperfunctions import *
# Example parameters

d = 75  # Dimension
tau = 50
alpha = 50 # Example value
n = int(tau * (d**2))  # Number of samples
l = int(alpha * d)  # Context length
lambda_val = 0.01  # Regularization parameter
rho = 0.1  # Example value
n_test = 40000  # Number of test samples
n_MC = 6  # Number of Monte Carlo runs

# Define the range of kappa values
kappa_values = [0.1]#,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9]

# Initialize empty lists to store the e_ICL_g_tr and e_ICL_g_th values
e_icl_g_tr_values = []
e_icl_g_tr_std = []

# Calculate e_ICL_g_tr and e_ICL_g_th for each kappa value
for kappa in kappa_values:
    print(tau, alpha, kappa)
    k = int(kappa*d)  # Number of unique regression vectors

    # Calculate e_ICL_g_th
    e_icl_g_value_th = e_ICL_g_th(tau, alpha, rho, kappa)
    print("e_ICL_g_th =", e_icl_g_value_th)

    # average over n_MC Monte Carlo runs
    e_icl_g_values = []
    for i in range(n_MC):
        x, y, w = draw_pretraining_data(n, d, l, k, rho)
        H_Z = construct_H_Z(x, y, l, d)
        y_l1 = y[:, l]
        Gamma_star = compute_Gamma_star(n, d, H_Z, y_l1, lambda_val)
        e_icl = e_ICL_g_tr(Gamma_star, d, alpha, rho)
        e_icl_g_values.append(e_icl)

    e_icl_g_tr_values.append(np.mean(e_icl_g_values))
    e_icl_g_tr_std.append(np.std(e_icl_g_values))
    print("e_ICL_g_tr =", np.mean(e_icl_g_values))
    print("e_ICL_g_tr std =", np.std(e_icl_g_values))



# Plot a fine grained theory curve
kappa_values_th = np.linspace(0.01, 2, 1000)
e_icl_g_th_values = []
for kappa in kappa_values_th:
    e_icl_g_value_th = e_ICL_g_th(tau, alpha, rho, kappa)
    e_icl_g_th_values.append(e_icl_g_value_th)


# Plot e_ICL_g_tr and e_ICL_g_th as a function of kappa
#plt.plot(kappa_values, e_icl_g_tr_values, label='e_ICL_g_tr(Γ*)',marker='o',linestyle='None')
plt.errorbar(kappa_values, e_icl_g_tr_values, yerr=e_icl_g_tr_std, fmt='o')
plt.plot(kappa_values_th, e_icl_g_th_values, label='e_ICL_g_th',linestyle='dashed')
plt.plot(kappa_values_th, 1+rho-rho*(tau/(tau-1))*alpha/(1+rho), label='k=0 limit',linestyle='dashed')
plt.xlabel('kappa')
plt.ylabel('e_ICL_g')
plt.legend()
plt.show()