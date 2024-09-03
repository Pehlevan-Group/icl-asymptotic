import numpy as np
from tqdm import tqdm 
import sys

def S_W(c, alpha):
    return 2/(c+alpha-1 + np.sqrt((c+alpha-1)**2 + 4*c))

d = 100
directory = sys.argv[1]
kappa = float(sys.argv[2]) 
alphaind = int(sys.argv[3]) - 1
sigma_beta = 1
sigma_noise = np.sqrt(0.01)

alphas = np.linspace(0.1,2,500);
alpha = alphas[alphaind]
N = np.int64(alpha * d)

nsim = 500000

e_B_full_ary = np.zeros(nsim)
e_B_finite_ary = np.zeros(nsim)
# K_ary = np.int64(np.logspace(1, 4, 3))
# K_ary = np.int64(d * np.logspace(-1, 2, 5))
# K_ary = np.int64(np.logspace(np.log10(2), np.log10(100*d), 6))
#kappa_ary = np.logspace(np.log10(0.01),np.log10(250),50); K = np.int64(kappa_ary*d);
#K_array = np.int64(np.array([0.01,0.03,0.06,0.11,0.2,0.37,0.69,1.27,2.33,4.28,7,84,14.38,26.36,48.32,88.58,162.37,297.63,545.55,1000])*d)
# K_array = list(np.int64(np.logspace(np.log10(0.05*d),np.log10(500*d),40))); #list(np.int64(np.logspace(np.log10(0.05*d),np.log10(500*d),40))); 
# K_array = [i for n, i in enumerate(K_array) if i not in K_array[:n]]
# K_array = np.array(K_array)
# K = K_array[ind]

K = int(kappa*d);
print(f'kappa is {kappa}')
B = np.random.randn(d, K)

# IsFinite = True
# for i in range(nsim):
#     X = np.random.randn(d, N) / np.sqrt(d)
#     if IsFinite:
#         beta = B[:, np.random.randint(K)].reshape(d, 1)
#     else:
#         beta = np.random.randn(d, 1) * sigma_beta

#     y = X.T @ beta + np.random.randn(N, 1) * sigma_noise

#     # Bayesian estimator for the Gaussian distribution
#     beta_hat = np.linalg.solve(X @ X.T + sigma_noise**2/sigma_beta**2 * np.eye(d), X @ y)

#     # Bayesian estimator for the finite distribution
#     c = -np.linalg.norm(y - X.T @ B, axis=0)**2/(2*sigma_noise**2)
#     ec = np.exp(c - np.max(c))
#     beta_hat_finite = B @ ec.reshape(K, 1) / np.sum(ec)

#     xv = np.random.randn(d, 1)/np.sqrt(d)
#     yv = (xv.T @ beta).item() + np.random.randn() * sigma_noise

#     e_B_full_ary[i] = ((xv.T @ beta_hat).item() - yv)**2
#     e_B_finite_ary[i] = ((xv.T @ beta_hat_finite).item() - yv)**2

# filename = f'{directory}/idg_dmmse_m.txt'
# with open(filename, 'a') as file:
#     file.write(f'{ind}, {np.mean(e_B_finite_ary)}\n')
# filename = f'{directory}/idg_ridge_m.txt'
# with open(filename, 'a') as file:
#     file.write(f'{ind}, {np.mean(e_B_full_ary)}\n')
# filename = f'{directory}/idg_dmmse_s.txt'
# with open(filename, 'a') as file:
#     file.write(f'{ind}, {np.std(e_B_finite_ary)}\n')
# filename = f'{directory}/idg_ridge_s.txt'
# with open(filename, 'a') as file:
#     file.write(f'{ind}, {np.std(e_B_full_ary)}\n')

IsFinite = False
e_B_full_ary = np.zeros(nsim)
e_B_finite_ary = np.zeros(nsim)
for i in range(nsim):
    X = np.random.randn(d, N) / np.sqrt(d)
    if IsFinite:
        beta = B[:, np.random.randint(K)].reshape(d, 1)
    else:
        beta = np.random.randn(d, 1) * sigma_beta

    y = X.T @ beta + np.random.randn(N, 1) * sigma_noise

    # Bayesian estimator for the Gaussian distribution
    beta_hat = np.linalg.solve(X @ X.T + sigma_noise**2/sigma_beta**2 * np.eye(d), X @ y)

    # Bayesian estimator for the finite distribution
    c = -np.linalg.norm(y - X.T @ B, axis=0)**2/(2*sigma_noise**2)
    ec = np.exp(c - np.max(c))
    beta_hat_finite = B @ ec.reshape(K, 1) / np.sum(ec)

    xv = np.random.randn(d, 1)/np.sqrt(d)
    yv = (xv.T @ beta).item() + np.random.randn() * sigma_noise

    e_B_full_ary[i] = ((xv.T @ beta_hat).item() - yv)**2
    e_B_finite_ary[i] = ((xv.T @ beta_hat_finite).item() - yv)**2

ind = alphaind
filename = f'{directory}/icl_dmmse_m.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.mean(e_B_finite_ary)}\n')
filename = f'{directory}/icl_ridge_m.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.mean(e_B_full_ary)}\n')
filename = f'{directory}/icl_dmmse_s.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.std(e_B_finite_ary)}\n')
filename = f'{directory}/icl_ridge_s.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.std(e_B_full_ary)}\n')
