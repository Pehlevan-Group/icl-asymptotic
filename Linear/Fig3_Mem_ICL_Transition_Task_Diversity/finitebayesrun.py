import numpy as np
from tqdm import tqdm 
import sys

def S_W(c, alpha):
    return 2/(c+alpha-1 + np.sqrt((c+alpha-1)**2 + 4*c))

d = 100
alpha = float(sys.argv[1])
sigma_beta = 1
sigma_noise = 0.1

N = np.int64(alpha * d)

nsim = 50000

e_B_full_ary = np.zeros(nsim)
e_B_finite_ary = np.zeros(nsim)
# K_ary = np.int64(np.logspace(1, 4, 3))
# K_ary = np.int64(d * np.logspace(-1, 2, 5))
# K_ary = np.int64(np.logspace(np.log10(2), np.log10(100*d), 6))
#kappa_ary = np.logspace(np.log10(0.01),np.log10(250),50); K = np.int64(kappa_ary*d);
K_array = list(np.int64(np.logspace(np.log10(0.05*d),np.log10(50*d),20))); 
K_array = [i for n, i in enumerate(K_array) if i not in K_array[:n]]
K_array = np.array(K_array)
e_finite_K_ary = np.zeros(len(K_array))
e_normal_K_ary = np.zeros(len(K_array))
s_finite_K_ary = np.zeros(len(K_array))
s_normal_K_ary = np.zeros(len(K_array))
kappa_ary = K_array/d;

print(kappa_ary)

IsFinite = True

for idx, K in tqdm(enumerate(K_array)):
    B = np.random.randn(d, K)

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

    e_finite_K_ary[idx] = np.mean(e_B_finite_ary)
    e_normal_K_ary[idx] = np.mean(e_B_full_ary)
    s_finite_K_ary[idx] = np.std(e_B_finite_ary)
    s_normal_K_ary[idx] = np.std(e_B_full_ary)

print("IDG dmmse mean", list(e_finite_K_ary))
print("IDG ridge mean", list(e_normal_K_ary))
print("IDG dmmse std", list(s_finite_K_ary))
print("IDG ridge std", list(s_normal_K_ary))

IsFinite = False
e_B_full_ary = np.zeros(nsim)
e_B_finite_ary = np.zeros(nsim)
e_finite_K_ary = np.zeros(len(K_array))
e_normal_K_ary = np.zeros(len(K_array))
s_finite_K_ary = np.zeros(len(K_array))
s_normal_K_ary = np.zeros(len(K_array))

for idx, K in tqdm(enumerate(K_array)):
    B = np.random.randn(d, K)

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

    e_finite_K_ary[idx] = np.mean(e_B_finite_ary)
    e_normal_K_ary[idx] = np.mean(e_B_full_ary)
    s_finite_K_ary[idx] = np.std(e_B_finite_ary)
    s_normal_K_ary[idx] = np.std(e_B_full_ary)

print("ICL dmmse mean", list(e_finite_K_ary))
print("ICL ridge mean", list(e_normal_K_ary))
print("ICL dmmse std", list(s_finite_K_ary))
print("ICL ridge std", list(s_normal_K_ary))