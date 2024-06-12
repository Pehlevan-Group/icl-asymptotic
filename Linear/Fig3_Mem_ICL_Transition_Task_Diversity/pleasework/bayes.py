import numpy as np

d = 100
alpha = 1
sigma_beta = 1
sigma_noise = 0.5

N = np.int64(alpha * d)

nsim = 10000

ridge_ary = np.zeros(nsim)
dmmse_ary = np.zeros(nsim)

K_ary = np.int64(np.logspace(np.log10(100*d), np.log10(10000*d), 5))
dmmse_ary_answer = np.zeros(len(K_ary))
ridge_ary_answer = np.zeros(len(K_ary))

print(K_ary)

IsFinite = True

for idx, K in enumerate(K_ary):
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
        beta_hat_dmmse = B @ ec.reshape(K, 1) / np.sum(ec)

        xv = np.random.randn(d, 1)/np.sqrt(d)
        yv = (xv.T @ beta).item() + np.random.randn() * sigma_noise

        ridge_ary[i] = ((xv.T @ beta_hat).item() - yv)**2
        dmmse_ary[i] = ((xv.T @ beta_hat_dmmse).item() - yv)**2

    dmmse_ary_answer[idx] = np.mean(dmmse_ary)
    ridge_ary_answer[idx] = np.mean(ridge_ary)


print(dmmse_ary_answer,ridge_ary_answer)

#print(dmmse_ary_answer, sigma_noise**2, ridge_ary_answer, sigma_noise**2 * (1 + S_W(sigma_noise**2/sigma_beta**2, alpha)))