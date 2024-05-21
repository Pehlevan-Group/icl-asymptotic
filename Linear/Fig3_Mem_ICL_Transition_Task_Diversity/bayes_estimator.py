import numpy as np

def S_W(c, alpha):
    return 2/(c+alpha-1 + np.sqrt((c+alpha-1)**2 + 4*c))

def bayes_estimator(d,Ks,N,sigma_beta,sigma_noise,nsim):
    IDG = np.zeros(2,len(Ks));
    ICL = np.zeros(2,len(Ks));

    for idx, K in enumerate(K):
        B = np.random.randn(d, K)
        for i in range(nsim):
            temp_IDG = np.zeros(2,nsim); temp_ICL = np.zeros(2,nsim);
            X = np.random.randn(d, N) / np.sqrt(d)
            beta_IDG = B[:, np.random.randint(K)].reshape(d, 1); beta_IDG = np.sqrt(d)*beta_IDG/np.linalg.norm(beta_IDG);
            beta_ICL = np.random.randn(d, 1) * sigma_beta

            y_IDG = X.T @ beta_IDG + np.random.randn(N, 1) * sigma_noise
            y_ICL = X.T @ beta_ICL + np.random.randn(N, 1) * sigma_noise

            # Bayesian estimator for the Gaussian distribution (ridge optimal estimator)
            w_ridge_ICL = np.linalg.solve(X @ X.T + sigma_noise**2/sigma_beta**2 * np.eye(d), X @ y_ICL);
            w_ridge_IDG = np.linalg.solve(X @ X.T + sigma_noise**2/sigma_beta**2 * np.eye(d), X @ y_IDG);

            # Bayesian estimator for the finite distribution 
            c = -np.linalg.norm(y_ICL - X.T @ B, axis=0)**2/(2*sigma_noise**2); ec = np.exp(c - np.max(c)); w_dmmse_ICL = B @ ec.reshape(K, 1) / np.sum(ec)
            c = -np.linalg.norm(y_IDG - X.T @ B, axis=0)**2/(2*sigma_noise**2); ec = np.exp(c - np.max(c)); w_dmmse_IDG = B @ ec.reshape(K, 1) / np.sum(ec)

            xv = np.random.randn(d, 1)/np.sqrt(d)
            yv = (xv.T @ beta_ICL).item() + np.random.randn() * sigma_noise
            temp_ICL[0,i] = ((xv.T @ w_ridge_ICL).item() - yv)**2; temp_ICL[1,i] = ((xv.T @ w_dmmse_ICL).item() - yv)**2; 
            yv = (xv.T @ beta_IDG).item() + np.random.randn() * sigma_noise
            temp_IDG[0,i] = ((xv.T @ w_ridge_IDG).item() - yv)**2; temp_IDG[1,i] = ((xv.T @ w_dmmse_IDG).item() - yv)**2; 
        
        ICL[0,idx] = np.mean(temp_ICL[0,:]); ICL[1,idx] = np.mean(temp_ICL[1,:]);
        IDG[0,idx] = np.mean(temp_IDG[0,:]); IDG[1,idx] = np.mean(temp_IDG[1,:])

