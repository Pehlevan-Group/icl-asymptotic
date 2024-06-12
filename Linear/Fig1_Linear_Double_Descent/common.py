import numpy as np

# STIELJIES TRANSFORM OF WISHART MATRICES
# COMPUTES m^* = M_k(x) NECESSARY FOR e^ICL, e^IDG DETERMINISTIC FORMULAS
def M_kappa(x, kappa):
    return 2 / ( (x + 1 - 1/kappa) + np.sqrt((x + 1 - 1/kappa)**2 + 4*x/kappa) )

# RIDGELESS FORMULA
# IMPLEMENTS RESULT 1
def e_ICL_theory(tau, alpha, rho, kappa):
    x_star = (1 + rho) / alpha
    m_star = M_kappa(x_star, kappa)
    mu_star = x_star * M_kappa(x_star, kappa/tau)

    if tau < 1:
        result = tau * (1 + x_star) / (1 - tau) * (1 - tau * (1 - mu_star)**2 - (x_star - rho) / x_star * mu_star ) - 2 * tau * (1 - mu_star) + rho + 1
    else:
        k_star = (1 - kappa * (m_star / (kappa + m_star))**2)**-1
        term1 = (k_star - 1/(tau - 1)) * (1 + x_star) * (x_star * m_star)**2
        term2 = (3 - 2 * tau) / (tau - 1) * m_star * (x_star)**2
        term3 = ((1 + rho) / (tau - 1) * m_star + 1) * x_star + rho * m_star / (tau - 1) + rho

        result = term1 + term2 + term3

    return result

def draw_pretraining_data(n, d, l, k, rho):
    x = np.random.randn(n, l+1, d) / np.sqrt(d)
    w_set = np.random.randn(k, d);
    norms = np.linalg.norm(w_set, axis=1)  # Calculate norms of each row
    scaling_factor = np.sqrt(d) / norms[:, np.newaxis]  # Compute scaling factor for normalization
    w_set = w_set * scaling_factor
    w_indices = np.random.randint(0, k, size=n)
    w = w_set[w_indices]
    epsilon = np.random.randn(n, l+1) * np.sqrt(rho)
    y = np.einsum('nij,nj->ni', x, w) + epsilon
    return x, y, w, w_set

def e_ICL_g_tr(Gamma_star, d, alpha, rho):
  #  Gamma_star = compute_Gamma_star(n, d, H_Z, y_l1, lambda_val)

    I_d = np.eye(d)
    zero_matrix = np.zeros((d, 1))
    A = np.block([
        [I_d],
        [zero_matrix.T]
    ])

    B = np.block([
        [((1 + rho) / alpha + 1) * I_d, zero_matrix],
        [zero_matrix.T, np.array([[1 + rho**2]])]
    ])

    term1 = 1 + rho
    term2 = -(2 / d) * np.trace(Gamma_star @ A)
    term3 = (1 / d) * np.trace(Gamma_star.T @ Gamma_star @ B)

    e_icl_g_value = term1 + term2 + term3
    return e_icl_g_value

def gen_err_analytical_NEW(Gamma, alpha, muhat, Rhat, rho):
    s = Gamma.shape; d = s[0]; ell = d*alpha;

    Atrain = np.zeros((d+1, d))
    Atrain[:d, :] = Rhat
    Atrain[d, :] = (1 + rho) * muhat.T

    Btrain = np.zeros((d+1, d+1))
    Btrain[:d, :d] = (1/alpha)*np.eye(d) + (1 + 1/ell)*Rhat/(1+rho)
    Btrain[d, :d] = (1 + 2/ell) * muhat.T
    Btrain[:d, d] = (1 + 2/ell) * muhat
    Btrain[d, d] = (1 + rho)*(1+2/ell)

    t1 = np.trace(np.matmul(Gamma, Atrain))
    t2 = np.trace(np.matmul(np.matmul(Gamma.T, Gamma), Btrain))

    value = 1 + rho + ((1+rho)/d)*t2 - (2/d)*t1
    return value

def monte_carlo_test_error(Gamma_star, d, l, n_test, rho):
    x_test = np.random.randn(n_test, l+1, d) / np.sqrt(d)
    w_test = np.random.randn(n_test, d)
    epsilon_test = np.random.randn(n_test, l+1) * np.sqrt(rho)
    y_test = np.einsum('nij,nj->ni', x_test, w_test) + epsilon_test

    # Construct test H_Z
    H_Z_test = construct_H_Z(x_test, y_test, l, d)

    # Predictions
    y_pred = np.einsum('nkl,kl->n', H_Z_test, Gamma_star)

    # Calculate mean squared error
    mse = np.mean((y_pred - y_test[:, l])**2)
    return mse


def Householder(beta, x):
    s = np.sign(beta[0])
    u = np.zeros_like(beta)
    u[0] = np.linalg.norm(beta) * s
    u += beta
    u /= np.linalg.norm(u)

    return -s * (x - 2 * (u.T @ x) * u)

def construct_H_NEW(beta, alpha, sigma_noise):
    d = len(beta)
    N = np.int64(alpha * d)
    theta_beta = np.linalg.norm(beta)/np.sqrt(d)

    theta_e = np.linalg.norm(np.random.randn(N)) * (sigma_noise / np.sqrt(d))

    a = np.random.randn(1)
    theta_q = np.linalg.norm(np.random.randn(N-1))/np.sqrt(d)

    # This is h in the notes
    v = np.zeros((d, 1))
    v[0] = theta_e * a / np.sqrt(d) + theta_beta * a**2 / d + theta_beta * theta_q**2
    v[1:] = np.sqrt(((theta_e + theta_beta *a /np.sqrt(d))**2 + theta_beta**2 * theta_q**2)/d) * np.random.randn(d-1, 1)
    # This is the (Mh).T part of the first dxd of H
    av = Householder(beta, v)
    # g = [s, u]
    g = np.random.randn(d, 1)
    s = g[0]
    # This is the (M[s, u]) part of the first dxd of H
    b = Householder(beta, g)
    # final term coming from sum(yy) in H
    yy = np.sqrt(1/d)*(theta_beta**2 * theta_q**2 + (theta_beta * a/np.sqrt(d) + theta_e)**2)
    new_av = np.append(av, yy)

    # Never used
    # mu1 = theta_beta**2 * a**2 / d + theta_beta**2 * theta_q**2
    # mu2 = ((theta_beta * a**2 / d + theta_beta * theta_q**2)**2 - mu1/d)/theta_beta**2

    # This is still correct
    y = theta_beta * s + sigma_noise * np.random.randn(1)
    return (d/N)*np.outer(b.flatten(), new_av), y

def construct_HHT_fast_NEW(beta, alpha, sigma_noise):
    n, d = beta.shape

    H = np.zeros((d*(d+1), n))
    y_ary = np.zeros((n, 1))

    for i in range(n):
        h, y_ary[i] = construct_H_NEW(beta[i,:].reshape(d, 1), alpha, sigma_noise)
        H[:, i] = h.reshape(-1)

    return H @ H.T, H @ y_ary

def learn_Gamma_fast_NEW(beta, alpha, sigma_noise, lam, tau_max):
    n, d = beta.shape

    n_max = np.int64(tau_max * d**2)

    idx = np.append(np.arange(0, n, n_max), n)

    M = np.zeros((d*(d+1), d*(d+1)))
    v = np.zeros((d*(d+1), 1))

    for i in range(len(idx)-1):
        H, y = construct_HHT_fast_NEW(beta[idx[i]:idx[i+1],:], alpha, sigma_noise)
        M += H
        v += y

    Gamma = np.linalg.solve(M + (n/d)*lam * np.eye(d*(d+1)), v)

    return Gamma.reshape(d, d+1)

def gen_err_analytical_NEW(Gamma, alpha, muhat, Rhat, rho):
    s = Gamma.shape; d = s[0]; ell = d*alpha;

    Atrain = np.zeros((d+1, d))
    Atrain[:d, :] = Rhat
    Atrain[d, :] = (1 + rho) * muhat.T

    Btrain = np.zeros((d+1, d+1))
    Btrain[:d, :d] = (1/alpha)*np.eye(d) + (1 + 1/ell)*Rhat/(1+rho)
    Btrain[d, :d] = (1 + 2/ell) * muhat.T
    Btrain[:d, d] = (1 + 2/ell) * muhat
    Btrain[d, d] = (1 + rho)*(1+2/ell)

    t1 = np.trace(np.matmul(Gamma, Atrain))
    t2 = np.trace(np.matmul(np.matmul(Gamma.T, Gamma), Btrain))

    value = 1 + rho + ((1+rho)/d)*t2 - (2/d)*t1
    return value

