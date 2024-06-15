import numpy as np
import matplotlib.pyplot as plt

def M_kappa(x, kappa):
    return 2 / ( (x + 1 - 1/kappa) + np.sqrt((x + 1 - 1/kappa)**2 + 4*x/kappa) )

def e_ICL_g_th(tau, alpha, rho, kappa):
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
    return x, y, w

def construct_H_Z(x, y, l, d):
    y_sum_x = np.einsum('nij,ni->nj', x[:, :l], y[:, :l])
    y_sum_y = np.sum(y[:, :l]**2, axis=1)
    H_Z = np.zeros((x.shape[0], d, d+1))
    H_Z[:, :, :d] = x[:, l, :, None] * (d / l) * y_sum_x[:, None, :]
    H_Z[:, :, d] = x[:, l] * (1 / l) * y_sum_y[:, None]
    return H_Z

def compute_Gamma_star(n, d, H_Z, y_l1, lambda_val):
    H_Z_vec = H_Z.reshape(n, -1)
    #regularization_term = (n / d) * lambda_val * np.eye(H_Z_vec.shape[1])
    # Compute sum of outer products using matrix multiplication
    sum_term = H_Z_vec.T @ H_Z_vec
    # Compute y_l1 weighted sum using broadcasting
    weighted_sum = H_Z_vec.T @ y_l1
    Gamma_star_vec = np.linalg.pinv(sum_term) @ weighted_sum
    return Gamma_star_vec.reshape(d, d+1)

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
    term2 = - (2 / d) * np.trace(Gamma_star @ A)
    term3 = (1 / d) * np.trace(Gamma_star.T @ Gamma_star @ B)

    e_icl_g_value = term1 + term2 + term3
    return e_icl_g_value

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
