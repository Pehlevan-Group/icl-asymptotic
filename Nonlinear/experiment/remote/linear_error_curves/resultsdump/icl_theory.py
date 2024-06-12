import numpy as np

def M_kappa(x, kappa):
    return 2 / ( (x + 1 - 1/kappa) + np.sqrt((x + 1 - 1/kappa)**2 + 4*x/kappa) )

def icl_theory(tau, alpha, rho, kappa):
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