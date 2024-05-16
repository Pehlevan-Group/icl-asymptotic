# We provide a few commonly used functions here.

import numpy as np
from scipy.optimize import root_scalar
from scipy.linalg import sqrtm

# We provide a few commonly used functions here.

def forward(Gamma, X_ary, y_ary):
    """
    This function implements the forward (i.e. inference) part of the linear model.

    Parameters:
    Gamma: a d-by-d weight matrix
    X_ary: an n-by-d-by-(N+1) tensor, where n is the number of samples, d is the dimension of feature vectors, and N is the number of features used in the context
    y_ary: an n-by-(N+1) matrix, where n is the number of samples, and N is the number of features used in the context

    Returns:
    y_pred: an n-by-1 column vector
    """

    if X_ary.ndim == 2:
        X_ary = X_ary[np.newaxis, :, :]
    
    if y_ary.ndim == 1:
        y_ary = y_ary[np.newaxis, :]
    
    n = X_ary.shape[0]
    
    return (X_ary[:, :, -1].reshape(n, 1, -1) @ (Gamma @ (X_ary[:,:,:-1] @ y_ary[:,:-1].reshape(n, -1, 1)))).flatten()

def est_gen_err(Gamma, X_ary, y_ary):
    """
    This function estimates the generalization error of the linear model.

    Parameters:
    Gamma: a d-by-d weight matrix
    X_ary: an n-by-d-by-(N+1) tensor, where n is the number of test samples, d is the dimension of feature vectors, and N is the number of features used in the context
    y_ary: an n-by-(N+1) matrix, where n is the number of test samples, and N is the number of features used in the context

    Returns:
    gen_err: the generalization error (averaged squared loss)
    """

    if X_ary.ndim == 2:
        X_ary = X_ary[np.newaxis, :, :]
    
    if y_ary.ndim == 1:
        y_ary = y_ary[np.newaxis, :]
    
    n = X_ary.shape[0]

    # get the target vector
    y_target = y_ary[:, -1]

    # get the prediction vector
    y_pred = forward(Gamma, X_ary, y_ary)

    # print(y_pred.shape)
    # print(y_target.shape)
    
    return np.linalg.norm(y_target - y_pred)**2 / n #, y_target, y_pred

def learn_Gamma(X_ary, y_ary, lam):
    """
    This function learns the Gamma matrix from the training data.

    Parameters:
    X: an n-by-d-by-(N+1) tensor, where n is the number of samples, d is the dimension of feature vectors, and N is the number of features used in the context
    y: an n-by-(N+1) matrix, where n is the number of samples, and N is the number of features used in the context
    lam: the regularization parameter

    Returns:
    Gamma: a d-by-d weight matrix
    """
    
    shape = X_ary.shape
    n = shape[0]
    d = shape[1]
    
    # y_target is a n-by-1 column vector
    y_target = y_ary[:, -1:]

    # X_vec is a n-by-d-by-1 tensor
    X_vec = X_ary[:, :, -1:]
    # y_vec is a n-by-N-by-1 tensor
    y_vec = y_ary[:, :-1].reshape(n, -1, 1)

    # H is the regression matrix of size d**2-by-n
    H = (X_vec @ (X_ary[:, :,:-1] @ y_vec).reshape(n, 1, -1)).reshape(n, -1).T

    # Ridge regression
    Gamma = np.linalg.solve(H @ H.T / d + lam * np.eye(d**2), H @ y_target / d).reshape(d, d)

    return Gamma

def generate_beta_normal(n, d, sc):
    """
    This function generates a collection of independent beta vectors from the normal distribution.

    Parameters:
    n: the number of contexts
    d: the dimension of feature vectors
    sc: scaling parameter
        if sc is a scalar, then beta is sampled from N(0, sc^2 I)
        if sc is a d-by-d matrix, then beta is sampled from N(0, sc @ sc.T)

    Returns:
    beta: a n-by-d matrix
    """

    if np.isscalar(sc):
        return np.random.normal(0, sc, (n, d))
    else:
        return np.random.randn(n, d) @ sc.T
    
def generate_X_y(beta, sc_X, sigma_noise, N):
    """
    This function generates a collection of contexts and responses.

    Parameters:
    beta: a n-by-d matrix of regression vectors
    sc_X: scaling parameter for contexts
        if sc_X is a scalar, then X is sampled from N(0, sc_X^2 I/d)
        if sc_X is a d-by-d matrix, then X is sampled from N(0, sc_X @ sc_X.T/d)
    sigma_noise: standard deviation of the (Gaussian) noise in each response
    N: the number of features used in the context

    Returns:
    X: an n-by-d-by-(N+1) tensor
    y: an n-by-(N+1) matrix
    """

    n, d = beta.shape

    if np.isscalar(sc_X):
        X = np.random.normal(0, sc_X/np.sqrt(d), (n, d, N+1))
    else:
        X = (sc_X / np.sqrt(d)) @ np.random.randn(n, d, N+1)

    # y is a n-by-(N+1) matrix
    y = (beta[:, np.newaxis, :] @ X)[:, 0, :]  + np.random.normal(0, sigma_noise, (n, N+1))

    return X, y

def make_H_Z(X_ary, y_ary):
    """
    This function generates the H and Z matrices from the training data.

    Parameters:
    X_ary: an n-by-d-by-(N+1) tensor, where n is the number of samples, d is the dimension of feature vectors, and N is the number of features used in the context
    y_ary: an n-by-(N+1) matrix, where n is the number of samples, and N is the number of features used in the context

    Returns:
    H: a d^2-by-n regression matrix
    Z: a (d^2+1)-by-n extended regression matrix
    """

    shape = X_ary.shape
    n = shape[0]
    d = shape[1]
    
    # y_target is a n-by-1 column vector
    y_target = y_ary[:, -1:]

    # X_vec is a n-by-d-by-1 tensor
    X_vec = X_ary[:, :, -1:]
    # y_vec is a n-by-N-by-1 tensor
    y_vec = y_ary[:, :-1].reshape(n, -1, 1)

    # H is the regression matrix of size d**2-by-n
    H = (X_vec @ (X_ary[:, :,:-1] @ y_vec).reshape(n, 1, -1)).reshape(n, -1).T

    # Z is the extended regression matrix of size (d**2+1)-by-n
    Z = np.vstack((y_target.T/d, H/np.sqrt(d)))

    return H, Z

def gen_err_analytical(Gamma, C, R, sigma_noise, alpha):
    """
    This function computes the analytical generalization error of the linear model.

    Parameters:
    Gamma: a d-by-d weight matrix
    C: a d-by-d covariance matrix of the regression vectors
    R: a d-by-d covariance matrix of the beta vectors
    sigma_noise: standard deviation of the (Gaussian) noise in each response
    alpha: N/d, where N is the number of features used in the context

    Returns:
    gen_err: the analytical generalization error (averaged squared loss)
    """

    d = Gamma.shape[0]

    return alpha * (np.trace(C @ R)/d + sigma_noise**2) * np.trace(C @ Gamma @ C @ Gamma.T)/d \
                + (alpha**2 + alpha/d)*np.trace(C @ Gamma @ C @ R @ C @ Gamma.T)/d \
                - 2 * alpha * np.trace(C @ Gamma @ C @ R)/d \
                + np.trace(C @ R)/d \
                + sigma_noise**2
    
def diverse_isotropic(alpha, tau, lam, sigma_beta, sigma_noise):
    """
    This function computes the asymptotic limits of several key parameters in the special case where the covariance matrices of the regression vectors and the beta vectors are both isotropic.

    Parameters:
    alpha: N/d, where N is the number of features used in the context
    tau: n/d^2, where n is the number of samples
    lam: the regularization parameter
    sigma_beta: standard deviation of the beta vectors
    sigma_noise: standard deviation of the (Gaussian) noise in each response

    Returns:
    chi, lam_tilde, c_star_inv, trace_Gamma, Gamma_F_norm, trace_G, e_g
    """

    lam_tilde = lam / alpha / (sigma_noise**2 + sigma_beta**2*(1+alpha))
    chi = 2 / (np.sqrt((tau-1+lam_tilde)**2 + 4 * lam_tilde) + tau - 1 + lam_tilde)

    sigma_bar = sigma_noise**2/sigma_beta**2

    c_star_inv = lam + sigma_beta**2 /(1+alpha+sigma_bar) * (tau * ((1+sigma_bar)**2 + alpha * sigma_bar) \
                                        + ((1+sigma_bar)**2 + alpha * sigma_bar - lam_tilde*alpha)*(lam_tilde *chi - 1))
    
    trace_Gamma = tau * chi / (1+chi)/(1+alpha+sigma_bar)

    trace_G = 1/(tau * alpha * sigma_beta**2/(1+chi)*(1+sigma_bar + alpha) + lam)

    Gamma_F_norm = tau * (sigma_beta**2 + sigma_noise**2) * chi / (lam * (1+chi)**2 + tau * alpha * (sigma_noise**2 + sigma_beta**2*(1+alpha))) \
            * (1 + tau * alpha * (1-chi) / (1+ alpha + sigma_bar)/(1+sigma_bar)/(1+chi))
    
    e_g = (sigma_beta**2*alpha*(1+alpha) + alpha * sigma_noise**2) * Gamma_F_norm - 2 * alpha * sigma_beta**2 * trace_Gamma + sigma_beta**2 + sigma_noise**2

    return chi, lam_tilde, c_star_inv, trace_Gamma, Gamma_F_norm, trace_G, e_g

# Some functions specific to the case where C = I and R = Wihart

def S_Wishart(x, sigma_beta, nu):
    """
    This funciton computes the Stieltjes transform of the Wishart distribution.
    """
    return 2/sigma_beta**2 / ((x/sigma_beta**2 + 1 - 1/nu) + np.sqrt((x/sigma_beta**2 + 1 - 1/nu)**2 + 4 * x/sigma_beta**2/nu))

def S_Wishart_derivative(x, sigma_beta, nu):
    """
    This funciton computes the derivative of the Stieltjes transform of the Wishart distribution.
    """

    s = S_Wishart(x, sigma_beta, nu)
    return -1/2 * s**2 * (1 + (x/sigma_beta**2 + 1 + 1/nu)/np.sqrt((x/sigma_beta**2 + 1 - 1/nu)**2 + 4 * x/sigma_beta**2/nu))


def diverse_Wishart_perturbation(alpha, tau, lam, sigma_beta, sigma_noise, nu, omega, K):
    """
    This function computes the asymptotic limits of several key parameters in the special case where the covariance matrices of the regression vectors and the beta vectors are both isotropic.

    Parameters:
    alpha: N/d, where N is the number of features used in the context
    tau: n/d^2, where n is the number of samples
    lam: the regularization parameter
    sigma_beta: standard deviation of the beta vectors
    sigma_noise: standard deviation of the (Gaussian) noise in each response
    nu: the degree of freedom of the Wishart distribution
    omega: the scalar parameter for the perturbation
    K: the type of perturbation matrix. Only two inputs are supported: 'I' and 'Wishart'

    Returns:
    xi, chi, trace_GammaR
    """

    mu1 = alpha * (sigma_noise**2 + sigma_beta**2)
    mu2 = alpha**2

    if K == 'I':
        f = lambda xi: (lam+omega) * xi**2/mu2 * S_Wishart(((lam+omega)*xi + mu1)/mu2, sigma_beta, nu) + (tau-1) * xi -1
    elif K == 'Wishart':
        f = lambda xi: mu2*xi/(mu2 + omega*xi) * ((lam * xi + mu1)/(mu2 + omega*xi) - mu1/mu2) * S_Wishart((lam*xi + mu1)/(mu2+omega*xi), sigma_beta, nu) \
            + tau * xi - 1 - mu2 * xi / (mu2 + omega * xi)
    else:
        raise ValueError("The input K is not supported.")

    lb = 1
    while f(lb) > -1e-5:
        lb /= 2
    ub = lb + 1
    while f(ub) < 1e-5:
        ub += 1

    sol = root_scalar(f, bracket=[lb, ub], method='brentq')
    assert sol.converged, "Failed to find a solution to the equation."
    
    xi = sol.root
    chi = tau * xi - 1

    c0 = (mu1 + lam * xi)/mu2

    trace_GammaR = 1/alpha * (sigma_beta**2 - c0 + c0**2 * S_Wishart(c0, sigma_beta, nu))


    if K == 'I':
        c0 = (lam * xi + omega * xi + mu1)/mu2
        cinv = (sigma_beta**2 + sigma_noise**2)/xi + lam - alpha**2 / xi / mu2 * (sigma_beta**2 - c0 + c0**2 * S_Wishart(c0, sigma_beta, nu))
    else:
        c0 = (lam * xi + mu1)/(mu2 + omega * xi)
        cinv = (sigma_beta**2 + sigma_noise**2)/xi + lam - alpha**2 / xi / (mu2 + omega * xi) * (sigma_beta**2 - c0 + c0**2 * S_Wishart(c0, sigma_beta, nu))

        
    return xi, chi, trace_GammaR, cinv

def diverse_Wishart(alpha, tau, lam, sigma_beta, sigma_noise, nu):
    """
    This function computes the asymptotic limits of several key parameters in the special case where the covariance matrices of the regression vectors and the beta vectors are both isotropic.

    Parameters:
    alpha: N/d, where N is the number of features used in the context
    tau: n/d^2, where n is the number of samples
    lam: the regularization parameter
    sigma_beta: standard deviation of the beta vectors
    sigma_noise: standard deviation of the (Gaussian) noise in each response
    nu: the degree of freedom of the Wishart distribution
    omega: the scalar parameter for the perturbation
    K: the type of perturbation matrix. Only two inputs are supported: 'I' and 'Wishart'

    Returns:
    xi, chi, trace_GammaR, Gamma_F_norm, trace_GammaRGamma, e_g
    """

    mu1 = alpha * (sigma_noise**2 + sigma_beta**2)
    mu2 = alpha**2
    f = lambda xi: lam * xi**2/mu2 * S_Wishart((lam*xi + mu1)/mu2, sigma_beta, nu) + (tau-1) * xi -1

    lb = 1
    while f(lb) > -1e-5:
        lb /= 2
    ub = lb + 1
    while f(ub) < 1e-5:
        ub += 1

    sol = root_scalar(f, bracket=[lb, ub], method='brentq')
    assert sol.converged, "Failed to find a solution to the equation."
    
    xi = sol.root
    chi = tau * xi - 1

    c0 = (mu1 + lam * xi)/mu2

    trace_GammaR = 1/alpha * (sigma_beta**2 - c0 + c0**2 * S_Wishart(c0, sigma_beta, nu))

    *_, cinv = diverse_Wishart_perturbation(alpha, tau, lam, sigma_beta, sigma_noise, nu, 0, 'I')

    eps = 1e-5
    *_, cinv_pert_I = diverse_Wishart_perturbation(alpha, tau, lam, sigma_beta, sigma_noise, nu, eps, 'I')

    Gamma_F_norm = (cinv_pert_I - cinv)/eps

    *_, cinv_pert = diverse_Wishart_perturbation(alpha, tau, lam, sigma_beta, sigma_noise, nu, eps, 'Wishart')
    trace_GammaRGamma = (cinv_pert - cinv)/eps

    e_g = alpha * (sigma_beta**2 + sigma_noise**2) * Gamma_F_norm + alpha**2 * trace_GammaRGamma - 2 * alpha * trace_GammaR + sigma_beta**2 + sigma_noise**2

    return xi, chi, trace_GammaR, Gamma_F_norm, trace_GammaRGamma, e_g




def finite_isotropic_cinv(alpha, tau, lam, sigma_beta, sigma_noise, pw, d):
    """
    This function computes 1/c in the finite task case with isotropic regression vectors

    Parameters:
    alpha: N/d, where N is the number of features used in the context
    tau: n/d^2, where n is the number of samples
    lam: the regularization parameter
    sigma_beta: standard deviation of the beta vectors
    sigma_noise: standard deviation of the (Gaussian) noise in each response
    pw: the probability distribution of the finite tasks

    Returns:
    c_inv
    """

    K = len(pw)
    mu1 = alpha * (sigma_noise**2 + sigma_beta**2)
    mu2 = alpha**2

    lam_bar = lam / alpha / (sigma_noise**2 + sigma_beta**2)
    chi = 2 / (np.sqrt((tau-1+lam_bar)**2 + 4 * lam_bar) + tau - 1 + lam_bar)

    upsilon = chi**2 * K * (1+chi - tau * chi) / ((1+chi)**2 - tau * chi**2)
    delta = -K * chi + upsilon + (1 + chi) / tau / pw
    chi_correction = chi + delta / d

    c_star_inv_correction = sigma_noise**2 * tau * np.sum(pw / (1.0 + chi_correction)) + lam + K * mu1 / mu2 / chi /d

    return c_star_inv_correction

def finite_isotropic(alpha, tau, lam, sigma_beta, sigma_noise, pw, d):
    """
    This function computes the asymptotic limits of several key parameters in the finite task case with isotropic regression vectors

    Parameters:
    alpha: N/d, where N is the number of features used in the context
    tau: n/d^2, where n is the number of samples
    lam: the regularization parameter
    sigma_beta: standard deviation of the beta vectors
    sigma_noise: standard deviation of the (Gaussian) noise in each response
    pw: the probability distribution of the finite tasks

    Returns:
    chi, chi_correction, c_star_inv, c_star_inv_correction, Gramma_trace, Gamma_F_norm, e_gen
    """

    K = len(pw)
    mu1 = alpha * (sigma_noise**2 + sigma_beta**2)
    mu2 = alpha**2

    lam_bar = lam / alpha / (sigma_noise**2 + sigma_beta**2)
    chi = 2 / (np.sqrt((tau-1+lam_bar)**2 + 4 * lam_bar) + tau - 1 + lam_bar)

    upsilon = chi**2 * K * (1+chi - tau * chi) / ((1+chi)**2 - tau * chi**2)
    delta = -K * chi + upsilon + (1 + chi) / tau / pw
    chi_correction = chi + delta / d

    c_star_inv_correction = sigma_noise**2 * tau * np.sum(pw / (1.0 + chi_correction)) + lam + K * mu1 / mu2 / chi /d
    c_star_inv = sigma_noise**2 * tau / (1 + chi) + lam

    Gamma_trace = K / alpha / d

    Gamma_F_norm = (finite_isotropic_cinv(alpha, tau, lam+1e-6, sigma_beta, sigma_noise, pw, d) - c_star_inv_correction) / 1e-6 - 1.0

    e_gen = (sigma_beta**2 * alpha * (1+alpha) + alpha * sigma_noise**2) * Gamma_F_norm - 2 * alpha * sigma_beta**2 * Gamma_trace + sigma_beta**2 + sigma_noise**2  

    return chi, chi_correction, c_star_inv, c_star_inv_correction, Gamma_trace, Gamma_F_norm, e_gen

