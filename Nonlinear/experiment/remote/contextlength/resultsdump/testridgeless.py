import numpy as np

def ridgeless(taus,rho,alpha,kappa,gamma=0):
    ridglss = []
    xstar = (1+rho)/alpha;
    Xstar = xstar/(1-gamma); 
    mstar = 2*(1/(1-gamma))/(Xstar+1-1/kappa + np.sqrt(4*Xstar/kappa + (Xstar+1-1/kappa)**2))
    for tau in taus:
        mudenominator = 1 + alpha*(1-tau/kappa)*(1-gamma)/(1+rho) + np.sqrt(4*alpha*tau*(1-gamma)/(kappa*(1+rho)) + (1 + alpha*(1-tau/kappa)*(1-gamma)/(1+rho))**2)
        mustar = 2/mudenominator;
        eps = (1-tau)*xstar/(tau*mustar); #(1-tau)*(1+rho)/(alpha*tau*mustar)
        cdenominator = 1 - kappa/((1 + kappa*eps/((1-tau)*(1-gamma)))**2); 
        c = 1/cdenominator;
        if tau<=1:
            val = ((xstar + eps)**2)/eps;
            val = val + (rho + xstar - 2*(1+rho)*(1-tau)*(1+xstar/eps)/alpha)/(1-tau-c*((1-tau)**2));
            val = tau*val;
        else:
            val = (tau/(tau-1))*(rho+xstar*(1-xstar*mstar))
        ridglss.append(val)
        val = 0;
    return np.array(ridglss)

mytaus = [0.999,1.001]
print(ridgeless(mytaus,0.25,1,3,0))
