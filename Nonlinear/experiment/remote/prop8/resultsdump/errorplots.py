import matplotlib.pyplot as plt
import numpy as np
import sys

myjob = sys.argv[1]
kappa = float(sys.argv[2])


normalvals = []
for i in range(35):
    if i != 11:
        filepath = f'./{myjob}/errors/error-{i}.txt'
        with open(filepath, 'r') as file:
            file_contents = file.read()
        normalvals.append(float(file_contents))
    if i == 11:
        normalvals.append(normalvals[-1])

ktestvals = []
for i in range(35):
    if i != 11:
        filepath = f'./{myjob}/errors/diverse-{i}.txt'
        with open(filepath, 'r') as file:
            file_contents = file.read()
        ktestvals.append(float(file_contents))
    if i == 11:
        ktestvals.append(ktestvals[-1])

# everything so far is the gamma = 0 case !!
def ridgeless(taus,rho,alpha,kappa,gamma=0):
    ridglss = []
    xstar = (1+rho)/alpha;
    Xstar = xstar/(1-gamma); 
    mstar = 2*(1/(1-gamma))/(Xstar+1-1/kappa + np.sqrt(4*Xstar/kappa + (Xstar+1-1/kappa)**2))
    for tau in taus:
        mudenominator = 1 + alpha*(1-tau/kappa)*(1-gamma)/(1+rho) + np.sqrt(4*alpha*tau*(1-gamma)/(kappa*(1+rho)) + (1 + alpha*(1-tau/kappa)*(1-gamma)/(1+rho))**2)
        mustar = 2/mudenominator
        eps = (1-tau)*(1+rho)/(alpha*tau*mustar)
        cdenominator = 1 - kappa/((1 + kappa*eps/((1-tau)*(1-gamma)))**2); c = 1/cdenominator;
        if tau<=1:
            val = (((1+rho)/alpha + eps)**2)/eps;
            val = val + (rho+(1+rho)/alpha - 2*(1+rho)*(1-tau)*(1+(1+rho)/(alpha*eps))/alpha)/(1-tau-c*((1-tau)**2));
            val = tau*val;
        else:
            val = (tau/(tau-1))*(rho+xstar*(1-xstar*mstar))
        ridglss.append(val)
        val = 0;
    return np.array(ridglss)



rho = (0.25/1)**2;

taus = (np.linspace(0.1,8,40))[0:35]
plottaus = np.linspace(taus[0],taus[-1],80)


plt.scatter(taus, normalvals,c='red',label="testing on diverse")
plt.scatter(taus, ktestvals,c='green',label="testing on finite")
plt.plot(plottaus,ridgeless(plottaus,rho,1,kappa,0),c='green',label="test error")
#plt.axvline(x=1.2,label="interpolation thresholds")
plt.legend()
plt.xlabel("tau")
plt.title(f'd = 40 Error against Tau for K = {kappa}')
plt.savefig(f'../plots/adam-d40-k{kappa}.png')

# print(ktestvals)
# print(ridgeless(np.linspace(0.1,8,40),rho,1,kappa,0))
