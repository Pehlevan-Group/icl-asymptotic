import matplotlib.pyplot as plt
import numpy as np
import sys

myjob = sys.argv[1]

experimentdata = []
for i in range(25):
    filepath = f'./{myjob}/errors/error-{i}.txt'
    with open(filepath, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

def printpython(arr):
    print("[", end="")
    for i in range(len(arr)):
        if i < len(arr) - 1:
            print(repr(arr[i]) + ",", end=" ")
        else:
            print(repr(arr[i]), end="")
    print("]")

printpython(experimentdata)

# d = 20;
# tau=50;
# alpha=5;
# kappas = [2/d,0.1,0.2,0.5,1,2,5,10,50,100];
# sigma=0.1;
# sigmabeta=1;
# rho = (sigma/sigmabeta)**2;

# #4.1882062,4.83802223,1.99632549,1.04458904,0.83643883,0.72565186,0.66393781,0.64668703,0.63389713,0.63176429];

# bayesdata = [1.9245829521574724, 1.3788177966196948, 1.3468275825678666, 1.4753011705226364, 2.0893748693415946, 1.6560834301721405, 1.1017773890498213, 1.1753983083076818, 1.1589213045048175, 1.157891941706612];
# bayesdata1 = [1.53453064357617, 1.4483083511483488, 1.7622191212941194, 1.6715067131281303, 1.4933975218962543, 1.501896456786344, 1.5042562786940101, 1.4472087796407902, 1.4114244589112372, 1.4532329393419952, 1.4219116558968705, 1.3624060242388263, 1.2471439793190047, 1.7692621691868928, 1.8447609776557639, 1.250252340028192, 1.176963726658329, 1.3106307177058654, 1.2290651440891798, 1.4544393028038791, 1.3357416607664119, 1.4009345950502694, 1.2629832617875092, 1.505221298069036, 1.5894428359213788, 1.4016614598004138, 1.2595181144731573, 1.136159959718403, 1.3858020941950546, 1.1949069323032036, 1.5942411456076344, 1.239242741643943, 1.215902635028066, 1.2620806817960235, 1.2374967420542706, 1.3653195137893468, 1.4224734869736568, 1.1781872961558044, 1.2690800030874692, 1.2405291223745816, 1.2417595543959108, 1.2233913840312352, 1.360669010669071, 1.331582513311249, 1.3075832478438405, 1.27525600193128, 1.2090031206929288, 1.619078787882527, 1.1168122720541922, 1.5667594030289358]
# bayesdata2 = [1.66855363, 1.58063513, 1.63519868, 1.46927766, 1.46967107, 1.39932752,
#  1.38590543, 1.32557986, 1.43954129, 1.29990186, 1.33486068, 1.30781869,
#  1.32957314, 1.28228336 ,1.32389381, 1.24049985, 1.28754558, 1.29865424,
#  1.35296566, 1.44400825, 1.3559446,  1.18993563, 1.25460759, 1.32446255,
#  1.28838427];

# Ksmatlab = np.array([2,3,4,5,6,7,8,9,10,12,14,15,18,20,24,38,32,36,40,45,50,55,60,70,80]);
# kappasmatlab = Ksmatlab/d;
# bayesdatamatlab = [1.5428,
#     1.3343,
#     1.2387,
#     1.2366,
#     1.1657,
#     1.1450,
#     1.1214,
#     1.0915,
#     1.0990,
#     1.0881,
#     1.0742,
#     1.0625,
#     1.0558,
#     1.0545,
#     1.0406,
#     1.0252,
#     1.0281,
#     1.0256,
#     1.0226,
#     1.0190,
#     1.0194,
#     1.0186,
#     1.0137,
#     1.0169,
#     1.0126];

# kappas = np.array(kappas)
# plt.scatter(kappasmatlab[1:20], bayesdatamatlab[1:20],c='green',label = 'finite K bayes bound wrong sigma')
# plt.scatter(kappas[1:7],experimentdata[1:7],c='red',label='Nonlinear Experiment Data')
# #plt.scatter(kappas,linearexperiment,c='gray',label='Linear Experiment Data')
# #plt.loglog(np.linspace(1,kappas[8],1000),(sigma**2 + sigmabeta**2*(sigma**2 + sigmabeta**2)/(sigma**2 + (1+alpha)*sigmabeta**2))*np.ones(1000),label='lower bound = diverse linear theory')
# #plt.loglog(np.linspace(kappas[1],0.9,1000),sigma**2 + (1+(alpha/tau)*rho/(1+rho))*(1-np.linspace(kappas[1],0.9,1000)),label='upper bound = finite bayesian estimator')
# #plt.plot(np.exp(np.linspace(np.log(0.1),np.log(50),25)),bayesdata2,label="bayes finite K error")
# plt.xlabel('kappa = K/d')
# plt.ylabel('diverse test error')
# plt.title(f'd={d}, tau={tau}, alpha={alpha}')
# plt.legend()
# plt.savefig(f'd{d}tau{tau}alpha{alpha}-nonlin.png')

# # # everything so far is the gamma = 0 case !!
# # def ridgeless(taus,rho,alpha,kappa,gamma=0):
# #     ridglss = []
# #     xstar = (1+rho)/alpha;
# #     Xstar = xstar/(1-gamma); 
# #     mstar = 2*(1/(1-gamma))/(Xstar+1-1/kappa + np.sqrt(4*Xstar/kappa + (Xstar+1-1/kappa)**2))
# #     for tau in taus:
# #         mudenominator = 1 + alpha*(1-tau/kappa)*(1-gamma)/(1+rho) + np.sqrt(4*alpha*tau*(1-gamma)/(kappa*(1+rho)) + (1 + alpha*(1-tau/kappa)*(1-gamma)/(1+rho))**2)
# #         mustar = 2/mudenominator
# #         eps = (1-tau)*(1+rho)/(alpha*tau*mustar)
# #         cdenominator = 1 - kappa/((1 + kappa*eps/((1-tau)*(1-gamma)))**2); c = 1/cdenominator;
# #         if tau<=1:
# #             val = (((1+rho)/alpha + eps)**2)/eps;
# #             val = val + (rho+(1+rho)/alpha - 2*(1+rho)*(1-tau)*(1+(1+rho)/(alpha*eps))/alpha)/(1-tau-c*((1-tau)**2));
# #             val = tau*val;
# #         else:
# #             val = (tau/(tau-1))*(rho+xstar*(1-xstar*mstar))
# #         ridglss.append(val)
# #         val = 0;
# #     return np.array(ridglss)



# # rho = (0.25/1)**2;

# # taus = (np.linspace(0.1,8,40))[0:35]
# # plottaus = np.linspace(taus[0],taus[-1],80)


# # plt.scatter(taus, normalvals,c='red',label="testing on diverse")
# # plt.scatter(taus, ktestvals,c='green',label="testing on finite")
# # plt.plot(plottaus,ridgeless(plottaus,rho,1,kappa,0),c='green',label="test error")
# # #plt.axvline(x=1.2,label="interpolation thresholds")
# # plt.legend()
# # plt.xlabel("tau")
# # plt.title(f'd = 40 Error against Tau for K = {kappa}')
# # plt.savefig(f'../plots/adam-d40-k{kappa}.png')

# # # print(ktestvals)
# # # print(ridgeless(np.linspace(0.1,8,40),rho,1,kappa,0))
