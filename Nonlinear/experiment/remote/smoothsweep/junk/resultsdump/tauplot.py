import numpy as np
import matplotlib.pyplot as plt
import sys

mydir = sys.argv[1]
experimentdata = []
for i in range(2,25):
    file_path = f'./{mydir}/error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

# experimentdata = np.array(experimentdata)
# print([list(i) for i in experimentdata])

print(experimentdata)

# def errorcurve(alpha,t,l,sigma,psi):
#   L = l/(alpha*(sigma**2) + alpha*(psi**2) + (alpha**2)*(psi**2));
#   chi = (1-t-L + np.sqrt(4*t*L + (1-t+L)**2))/(2*L);
#   g1theory = (t*chi)/((1+chi)*(1+alpha+(sigma/psi)**2));
#   g2theory = (1+t*alpha*(1-chi)/((1+chi)*(1+(sigma/psi)**2)*(1+alpha+(sigma/psi)**2)))*t*chi*(psi**2+sigma**2)/(l*(1+chi)**2 + t*alpha*(sigma**2 + (1+alpha)*psi**2));
#   return g2theory*(alpha*(sigma**2) + alpha*(psi**2) + (alpha**2)*(psi**2)) - 2*alpha*(psi**2)*g1theory + psi**2 + sigma**2;

# alpha = 1;
# sigma = 0.25;
# psi = 1;

# layers = 2;
# hidden = 100;
# d = 10;
# estimate = layers*hidden/d


# # p1 = [errorcurve(alpha,t/estimate,0.5,sigma,psi) for t in range(1,101)];
# # p2 = [errorcurve(alpha,t/estimate,0.1,sigma,psi) for t in range(1,101)];
# # p3 = [errorcurve(alpha,t/estimate,1,sigma,psi) for t in range(1,101)];

# # plt.scatter(range(1,101),np.mean(experimentdata,axis=1),label="data")
# # plt.plot(range(1,101),p1,label="lambda = 0.5")
# # plt.plot(range(1,101),p2,label="lambda = 0.1")
# # plt.plot(range(1,101),p3,label="lambda = 1")
# # #plt.errorbar(range(1,101),np.mean(experimentdata,axis=1), yerr=np.var(experimentdata,axis=1), fmt='o', ecolor='red', capsize=2, capthick=1, label='Variance')
# # plt.title("d = 10: 2 layers and 100 hidden dim")
# # plt.xlabel("tau")
# # plt.legend()
# # plt.savefig('theory.png')

# iterated = np.mean(experimentdata,axis=1)
# print('Inflection on average at', np.where(iterated==np.max(iterated))[0][0])

# taus = np.linspace(0.1,50.1,26)

# iterated[91] = 1.5 # remove stupid outlier
# print('A different averaging method', np.sum(np.multiply(taus[0:50],iterated[0:50]))/np.sum(iterated[0:50]))

# print('all maxes ',[np.where(experimentdata[:,i]==np.max(experimentdata[:,i]))[0][0] for i in range(10)])
