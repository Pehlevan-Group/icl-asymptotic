import numpy as np
import matplotlib.pyplot as plt

experimentdata = []
for i in range(100):
    file_path = f'fixed-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

experimentdata = np.array(experimentdata)

print([list(i) for i in experimentdata])

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

# # Set global font settings to Times New Roman
# plt.rcParams["font.family"] = 'serif'
# plt.rcParams['font.size'] = 12
# plt.rcParams['lines.linewidth'] = 2

# # Define a custom color palette
# custom_colors = ['#8283F1', '#281682', '#6FA3EC', '#7F68AA']
# colorah = '#6FA3EC'
# green1 = '#96CD79'
# green2 = '#0B5C36'
# red1 = '#B80000'
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams["figure.figsize"] = (10, 6)

# plt.scatter(range(1,101),np.mean(experimentdata,axis=1))
# plt.errorbar(range(1,101),np.mean(experimentdata,axis=1), yerr=np.var(experimentdata,axis=1), linewidth = 1, fmt='o', capsize=1, capthick=1)
# plt.axvline(x=25,c=green2,label='Interpolation Threshold')
# plt.xlabel("tau")
# plt.ylabel("Average Test Error")
# plt.legend()
# plt.savefig("varfinal.png",bbox_inches='tight')

# iterated = np.mean(experimentdata,axis=1)
# print('Inflection on average at', np.where(iterated==np.max(iterated))[0][0])

# taus = range(1,101)
# iterated[91] = 1.5 # remove stupid outlier
# print('A different averaging method', np.sum(np.multiply(taus[0:50],iterated[0:50]))/np.sum(iterated[0:50]))

# print('all maxes ',[np.where(experimentdata[:,i]==np.max(experimentdata[:,i]))[0][0] for i in range(10)])
