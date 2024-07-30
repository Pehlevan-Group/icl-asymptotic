import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

mydir = sys.argv[1]
d = int(sys.argv[2])
experimentdata = []
for i in range(5):
    file_path = f'./{mydir}/idg-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

## PLOTS !!!!!
# DEFINE STANDARD FORMATING FOR FIGURES USED THROUGHOUT PAPER
sns.set(style="white",font_scale=2.5,palette="colorblind")
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["figure.figsize"] = (12, 10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
myred = '#8B0000'
colors = [myred, color_cycle[1], color_cycle[2]]
plt.gca().spines['top'].set_color('lightgray')
plt.gca().spines['right'].set_color('lightgray')
plt.gca().spines['bottom'].set_color('lightgray')
plt.gca().spines['left'].set_color('lightgray')
print([np.mean(experimentdata[i]) for i in range(len(experimentdata))])
print([np.std(experimentdata[i]) for i in range(len(experimentdata))])
means = np.array([np.mean(experimentdata[i]) for i in range(len(experimentdata))]);
stds = np.array([np.std(experimentdata[i]) for i in range(len(experimentdata))]);

Ks20 = list(range(2,d+1,2)) + list(np.int64(np.log(np.logspace(1.5*d,10*d,30)))); 
Ks40 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(10*d),30)));
Kappasfix = [23, 28, 35, 45, 57];
Ks = np.array(Kappasfix)
print([k for k in Ks])
kappas = Ks/d;

plt.scatter(kappas[1:], means[1:])
plt.fill_between(kappas[1:], means[1:]-stds[1:], means[1:]+stds[1:],alpha=0.2)
plt.xscale('log')

# Nice legend
leg = plt.legend()
leg.get_frame().set_alpha(0)
# Axis Formatting
plt.xlabel(r'$\tau = n/d^2$')
plt.ylabel(r'$e^{ICL}(\Gamma^*)$')
plt.xticks(fontsize=20);
plt.yticks(fontsize=20);
plt.savefig(f'{mydir}/plot.png', bbox_inches='tight')
