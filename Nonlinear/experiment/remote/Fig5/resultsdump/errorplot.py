import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

mydir = sys.argv[1]
d = int(sys.argv[2])
icl = []
inds = [a for a in range(8)] + [a for a in range(25,31)]
for i in inds:
    file_path = f'./{mydir}/icl-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    icl.append(numbers)
idg = []
for i in inds:
    file_path = f'./{mydir}/idg-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    idg.append(numbers)

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
means_icl = np.array([np.mean(icl[i]) for i in range(len(icl))]);
stds_icl = np.array([np.std(icl[i]) for i in range(len(icl))]);
means_idg = np.array([np.mean(idg[i]) for i in range(len(idg))]);
stds_idg = np.array([np.std(idg[i]) for i in range(len(idg))]);

# Ks20 = list(range(2,d+1,2)) + list(np.int64(np.log(np.logspace(1.5*d,10*d,30)))); 
# Ks40 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(10*d),30)));
# Kappasfix = [23, 28, 35, 45, 57];
Ks20 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 28, 35, 45, 57, 69, 82, 96, 109, 123, 136, 150, 163, 177, 190, 204, 217, 231, 244, 258, 271, 285, 298, 312, 325, 339, 352, 366, 379, 393, 406, 420, 433, 447, 460];
Ks40 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(10*d),30)))
Ks80 = np.array(list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(5*d),15))))
if d == 80:
    Ks = np.array(Ks80[inds])
if d == 40:
    Ks = np.array(Ks40[:34])
if d == 20:
    Ks = np.array(Ks20)
print([k for k in Ks])
kappas = Ks/d;

start = 0
plt.scatter(kappas[start:], means_icl[start:],label='icl')
plt.fill_between(kappas[start:], means_icl[start:]-stds_icl[start:], means_icl[start:]+stds_icl[start:],alpha=0.2)
plt.scatter(kappas[start:], means_idg[start:],label='idg')
plt.fill_between(kappas[start:], means_idg[start:]-stds_idg[start:], means_idg[start:]+stds_idg[start:],alpha=0.2)
plt.axvline(1,linestyle=':',color='grey')
plt.legend()
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
