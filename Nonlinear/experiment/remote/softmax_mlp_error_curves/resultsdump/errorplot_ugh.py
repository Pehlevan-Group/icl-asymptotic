import numpy as np
import matplotlib.pyplot as plt
import sys

d = int(sys.argv[1])

mydir = 'job_soft2mlp40'
experimentdata1 = []
for i in range(30):
    file_path = f'./{mydir}/error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata1.append(numbers)
mydir = 'job_soft2mlp40ugh'
experimentdata2 = []
for i in range(12):
    file_path = f'./{mydir}/error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata2.append(numbers)

taus1 = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.75,2,2.25,2.5,3,3.5,4,4.5,5])
taus2 = np.array([2.6,2.7,2.8,2.9,3.25,3.75,4.25,4.75,6,7,8,10])

means1 = np.array([np.mean(experimentdata1[i]) for i in range(len(experimentdata1))]); stds1 = np.array([np.std(experimentdata1[i]) for i in range(len(experimentdata1))]);

mpair1 = list(zip(taus1, means1)); spair1 = list(zip(taus1,stds1))
means2 = np.array([np.mean(experimentdata2[i]) for i in range(len(experimentdata2))]); stds2 = np.array([np.std(experimentdata2[i]) for i in range(len(experimentdata2))])

mpair2 = list(zip(taus2, means2)); spair2 = list(zip(taus2,stds2))
means = np.concatenate((mpair1, mpair2), axis=0); 
means = sorted(means, key=lambda pair: pair[0]);
#sorted_indices = np.argsort(means[0, :]); means = means[:, sorted_indices]
stds = np.concatenate((spair1, spair2), axis=0);
stds = sorted(stds, key=lambda pair: pair[0])
#sorted_indices = np.argsort(stds[0, :]); stds = stds[:, sorted_indices]

means = np.array(means); stds = np.array(stds);
plt.plot(means[:,0],means[:,1],label=f'd = {d}')
plt.fill_between(means[:,0], means[:,1]-stds[:,1], means[:,1]+stds[:,1], alpha = 0.2)
print("ordered taus are")
print([i for i in means[:,0]])
print("ordered means are")
print([i for i in means[:,1]])
print("ordered stds are")
print([i for i in stds[:,1]])
#plt.fill_between(taus,np.mean(experimentdata,axis=1)-np.std(experimentdata,axis=1),np.mean(experimentdata,axis=1)+np.std(experimentdata,axis=1),alpha=0.2)
#plt.plot(np.linspace(taus[0],taus[-1],1000),[icl_theory(tau,1,0.01,10000) for tau in np.linspace(taus[0],taus[-1],1000)],'-',label='theory')
# plt.plot(taus, np.mean(ar40_14,axis=1),label=f'40')
# plt.plot(taus, np.mean(ar20_14,axis=1),label=f'20')
plt.legend()

plt.savefig(f'full_plot{d}.png')

