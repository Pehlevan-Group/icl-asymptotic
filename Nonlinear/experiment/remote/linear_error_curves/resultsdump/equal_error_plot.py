import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_closest_index(arr, target):
    # Check if the array is empty
    if not arr:
        return None
    
    # Initialize variables to store the index and the smallest difference
    closest_index = 0
    smallest_difference = abs(arr[0] - target)
    
    # Iterate through the array
    for i in range(1, len(arr)):
        current_difference = abs(arr[i] - target)
        if current_difference < smallest_difference:
            smallest_difference = current_difference
            closest_index = i
            
    return closest_index

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *

taus_so_far = 25

taus = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5,7,10])[:taus_so_far]
#dirs = [ 'job_linear20', 'job_linear20_a0p5', 'job_linear40' ,  'job_linear40_a0p5', 'job_linear80', 'job_linear80_a0p5']
#names = ['linear, d = 20, a =1', 'linear, d = 20, a = 0.5', 'linear, d = 40, a =1', 'linear, d = 40, a =0.5', 'linear, d = 80, a =1', 'linear, d = 80, a = 0.5']
dirs = ['job_linear20','job_linear40','job_linear80']
names = ['d = 20', 'd = 40', 'd = 80'] 

#styles = ['-',':','-',':','-',':']
num_iters = 5
total_data = np.zeros((len(dirs),taus_so_far,num_iters))
total_data_1 = np.zeros((len(dirs),taus_so_far,num_iters))
for tauindex in range(taus_so_far):
    for iter in range(num_iters):
        # read data from all files
        full_train = []; full_test = []
        for mydir in dirs:
            trainvals = []
            testvals = []
            file_path = f'{mydir}/pickles/train-{tauindex}-{iter}.pkl'
            with open(file_path, 'rb') as fp:
                loaded = pickle.load(fp)
            trainloss = [Metrics.loss for Metrics in loaded['train']]
            testloss = [Metrics.loss for Metrics in loaded['test']]
            for loss_array in trainloss:
                trainvals.append(loss_array.item())
            for loss_array in testloss:
                testvals.append(loss_array.item())
            trainvals=np.array(trainvals)
            testvals=np.array(testvals)
            full_train.append(trainvals)
            full_test.append(testvals)
        
        # compute variances of training data
        vars_train = []
        for i in range(len(dirs)):
            vars_train.append([np.var(full_train[i][j-5:j+1]) for j in range(10,len(full_train[i]))])
            
        # compare variances to largest d run
        match_indices = []
        for i in range(len(dirs)):
            match_indices.append(find_closest_index(vars_train[i],vars_train[-1][-1]) + 10)
            total_data[i,tauindex,iter] = full_test[i][match_indices[-1]]
            total_data_1[i,tauindex,iter] = full_test[i][-1]
        

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
print([a for a in taus])
for i in range(len(dirs)):
    print(names[i])
    print([a for a in np.mean(total_data[i,:,:], axis = 1)])
    print([a for a in np.std(total_data[i,:,:], axis = 1)])
    # plt.plot(taus,np.mean(total_data[i,:,:], axis = 1),'-',color=colors[i],label = f'{names[i]}')
    # plt.fill_between(taus,np.mean(total_data[i,:,:], axis = 1)-np.std(total_data[i,:,:], axis = 1),np.mean(total_data[i,:,:], axis = 1)+np.std(total_data[i,:,:], axis = 1),alpha=0.2,color=colors[i])
    # # plt.plot(taus,np.mean(total_data_1[i,:,:], axis = 1),label = f'unmodified {dirs[i]}')
    # # plt.fill_between(taus,np.mean(total_data_1[i,:,:], axis = 1)-np.std(total_data_1[i,:,:], axis = 1),np.mean(total_data_1[i,:,:], axis = 1)+np.std(total_data_1[i,:,:], axis = 1),alpha=0.2)

# # Nice legend
# leg = plt.legend()
# leg.get_frame().set_alpha(0)
# # Axis Formatting
# plt.xlabel(r'$\tau = n/d^2$')
# plt.ylabel(r'$e^{ICL}(\Gamma^*)$')
# plt.xticks(fontsize=20);
# plt.yticks(fontsize=20);
# plt.savefig("icl_dd_full_linear.pdf", bbox_inches='tight')