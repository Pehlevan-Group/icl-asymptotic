import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv_to_array(file_path):
    data_array = []
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data_array.append(row)
    return np.array(data_array)

names = ['icl_dmmse','icl_ridge','idg_dmmse','idg_ridge'];
for name in names:
    file_path = f'./{name}.csv'
    data = load_csv_to_array(file_path)
    new_var_name = f"data_{name}"
    exec(f"{new_var_name} = data")


d = 100;
kappas = np.logspace(np.log10(0.5),np.log10(100),50);
alphas = [0.25,50] #[0.25,0.5,0.75,1,5,10,50];

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(len(alphas)):
    ax1.plot(kappas,data_icl_dmmse[i],color=color_cycle[i],label=f'dMMSE alpha = {alphas[i]}')
    ax1.plot(kappas,data_icl_ridge[i],':',color=color_cycle[i],label=f'RIDGE alpha = {alphas[i]}')
    ax2.plot(kappas,data_idg_dmmse[i],color=color_cycle[i],label=f'dMMSE alpha = {alphas[i]}')
    ax2.plot(kappas,data_idg_ridge[i],':',color=color_cycle[i],label=f'RIDGE alpha = {alphas[i]}')
ax1.legend()
ax1.set_xscale('log')
ax2.legend()
ax2.set_xscale('log')
fig.savefig('comparision.png')

# color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# i = 6;
# plt.plot(kappas,data_icl_dmmse[i],color=color_cycle[i],label=f'dMMSE alpha = {alphas[i]}')
# plt.plot(kappas,data_icl_ridge[i],':',color=color_cycle[i],label=f'RIDGE alpha = {alphas[i]}')
# plt.legend()
# plt.xscale('log')
# plt.savefig('comparision.png')