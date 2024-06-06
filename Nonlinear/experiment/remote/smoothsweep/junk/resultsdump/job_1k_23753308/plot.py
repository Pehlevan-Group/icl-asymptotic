import numpy as np
import matplotlib.pyplot as plt

experimentdata = []
for i in range(25):
    file_path = f'error-{i}-0.txt'
    # Read the numbers from the file and convert them to floats
    if i == 11:
        numbers = [experimentdata[-1][0]];
    else:
        with open(file_path, 'r') as file:
            numbers = [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-1.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-2.txt'
    # Read the numbers from the file and convert them to floats
    if i == 14 or i == 23:
        numbers = [experimentdata[-1][2]];
    else:
        with open(file_path, 'r') as file:
            numbers = [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-3.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-4.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

addd = '../job_3k_23753383'
high = []
for i in range(25):
    file_path = f'{addd}/error-{i}-0.txt'
    # Read the numbers from the file and convert them to floats
    if i == 11:
        numbers = [high[-1][0]];
    else:
        with open(file_path, 'r') as file:
            numbers = [float(line.strip()) for line in file if line.strip()]
    file_path = f'{addd}/error-{i}-1.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    file_path = f'{addd}/error-{i}-2.txt'
    # Read the numbers from the file and convert them to floats
    if i == 14 or i == 23:
        numbers = [high[-1][2]];
    else:
        with open(file_path, 'r') as file:
            numbers = [float(line.strip()) for line in file if line.strip()]
    file_path = f'{addd}/error-{i}-3.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    file_path = f'{addd}/error-{i}-4.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    high.append(numbers)

experimentdata = np.array(experimentdata)
high = np.array(high)
taus = np.linspace(0.1,50.1,26);

# Set global font settings to Times New Roman
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

# Define a custom color palette
custom_colors = ['#8283F1', '#281682', '#6FA3EC', '#7F68AA']
colorah = '#6FA3EC'
green1 = '#96CD79'
green2 = '#0B5C36'
red1 = '#B80000'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams["figure.figsize"] = (10, 6)

stop = len(np.mean(experimentdata,axis=1))
stop1 = len(np.mean(high,axis=1))
plt.scatter(taus[range(stop)],np.mean(experimentdata,axis=1),label='low k')
plt.errorbar(taus[range(stop)],np.mean(experimentdata,axis=1), yerr=np.var(experimentdata,axis=1), linewidth = 1, fmt='o', capsize=1, capthick=1)
plt.axvline(x=22,c=green2,label='Interpolation Threshold for low K')
plt.scatter(taus[range(stop1)],np.mean(high,axis=1),label = 'high k')
plt.errorbar(taus[range(stop1)],np.mean(high,axis=1), yerr=np.var(experimentdata,axis=1), linewidth = 1, fmt='o', capsize=1, capthick=1)
plt.xlabel("tau")
plt.ylabel("Average Test Error")
plt.legend()
plt.savefig("varlow.png",bbox_inches='tight')
