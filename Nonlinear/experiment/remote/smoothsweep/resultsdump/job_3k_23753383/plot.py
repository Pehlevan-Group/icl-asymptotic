import numpy as np
import matplotlib.pyplot as plt

experimentdata = []
for i in range(26):
    file_path = f'error-{i}-0.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-1.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-2.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-3.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    file_path = f'error-{i}-4.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = numbers + [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

experimentdata = np.array(experimentdata)
print(np.sum(experimentdata, axis=1)/5)
taus = np.linspace(0.1,50.1,26);
plt.plot(taus, np.sum(experimentdata, axis=1)/5)
plt.savefig('diverse.png')
