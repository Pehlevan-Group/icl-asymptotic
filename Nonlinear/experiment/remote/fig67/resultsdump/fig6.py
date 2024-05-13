import matplotlib.pyplot as plt
import numpy as np
import sys
import os

experimentdata = []
directory = './fig6/errors'
for filename in os.listdir(directory):
    print(filename)
    filepath = f'{directory}/{filename}'
    with open(filepath, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)
experimentdata = np.array(experimentdata)

print(experimentdata.shape)
orderkappas = [5,0.5,1,0.1]

taus = np.linspace(10,50,20); 
d=20;
plt.plot(taus,experimentdata[0,:],label = f'kappa = {orderkappas[0]}')
plt.plot(taus,experimentdata[1,:],label = f'kappa = {orderkappas[1]}')
plt.plot(taus,experimentdata[2,:],label = f'kappa = {orderkappas[2]}')
plt.plot(taus,experimentdata[3,:],label = f'kappa = {orderkappas[3]}')
plt.plot(taus,experimentdata[4,:],label = f'fully diverse')
plt.legend()
plt.xlabel('tau')
plt.ylabel('test error')
plt.title(f'Comparision of Task Diversity Test Curves against Tau - d = {d}')
plt.savefig('myfig6.png')