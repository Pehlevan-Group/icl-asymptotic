import matplotlib.pyplot as plt
import numpy as np
import sys

# Loop over the filtered files
vals = []
for i in range(50):
    filepath1 = f'./k1a1_20048792/errors/error-{i}.txt'
    filepath2 = f'./k1a1_20049252/errors/error-{i}.txt'
    with open(filepath1, 'r') as file:
        file_contents1 = file.read() 
    with open(filepath2, 'r') as file:
        file_contents2 = file.read() 
    vals.append((float(file_contents1) + float(file_contents2))/2)

plt.plot(range(11,61), vals,label="2averaged test error")
plt.xlabel("tau")
plt.title(f'2 Layers, 100 hidden dim, d = 10, k = 2*d, alpha = 1')
plt.savefig(f'../plots/errorplot-1.png')