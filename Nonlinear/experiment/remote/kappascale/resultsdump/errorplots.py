import matplotlib.pyplot as plt
import numpy as np
import sys

myjob = sys.argv[1]
myalpha = float(sys.argv[2])

# Loop over the filtered files
vals = []
for i in range(50):
    filepath = f'./{myjob}/errors/error-{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))

plt.plot(range(11,61), vals,label="test error")
plt.xlabel("tau")
plt.title(f'2 Layers, 100 hidden dim, d = 7, k = 2*d, alpha = {myalpha}')
plt.savefig(f'../plots/errorplot-{myalpha}.png')