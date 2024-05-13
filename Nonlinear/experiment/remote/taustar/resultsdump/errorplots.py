import matplotlib.pyplot as plt
import numpy as np
import sys

myjob = sys.argv[1]
myd = int(sys.argv[2])

# Loop over the filtered files
vals = []
for i in range(30):
    filepath = f'./{myjob}/errors/error-{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))

plt.plot(range(1,31), vals,label="test error")
plt.xlabel("tau")
plt.title(f'2 Layers, 10*d hidden dim, d = {myd}')
plt.savefig(f'./{myjob}/errorplot-{myd}.png')