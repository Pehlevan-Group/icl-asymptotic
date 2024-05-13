import matplotlib.pyplot as plt
import numpy as np
import sys

myjob='linestar5_19923005'
vals = []
for i in range(40):
    filepath = f'./{myjob}/errors/error-{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))
myjob='linestar5_19951324'
for i in range(40):
    filepath = f'./{myjob}/errors/error-{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))
plt.plot(range(1,81), vals,label="test error")
plt.xlabel("tau")
plt.title(f'2 Layers, 10*d hidden dim, d = 5')
plt.savefig(f'./errorplot-5.png')