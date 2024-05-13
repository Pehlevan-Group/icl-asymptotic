import matplotlib.pyplot as plt
import numpy as np

# Loop over the filtered files
vals = []
for i in range(40):
    filepath = f'./error-P{i}.txt'
    with open(filepath, 'r') as file:
        file_contents = file.read()
    vals.append(float(file_contents))

plt.plot(range(10,50), vals,label="test error")
plt.xlabel("tau")
plt.title("2 Layers, 100 hidden dim")
plt.savefig("minibatch-errorplot-2-100.png")