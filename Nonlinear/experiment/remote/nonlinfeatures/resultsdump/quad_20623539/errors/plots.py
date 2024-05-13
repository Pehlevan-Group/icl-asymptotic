import numpy as np
import matplotlib.pyplot as plt

experimentdata = []
for i in range(30):
    file_path = f'error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
        if len(numbers) == 1:
            print("ew", i)
            numbers = [numbers[0],numbers[0]]
    experimentdata.append(numbers)

experimentdata = np.array(experimentdata)
print(experimentdata.shape)

plt.scatter(range(11,41),np.mean(experimentdata,axis=1),label="data")
plt.errorbar(range(11,41),np.mean(experimentdata,axis=1), yerr=np.var(experimentdata,axis=1), fmt='o', ecolor='red', capsize=2, capthick=1, label='Variance')
plt.title("QUAD: d = 10, 2 layers, 100 hidden dim")
plt.xlabel("tau")
plt.legend()
plt.savefig('quad.png')