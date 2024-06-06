import numpy as np
import matplotlib.pyplot as plt

experimentdata = []
for i in range(25):
    file_path = f'error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

print(experimentdata)