import numpy as np
import sys

# Replace 'filename.txt' with the path to your .txt file
directory = sys.argv[1]
filename = sys.argv[2]
filename = f'{directory}/{filename}.txt'

# Read the file and convert to a 2D array
with open(filename, 'r') as file:
    data = []
    for line in file:
        # Split the line by comma and convert to float (or int)
        row = [float(num) for num in line.strip().split(',')]
        data.append(row)

# Convert to a numpy array for further use
data = np.array(data)

# Sort the data by the first column
sorted_data = data[data[:, 0].argsort()]

# Extract the second column values from the sorted data
sorted_second_column = sorted_data[:, 1].tolist()

print(list(sorted_second_column))