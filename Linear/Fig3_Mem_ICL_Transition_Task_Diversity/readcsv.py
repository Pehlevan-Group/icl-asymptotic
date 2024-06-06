import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv_to_array(file_path):
    data_array = []
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data_array.append(row)
    return np.array(data_array)

names = ['icl_dmmse','icl_ridge','idg_dmmse','idg_ridge'];
for name in names:
    file_path = f'./{name}.csv'
    data = load_csv_to_array(file_path)
    new_var_name = f"data_{name}"
    exec(f"{new_var_name} = data")

plt.plot()