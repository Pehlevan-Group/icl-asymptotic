import sys

sys.path.append('../../')
sys.path.append('../../../')

for i in range(10):
    if i != 40:
        file_path = f'../../../../resultstheory/results10d{i}.txt'
        with open(file_path, 'r') as file:
            file_contents = file.read()
        print(file_contents)

