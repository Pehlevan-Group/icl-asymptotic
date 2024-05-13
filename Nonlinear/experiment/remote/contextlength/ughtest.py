import numpy as np

alphatests = np.linspace(0.1,2,20);
for alphatest in alphatests:
    Ntest = int(alphatest*20);
    file_path = f'./ugh.txt'
    with open(file_path, 'a') as file:
        file.write(f'{Ntest}\n')