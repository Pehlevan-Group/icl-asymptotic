import os

directory1 = 'job_soft80fix'
directory2 = 'job_80finish'
for filename in os.listdir(directory1):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Construct the full path to the file
            filepath1 = os.path.join(directory1, filename)
            # Open and read the content of the file
            with open(filepath1, 'r') as file:
                numbers = [float(line.strip()) for line in file if line.strip()]
            filepath2 = os.path.join(directory2, filename)
            # Open and read the content of the file
            with open(filepath2, 'a') as file:
                for val in numbers:
                    file.write(f'{val}\n')