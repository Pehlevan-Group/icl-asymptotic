import os

# Directory containing the files
directory = '/n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/linear_error_curves/resultsdump/job_linear_40_part2/pickles'
# List all files in the directory
files = os.listdir(directory)

# Loop through each file
ordered_orig = [9,8,7,6,5,4,3,2,1,0]
ordered_new = [19,17,15,13,11,9,7,5,3,1]
for i in range(len(ordered_orig)):
    for filename in files:
        # Check if the filename matches the pattern "train-9-*"
        if filename.startswith(f'train-{ordered_orig[i]}-') and filename.endswith('.pkl'):
            # Create the new filename by replacing "train-9-" with "train-19-"
            new_filename = filename.replace(f'train-{ordered_orig[i]}-', f'train-{ordered_new[i]}-')
            
            # Create the full old and new file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            
            # Print the change for verification
            #print(f'Renamed: {filename} -> {new_filename}')

print("Renaming completed.")