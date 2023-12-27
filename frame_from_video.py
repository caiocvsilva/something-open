import os
import sys

txt_file = sys.argv[1]
output_file = 'frame_'+txt_file

# Open the input and output files
with open(txt_file, 'r') as infile, open(output_file, 'w') as outfile:
    # For each line in the input file
    for line in infile:
        # Split the line into path_directory and class_int
        path_directory, class_int = line.strip().split()

        # replace string 20bn-something-something-v2 with raw_frames
        path_directory = path_directory.replace('20bn-something-something-v2', 'rawframes')

        # remove .webm from the path_directory
        path_directory = path_directory.replace('.webm', '')

        # Count the number of files in the path_directory
        num_files = len(os.listdir(path_directory))

        # Write a new line to the output file
        outfile.write(f'{path_directory} {num_files} {class_int}\n')