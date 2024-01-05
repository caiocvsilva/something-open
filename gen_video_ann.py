import sys
# Input and output file paths
input_file = sys.argv[1]
output_file = sys.argv[2]
server = sys.argv[3]

# Function to convert the directory path to the video path
def convert_path(directory_path, class_number):
    video_id = directory_path.split("/")[-1]
    if server == "lab":
        video_path = f"/shared/datasets/something/20bn-something-something-v2/{video_id}.webm"
    elif server == "gaivi":
        video_path = f"/data/sarkar-vision/something_something/20bn-something-something-v2/{video_id}.webm"
    return f"{video_path} {class_number}"

# Open input and output files
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.strip().split()
        if len(parts) == 3:
            directory_path, num_files, class_number = parts
            converted_line = convert_path(directory_path, class_number)
            outfile.write(f"{converted_line}\n")
        elif len(parts) == 2:
            directory_path, class_number = parts
            converted_line = convert_path(directory_path, class_number)
            outfile.write(f"{converted_line}\n")

print("Conversion completed.")
