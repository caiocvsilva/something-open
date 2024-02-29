import sys
# Input and output file paths
scenario = sys.argv[1]
splits = ["train", "val", "test"]
for i,val in enumerate(splits):
    input_file = "ann_"+scenario+"_"+val+".txt"
    # output_file = sys.argv[2]
    output_file = "video/video_"+input_file
    gaivi_output_file = "gaivi/gaivi_video_"+input_file

    # Function to convert the directory path to the video path
    def convert_path(directory_path, class_number, server):
        video_id = directory_path.split("/")[-1]
        if server == "lab":
            video_path = f"/shared/datasets/something/20bn-something-something-v2/{video_id}.webm"
        elif server == "gaivi":
            video_path = f"/data/sarkar-vision/something_something/20bn-something-something-v2/{video_id}.webm"
        return f"{video_path} {class_number}"

    # Open input and output files
    with open(input_file, "r") as infile, open(output_file, "w") as outfile, open(gaivi_output_file, "w") as gaivi_outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) == 3:
                directory_path, num_files, class_number = parts
                converted_line = convert_path(directory_path, class_number, "lab")
                outfile.write(f"{converted_line}\n")
                converted_line = convert_path(directory_path, class_number, "gaivi")
                gaivi_outfile.write(f"{converted_line}\n")
            elif len(parts) == 2:
                directory_path, class_number = parts
                converted_line = convert_path(directory_path, class_number, "lab")
                outfile.write(f"{converted_line}\n")
                converted_line = convert_path(directory_path, class_number, "gaivi")
                gaivi_outfile.write(f"{converted_line}\n")

print("Conversion completed.")
