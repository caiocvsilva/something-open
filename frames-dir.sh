#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <json_file> <main_dir>"
    exit 1
fi

# JSON file path and main directory
json_file="$1"
main_dir="$2"

# Check if the main directory exists, if not, create it
if [ ! -d "$main_dir" ]; then
    mkdir -p "$main_dir"
fi

# Function to find the middle frame index
find_middle_frame() {
    local frame_dir="$1"
    local frame_files=("$frame_dir"/*)
    local num_frames=${#frame_files[@]}

    # Calculate the index of the middle frame
    local middle_index=$((num_frames / 2))
    echo "${frame_files[$middle_index]}"
}

# Read and process the JSON data
while IFS= read -r line; do
    id=$(echo "$line" | jq -r '.id')
    noun=$(echo "$line" | jq -r '.nouns[0]')

    # Subdirectory inside the main directory
    sub_dir="$main_dir/$noun"

    # Check if the subdirectory exists, if not, create it
    if [ ! -d "$sub_dir" ]; then
        mkdir -p "$sub_dir"
    fi

    # Path to the middle frame of the video
    frame_dir="/home/caio/Documents/datasets/something/rawframes/$id"
    middle_frame=$(find_middle_frame "$frame_dir")

    # Soft link to the middle frame
    frame_link="$sub_dir/$id.jpg"

    # Check if the link already exists, if not, create it
    if [ ! -e "$frame_link" ]; then
        ln -s "$middle_frame" "$frame_link"
    fi

done < "$json_file"
