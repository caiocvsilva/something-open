#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <output_directory>"
    exit 1
fi

# Source directory and output directory
source_dir="$1"
output_dir="$2"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir/train"
mkdir -p "$output_dir/validation"

# Function to create subdirectories in the output directories
create_subdirectories() {
    local class_dir="$1"
    local output_dir="$2"
    local class_name=$(basename "$class_dir")

    # Create subdirectories in the output directory
    mkdir -p "$output_dir/validation/$class_name"
    mkdir -p "$output_dir/train/$class_name"
}

# List subdirectories (classes) in the source directory
echo "Searching for class directories in $source_dir..."
for class_dir in "$source_dir"/*/; do
    if [ -d "$class_dir" ]; then
        class_name=$(basename "$class_dir")

        # Count the number of .jpg images in the class directory
        echo "dir: $class_dir"
        num_images=$(ls "$class_dir"| wc -l)
        echo "Found $num_images .jpg images in class $class_name..."

        # Calculate the number of images for validation (10%)
        validation_count=$((num_images / 10))
        echo "Creating $validation_count validation links for class $class_name..."

        # Ensure that the validation count is at least 1
        if [ "$validation_count" -lt 1 ]; then
            validation_count=1
        fi

        # Create subdirectories in the output directories
        create_subdirectories "$class_dir" "$output_dir"

        # Create soft links for the validation set
        # find "$class_dir" -type f -iname "*" | head -n "$validation_count" | while read -r image; do
        #     ln -s "$image" "$output_dir/validation/$class_name/$(basename "$image")"
        #     echo "Linking $image to validation set..."
        # done

        for file in $(ls "$class_dir" | head -"$validation_count"); do cp "$class_dir"/"$file" "$output_dir/validation/$class_name/$(basename "$file")"; done


        # # Create soft links for the training set
        # find "$class_dir" -type f -iname "*" | tail -n "+$((validation_count + 1))" | while read -r image; do
        #     ln -s "$image" "$output_dir/train/$class_name/$(basename "$image")"
        #     echo "Linking $image to training set..."
        # done

        for file in $(ls "$class_dir" | tail -n +"$validation_count"); do cp "$class_dir"/"$file" "$output_dir/train/$class_name/$(basename "$file")"; done



    fi
done

echo "Data splitting and linking completed."
