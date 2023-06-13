#!/bin/bash

input_directory="$1"
output_directory="$2"

# Check if the input and output directories are provided
if [[ -z $input_directory || -z $output_directory ]]; then
  echo "Usage: $0 <input_directory> <output_directory>"
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through all files in the input directory
for input_file in "$input_directory"/*; do
  # Check if the file is a regular file
  if [[ -f $input_file ]]; then
    # Extract the file name and extension
    file_name=$(basename "$input_file")
    file_extension="${file_name##*.}"

    # Generate the output file name
    output_file="$output_directory/${file_name%.*}.$file_extension"

    # Apply the sct_math command
    sct_maths -i "$input_file" -o "$output_file" -denoise 1
  fi
done
