#!/bin/bash

folder_path=$1

# Check if folder exists
if [ ! -d "$folder_path" ]; then
  echo "Folder does not exist."
  exit 1
fi

# Iterate over files in the folder
for file in "$folder_path"/*; do
  # Exclude the parent directory file
  if [ "$(basename "$file")" != ".." ]; then
    # Extract file name from the path
    file_name=$(basename "$file")

    # Print the file name
    echo "$file_name"
    sct_maths -i $file_name -o $file_name -bin 0.2
  fi
done
