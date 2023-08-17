# Script to create json file coresponding to a .nii.gz file
import json
import datetime
import sys

def write_json(filename):
    data = {
        "Author": "Theo Mathieu", # Change with your name
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write the data to the JSON file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"JSON data written to {filename} successfully.")

# Check if a filename is provided as a command-line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
    write_json(f"{filename}.json")
else:
    print("Please provide a filename as a command-line argument.")
