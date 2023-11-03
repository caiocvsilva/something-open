import json
import sys

# Input JSON file with multiple objects
input_file = sys.argv[1]

# Output JSON file with a single object
output_file = input_file.split('.')[0]+'_fix.json'

# Read the input JSON file
with open(input_file, 'r') as infile:
    # Initialize an empty list to store the individual JSON objects
    json_objects = []

    # Read each line (each line is a separate JSON object) and parse it
    for line in infile:
        try:
            json_obj = json.loads(line)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON: {e}")

    # Combine all objects into a single dictionary
    combined_json = {"combined_objects": json_objects}

# Write the combined JSON to the output file
with open(output_file, 'w') as outfile:
    json.dump(combined_json, outfile, indent=4)

print(f"Combined {len(json_objects)} JSON objects into a single JSON file: {output_file}")
