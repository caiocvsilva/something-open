import sys

classes_file = sys.argv[1]
unknown_classes = ['moving', 'right', 'is', 'pushing', 'catching', 'throwing', 'dropping', 'left', 'collide', 'showing']
test_file = sys.argv[2]
val_file = test_file.replace('test', 'val')

# Open the file for reading
with open(classes_file, 'r') as file:
    # Read the contents of the file into a string
    file_contents = file.read()

# Split the string by comma and remove any leading/trailing whitespace
classes = [string.strip() for string in file_contents.split(',')]

# Remove any single quotes from the strings
classes = [string.replace("'", "") for string in classes]

# Get the index of array classes, in which an item or part of an item is in unknown_classes
unknown_indices = [i for i, item in enumerate(classes) if any(word in item.split() for word in unknown_classes)]

# print(indices)
# print(len(indices))

# read text file into a list of lines
with open(test_file, 'r') as file:
    lines = file.readlines()

# remove lines with indices in unknown_indices
lines = [line for i, line in enumerate(lines) if i not in unknown_indices]

# write lines to a new file
with open(val_file, 'w') as file:
    file.writelines(lines)