import sys

train_txt = sys.argv[1]
test_txt = sys.argv[2]
scenario = sys.argv[3]

# Read the first text file
with open(train_txt, 'r') as file:
    train_data = {line.split()[1] for line in file}

# Read the second text file
with open(test_txt, 'r') as file:
    test_data = {line.split()[1] for line in file}

# Extract the IDs from the second text file that are not in the first text file
unknown_ids = test_data.difference(train_data)

# Write the new IDs to a file
with open('unknown_classes_'+scenario+'.txt', 'w') as file:
    ids_list = [id for id in unknown_ids]
    file.write(str(ids_list))
