import ast
import random

input_train_file = '/shared/something-open/frame_ann_unkv_train.txt'
input_val_file = '/shared/something-open/frame_ann_unkv_val.txt'
input_test_file = '/shared/something-open/frame_ann_unkv_test.txt'
input_classes_file = '/shared/something-open/class_unkv.txt'

output_train_file = input_train_file.split('.')[0] + '_10_classes.txt'
output_val_file = input_val_file.split('.')[0] + '_10_classes.txt'
output_test_file = input_test_file.split('.')[0] + '_10_classes.txt'
output_classes_file = input_classes_file.split('.')[0] + '_10_classes.txt'

# Read train.txt, val.txt, and test.txt
with open(input_train_file, 'r') as train_file, open(input_val_file, 'r') as val_file, open(input_test_file, 'r') as test_file:
    train_lines = train_file.readlines()
    val_lines = val_file.readlines()
    test_lines = test_file.readlines()

#get class ids from train.txt, removing duplicates
class_ids = list(set([int(line.split()[2]) for line in train_lines]))

# Randomly choose 10 class IDs
random_class_ids = random.sample(class_ids, 10)

# get class ids from test.txt, removing duplicates
class_ids_test = list(set([int(line.split()[2]) for line in test_lines]))

# randomly choose 5 class IDs not in the 10 chosen class IDs
random_class_ids_tests = random.sample([class_id for class_id in class_ids_test if class_id not in random_class_ids], 5)



# Filter train.txt, val.txt, and test.txt based on the chosen class IDs
train_lines = [line for line in train_lines if int(line.split()[2]) in random_class_ids]
val_lines = [line for line in val_lines if int(line.split()[2]) in random_class_ids]
# remove all lines not in random_class_ids and random_class_ids_tests
test_lines = [line for line in test_lines if int(line.split()[2]) in random_class_ids or int(line.split()[2]) in random_class_ids_tests]


# Read classes.txt
with open(input_classes_file, 'r') as classes_file:
    classes = classes_file.read().splitlines()

# Convert the string representation of the list into an actual list
classes = ast.literal_eval(classes[0])

# Remove classes not in random_class_ids or random_class_ids_tests
classes = [class_name for class_id, class_name in enumerate(classes) if class_id in random_class_ids or class_id in random_class_ids_tests]

# Fix class IDs in train.txt, val.txt, and test.txt based on the new order
# class_id_mapping = {old_id: new_id for new_id, old_id in enumerate(random_class_ids, start=0)}
# Fix class IDs in train.txt, val.txt, and test.txt based on the new order
class_id_mapping = {old_id: new_id for new_id, old_id in enumerate(random_class_ids, start=0)}
class_id_mapping_test = {old_id: new_id for new_id, old_id in enumerate(random_class_ids_tests, start=len(random_class_ids))}


train_lines = [f"{line.split()[0]} {line.split()[1]} {class_id_mapping[int(line.split()[2])]} \n" for line in train_lines]
val_lines = [f"{line.split()[0]} {line.split()[1]} {class_id_mapping[int(line.split()[2])]} \n" for line in val_lines]
test_lines = [f"{line.split()[0]} {line.split()[1]} {class_id_mapping[int(line.split()[2])]} \n" if int(line.split()[2]) in class_id_mapping else f"{line.split()[0]} {line.split()[1]} {class_id_mapping_test[int(line.split()[2])]} \n" for line in test_lines]

# Write the updated train.txt, val.txt, and test.txt
with open(output_train_file, 'w') as train_file, open(output_val_file, 'w') as val_file, open(output_test_file, 'w') as test_file:
    train_file.writelines(train_lines)
    val_file.writelines(val_lines)
    test_file.writelines(test_lines)


# Convert the list to a string
list_str = str(classes)

# Replace square brackets with curly braces
list_str = list_str.replace('[', '{').replace(']', '}')

# Write the updated classes.txt
with open(output_classes_file, 'w') as classes_file:
    classes_file.write(list_str)
