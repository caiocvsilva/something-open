import sys

def count_class_ids(file_path):
    class_id_counts = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id = int(line.strip().split()[-1])
            if class_id in class_id_counts:
                class_id_counts[class_id] += 1
            else:
                class_id_counts[class_id] = 1

    # sort by class_id
    class_id_counts = dict(sorted(class_id_counts.items(), key=lambda item: item[0]))

    return class_id_counts

def display_count_class_ids(file_path, split_name):
    class_id_counts = count_class_ids(file_path)
    print(f"Split: {split_name}")
    for class_id, count in class_id_counts.items():
        print(f"Class ID: {class_id}, Count: {count}")

train_file_path = sys.argv[1]
val_file_path = sys.argv[2]
test_file_path = sys.argv[3]

display_count_class_ids(train_file_path, 'train')
display_count_class_ids(val_file_path, 'val')
display_count_class_ids(test_file_path, 'test')


# for class_id, count in class_id_counts.items():
#     print(f"Class ID: {class_id}, Count: {count}")