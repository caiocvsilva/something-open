import json

def generate_annotation_and_class_files(json_files, annotation_file, class_file):
    # Initialize a set to collect unique nouns
    unique_nouns = set()

    # Iterate through each JSON file
    for json_file in json_files:
        # Read the JSON file
        with open(json_file, 'r') as json_data:
            data = json.load(json_data)
        
        # Extract unique nouns from the JSON data and add to the set
        for item in data['combined_objects']:
            # print('item: ', item)
            unique_nouns.update(item['nouns'])

    # Create a mapping of nouns to class numbers
    noun_to_class = {noun: idx for idx, noun in enumerate(unique_nouns)}

    # Define the path for annotation
    annotation_path = '/home/caio/Documents/datasets/something/rawframes/'

    # Create annotation files
    for idx, json_file in enumerate(json_files):
        with open(json_file, 'r') as json_data, open(annotation_file[idx], 'w') as ann_file:
            data = json.load(json_data)
            for item in data['combined_objects']:
                video_id = item['id']
                nouns = item['nouns']
                class_number = noun_to_class[nouns[0]]

                # Write annotation file
                ann_file.write(f'{annotation_path}{video_id} {class_number}\n')

    # Create a shared class file
    with open(class_file, 'w') as cls_file:
        class_names = unique_nouns
        cls_file.write(str(class_names))

if __name__ == "__main__":
    # Replace 'json1.json' and 'json2.json' with your JSON file names
    generate_annotation_and_class_files(['unkv_train_fix.json', 'unkv_test_fix.json'], ['ann_unkv_train.txt', 'ann_unkv_test.txt'], 'class_unkv.txt')

