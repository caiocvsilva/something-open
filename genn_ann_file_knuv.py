import json
import os

def count_frames_in_directory(directory):
    # Count the number of frames in the given directory
    frame_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]  # Change the extension accordingly
    return len(frame_files)

def generate_annotation_and_class_files(json_files, annotation_file, class_file):
    # Initialize a set to collect unique nouns
    unique_verbs_nouns = []

    # Iterate through each JSON file
    for json_file in json_files:
        # Read the JSON file
        with open(json_file, 'r') as json_data:
            data = json.load(json_data)
        
        # Extract unique nouns from the JSON data and add to the set
        for item in data['combined_objects']:
            unique_verbs_nouns.append(item['verb+noun'])
          
    
    unique_verbs_nouns = set(unique_verbs_nouns)
    # Create a mapping of nouns to class numbers
    verb_noun_to_class = {verb_noun: idx for idx, verb_noun in enumerate(unique_verbs_nouns)}
    
    print('num of classes: ', len(unique_verbs_nouns))
    #print('max idx: ', max(verb_noun_to_class))
    #x = input()

    # Define the path for annotation
    annotation_path = '/shared/datasets/something/rawframes/'

    # Create annotation files
    for idx, json_file in enumerate(json_files):
        with open(json_file, 'r') as json_data, open(annotation_file[idx], 'w') as ann_file:
            data = json.load(json_data)
            for item in data['combined_objects']:
                video_id = item['id']
                verb_nouns = item['verb+noun']
                class_number = verb_noun_to_class[verb_nouns]
                if class_number > 26339:
                    print('class: ', verb_nouns)
                    print('cnum: ',class_number)
                directory = os.path.join('/shared/datasets/something/rawframes/', video_id)
                num_files = count_frames_in_directory(directory)

                # Write annotation file
                ann_file.write(f'{annotation_path}{video_id} {num_files} {class_number}\n')

    # Create a shared class file
    with open(class_file, 'w') as cls_file:
        class_names = unique_verbs_nouns
        cls_file.write(str(class_names))

if __name__ == "__main__":
    # Replace 'json1.json' and 'json2.json' with your JSON file names
    generate_annotation_and_class_files(['knuv_train_fix.json', 'knuv_test_fix.json'], ['ann_knuv_train.txt', 'ann_knuv_test.txt'], 'class_knuv.txt')

