
import json
import sys
import pandas as pd
import random
import nltk

nltk.download('averaged_perceptron_tagger')   # Downloading the required NLTK model




# Open two jsons given by the user as commmand line arguments
# combine the data from both jsons
def read_jsons():

    pool = []
    # Check if the user gave two arguments
    if len(sys.argv) != 3:
        print("Usage: python osar-jsons.py <json1> <json2>")
        sys.exit(1)

    # Open the first json
    with open(sys.argv[1]) as json_file:
        data1 = pd.json_normalize(json.load(json_file))
        pool.append(data1)

    # Open the second json
    with open(sys.argv[2]) as json_file:
        data2 = pd.json_normalize(json.load(json_file))
        pool.append(data2)

    return pd.concat(pool, ignore_index=True)


# Get all items from pool of jsons in the 'placeholders' index
def get_placeholders(pool):
    return [list(x) for x in set(tuple(x) for x in pool['placeholders'])]

# Get all items from pool of jsons in the 'template' index
def get_template(pool):
    return pool['template'].unique()

# Choose 10% of the placeholders at random, but no two items can share a word
def choose_placeholders(pool):

    core_check = True
    chosen_placeholder=list()
    chosen_template=list()
    percent_placeholder = 0.0001; 
    percent_template = 0.1;

    
    placeholders = get_placeholders(pool)
    templates = get_template(pool)
    num_placeholders = len(placeholders)
    num_templates = len(templates)

    train_placeholders = placeholders
    random_indexes_past=[]

    # list of articles in english language



    print(num_placeholders)
    print(num_templates)
    # print("Train Placeholders: ", train_placeholders)

    # generate num_placeholders/10 random numbers in the range 0 to num_placeholders
    for i in range(int(num_placeholders*percent_placeholder)):
        core_check = True
        while(core_check):
            core_check=False
            place_dict=[]
            placeholders_index = random.randint(0, num_placeholders-1)
            if placeholders_index not in random_indexes_past:
                random_indexes_past.append(placeholders_index)
            else:
                core_check = True
                continue
            for j, val in enumerate(placeholders[placeholders_index]):
                train_placeholders.remove(placeholders[placeholders_index])

                core_check = False
                # Extracting all nouns from first_list
                nouns_to_check = [word[0] for word in nltk.pos_tag([word for sublist in val for word in sublist]) if word[1].startswith('N')]
                if not any(word in sublist for sublist in train_placeholders for word in nouns_to_check):
                    place_dict.append(val)
                else:
                    core_check = True
                    break

    for i in range(int(num_placeholders*percent_placeholder)):
        chosen_placeholder.append(placeholders[random_indexes_past[i]])

    print("Placeholders chosen: ", chosen_placeholder)

if __name__ == "__main__":
    pool = read_jsons()
    choose_placeholders(pool)