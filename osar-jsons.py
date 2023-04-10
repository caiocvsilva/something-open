
import json
import sys
import pandas as pd
import random
import nltk
from sklearn.model_selection import train_test_split


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


# Get all items from pool of jsons in the 'nouns' index
def get_nouns(pool):
    return [list(x) for x in set(tuple(x) for x in pool['placeholders'])]

# Get all items from pool of jsons in the 'template' index
def get_verbs(pool):
    templates = pool['template'].unique()
    all_verbs = []

    for sentence in templates:

        tokens = nltk.word_tokenize(sentence)

        # Tag the tokens with their part of speech
        tagged_tokens = nltk.pos_tag(tokens)

        # Extract the verbs from the tagged tokens
        verbs = [word.lower() for word, pos in tagged_tokens if pos.startswith('V')]

        # append all the verbs
        all_verbs.append(verbs)
    return all_verbs

# Choose 10% of the nouns at random, but no two items can share a word
def choose_nouns(pool):
    # Set initial variables for tracking and output
    core_check = True
    chosen_noun=list()
    percent_placeholder = 0.0001;
    
    # Call get_nouns function to retrieve all nouns from the pool of data
    nouns = get_nouns(pool)
    num_nouns = len(nouns)

    # Create train_nouns list with all initial nouns from noun list.
    train_nouns = nouns
    random_indexes_past=[]

    # Generate num_nouns/10 random numbers in the range 0 to num_nouns
    # Random number is used to select a noun from the nouns list which will be added to chosen_noun list
    for i in range(int(num_nouns*percent_placeholder)):
        core_check = True
        while(core_check):
            # set core_check flag to False so that it can be checked inside inner loop
            core_check=False
            place_dict=[]
            nouns_index = random.randint(0, num_nouns-1)
            # Check whether the randomly selected noun already exists in past selections.
            if nouns_index not in random_indexes_past:
                random_indexes_past.append(nouns_index)
                chosen_noun.append(nouns[nouns_index])
            else:
                core_check = True
                continue
            # Extract all nouns from selected noun and check whether they are present in any other sublist of train_nouns
            for j, val in enumerate(nouns[nouns_index]):
                train_nouns.remove(nouns[nouns_index])
                num_nouns = len(train_nouns)

                core_check = False
                # Extracting all nouns from first_list
                nouns_to_check = [word[0] for word in nltk.pos_tag([word for sublist in val for word in sublist]) if word[1].startswith('N')]
                if not any(word in sublist for sublist in train_nouns for word in nouns_to_check):
                    place_dict.append(val)
                else:
                    core_check = True
                    break

    print("nouns chosen: ", chosen_noun)
    # save all the nouns that were not chosen to a file called known_nouns.txt
    with open('known_nouns.txt', 'w') as f:
        for item in train_nouns:
            f.write("%s\n" % item)
    return chosen_noun

# Choose 10% of the verbs at random, similar to what was done for nouns
def choose_verbs(pool):
    # Set initial variables for tracking and output
    core_check = True
    chosen_verbs=list()
    percent_verbs = 0.1; 
    
    # Call get_verbs function to retrieve all verbs from the pool of data
    verbs = get_verbs(pool)
    num_verbs = len(verbs)

    # Create train_verbs list with all initial verbs from verb list.
    train_verbs = verbs
    random_indexes_past=[]

    # Generate num_verbs/10 random numbers in the range 0 to num_verbs
    # Random number is used to select a verb from the verbs list which will be added to chosen_verbs list
    for i in range(int(num_verbs*percent_verbs)):
        core_check = True
        while(core_check):
            # set core_check flag to False so that it can be checked inside inner loop
            core_check=False
            verbs_index = random.randint(0, num_verbs-1)
            # Check whether the randomly selected verb already exists in past selections.
            if verbs_index not in random_indexes_past:
                random_indexes_past.append(verbs_index)
                for j, val in enumerate(verbs[verbs_index]):
                    train_verbs.remove(verbs[verbs_index])
                    num_verbs = len(train_verbs)

                    core_check = False
                    # Extracting all verbs from first_list
                    verbs_to_check = [word[0] for word in nltk.pos_tag([word for sublist in val for word in sublist]) if word[1].startswith('V')]
                    # Check whether the selected verb exists in chosen_verbs list or not.
                    # Also, check whether any of the verbs_to_check exist in train_verbs or not.
                    if not any(word in sublist for sublist in train_verbs for word in val) and (val not in chosen_verbs):
                        chosen_verbs.append(val)
                    else:
                        core_check = True
                        break
            else:
                core_check = True
                continue

    print("verbs chosen: ", chosen_verbs)
    return chosen_verbs

# function that will receive the pool and the unknown nouns. From this it will generate two json, in the same format as the original json, in the train json it will contain 70% of all the sentences in which any item of placeholders, converted to lowercase, is not equal to any item in the unknown nouns, and in the test json it will contain the remaining 30% of the sentences in which any item of placeholders, converted to lowercase, is not equal to any item in the unknown nouns plus all the sentences in which any item of placeholders, converted to lowercase, is equal to any item in the unknown nouns.
def unknown_noun_known_verb(pool, unknown_nouns):
    # Create two dataframes in the sstructure as the pool to store the known and unknown labels
    known_labels = pool[pool['placeholders'].apply(lambda x: any(any(word in x for word in subarray) for subarray in unknown_nouns))]
    unknown_labels = pool[~pool['placeholders'].apply(lambda x: any(any(word in x for word in subarray) for subarray in unknown_nouns))]

    # train dataframe is a random 70% of the known_labels dataframe
    unkv_train = known_labels.sample(frac=0.7)
    # test dataframe is the known_labels dataframe rows not in train dataframe plus all the rows in the unknown_labels dataframe
    unkv_test = pd.concat([known_labels[~known_labels.isin(unkv_train)].dropna(), unknown_labels], ignore_index=True)
    # unkv_test = unknown_labels
        
    # Save the unkv_train and unkv_test dataframes to json files
    unkv_train.to_json('unkv_train.json', orient='records', lines=True)
    unkv_test.to_json('unkv_test.json', orient='records', lines=True)


if __name__ == "__main__":
    pool = read_jsons()
    unknwon_nouns = choose_nouns(pool)
    unknown_verbs = choose_verbs(pool)
    unknown_noun_known_verb(pool, unknwon_nouns)
    # unknown_verb_known_noun(pool, unknown_verbs)
    