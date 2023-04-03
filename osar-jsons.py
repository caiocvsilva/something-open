
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
        verbs = [word for word, pos in tagged_tokens if pos.startswith('V')]

        # append all the verbs
        all_verbs.append(verbs)
    return all_verbs

# Choose 10% of the nouns at random, but no two items can share a word
def choose_nouns(pool):

    core_check = True
    chosen_noun=list()
    percent_placeholder = 0.0001; 
    
    nouns = get_nouns(pool)
    num_nouns = len(nouns)

    train_nouns = nouns
    random_indexes_past=[]

    print(num_nouns)

    # generate num_nouns/10 random numbers in the range 0 to num_nouns
    for i in range(int(num_nouns*percent_placeholder)):
        core_check = True
        while(core_check):
            core_check=False
            place_dict=[]
            nouns_index = random.randint(0, num_nouns-1)
            if nouns_index not in random_indexes_past:
                random_indexes_past.append(nouns_index)
                chosen_noun.append(nouns[nouns_index])
            else:
                core_check = True
                continue
            print('len of nouns: ', len(nouns))
            print('nouns_index: ', nouns_index)
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

# Choose 10% of the verbs at random, similar to what was done for nouns
def choose_verbs(pool):
    
    core_check = True
    chosen_verbs=list()
    percent_verbs = 0.1; 
    
    verbs = get_verbs(pool)
    num_verbs = len(verbs)

    train_verbs = verbs
    random_indexes_past=[]

    print(num_verbs)

    # generate num_verbs/10 random numbers in the range 0 to num_verbs
    for i in range(int(num_verbs*percent_verbs)):
        core_check = True
        while(core_check):
            core_check=False
            verbs_index = random.randint(0, num_verbs-1)
            if verbs_index not in random_indexes_past:
                random_indexes_past.append(verbs_index)
                chosen_verbs.append(verbs[verbs_index])
            else:
                core_check = True
                continue
            print('len of verbs: ', len(verbs))
            print('verbs_index: ', verbs_index)
            for j, val in enumerate(verbs[verbs_index]):
                train_verbs.remove(verbs[verbs_index])
                num_verbs = len(train_verbs)

                core_check = False
                # Extracting all verbs from first_list
                verbs_to_check = [word[0] for word in nltk.pos_tag([word for sublist in val for word in sublist]) if word[1].startswith('V')]
                if not any(word in sublist for sublist in train_verbs for word in verbs_to_check):
                    chosen_verbs.append(val)
                else:
                    core_check = True
                    break

    # for i in range(int(num_verbs*percent_verbs)):
    #     chosen_verbs.append(verbs[random_indexes_past[i]])

    print("verbs chosen: ", chosen_verbs)


if __name__ == "__main__":
    pool = read_jsons()
    choose_nouns(pool)
    choose_verbs(pool)