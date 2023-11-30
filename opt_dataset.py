#  python opt_dataset.py labels/train.json labels/validation.json    
    # scenario   verbs   nouns
    # closed      K       K
    # UNKV        K       U
    # KNUV        U       K
    # UNUV        U       U
    # FULL        VERB+NOUN atomic


import json
import sys
import pandas as pd
import random
import nltk
from itertools import chain
from nltk.stem import WordNetLemmatizer
import re
from spellchecker import SpellChecker
import os


nltk.download('averaged_perceptron_tagger')   # Downloading the required NLTK model
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
p = WordNetLemmatizer()

json_file_name = 'misspelled_nouns.json'
with open(json_file_name, 'r') as json_file:
    misspelled_nouns = json.load(json_file)


json_file_name = 'corrected_nouns.json'
with open(json_file_name, 'r') as json_file:
    corrected_nouns = json.load(json_file)


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



# fucntion that receives two arrays fo strings, called unknown_nouns and unknown_verbs, to a json file called unknown_choices.json
def save_unknown_choices(unknown_templates, unknown_nouns, unknown_verbs):
    # Create a dictionary with the two arrays
    unknown_choices = {'unknown_templates': unknown_templates,'unknown_nouns': unknown_nouns, 'unknown_verbs': unknown_verbs}
    # Create a json file with the dictionary
    with open('unknown_choices.json', 'w') as json_file:
        json.dump(unknown_choices, json_file)


def unique_verb_noun(pool):
    """
    Returns a list of unique strings in the 'verb+noun' column of a Pandas DataFrame.
    """
    unique = pool['verb+noun'].unique()
    return list(unique)

# function that checks if a string is composed of letters (to remove numbers and special characters)
def is_composed_of_letters(input_string):
    pattern = r'^[a-zA-Z]+$'
    return re.match(pattern, input_string) is not None

# A function that adds a new column to the dataframe pool called nouns
# for each existing row, the function will extract the nouns in the column placeholders
# and add to the new column
def extract_nouns(row):
    blocked_strings = ['..', '.40', '/', '\\', ']','%', '‘','’']
    text_array = row['placeholders'] #Replace 'placeholders' with the actual name of the column containing your text data
    nouns = []
    for text in text_array:
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if is_composed_of_letters(token)]
        tokens = filtered_tokens
        tagged = nltk.pos_tag(tokens)
        
        # iterate through the tagged words list and append singular nouns to the nouns list
        for word, pos in tagged:
            if (pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS' or pos == 'VB') and (not word.isnumeric() and (word not in blocked_strings) and ("'" not in word) and (len(word) > 1)):
                # lemmatize the word to its singular form
                word = word.lower()
                word_s = p.lemmatize(word, pos='n')
                # if the word is not changed into a singular form, keep it as it is
                if not word_s:
                    singular_noun = word
                else:
                    singular_noun = word_s
                if singular_noun in misspelled_nouns and corrected_nouns[misspelled_nouns.index(singular_noun)] is not None:
                    nouns.append(corrected_nouns[misspelled_nouns.index(singular_noun)])
                else:
                    nouns.append(singular_noun)
    return nouns

# A function that adds a new column to the dataframe pool called verbs
# for each existing row, the function will extract the verbs in the column template
# and add to the new column
def extract_verbs(row):
    text= row['template'] #Replace 'template' with the actual name of the column containing your text data
    verbs = []
    
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if is_composed_of_letters(token)]
    tokens = filtered_tokens
    tagged = nltk.pos_tag(tokens)
    verbs = [word.lower() for word, pos in tagged if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ')and (not word.isnumeric())]
    
    return verbs

def add_nouns_verb_column(pool):
    pool['nouns'] = pool.apply(extract_nouns, axis=1)
    pool['verbs'] = pool.apply(extract_verbs, axis=1)

# Function that chooses x random rows from the dataframe pool
def choose_random_verb_nouns(unique_verb_noun, x):
    all_verb_noun = []
    past_index = []
    #  loop x times chosing one row at a time
    for i in range(x):
        while True:
            random_index = random.randint(0, len(unique_verb_noun)-1)
            if random_index not in past_index:
                past_index.append(random_index)
                break
        templates = unique_verb_noun[random_index]
        all_verb_noun.append(templates)
    return all_verb_noun

# Function that chooses x random rows from the dataframe pool
def choose_random_nouns(pool, x):
    all_nouns = []
    past_index = []
    count_all_nouns=0
    #  loop x times chosing one row at a time
    while True:
        while True:
            random_index = random.randint(0, len(pool)-1)
            nouns = pool['nouns'][random_index]
            # Flatten the nested list into a single list
            if random_index not in past_index:
                past_index.append(random_index)
                break
        all_nouns.append(nouns)
        count_all_nouns=len(set([item for sublist in all_nouns for item in sublist if item not in ['', None]]))
        if count_all_nouns >= x:
            break
    return list(set([item for sublist in all_nouns for item in sublist if item not in ['', None]]))[:x]

def choose_random_verbs(pool, x):
    all_verbs = []
    past_index = []
    count_all_verbs=0
    #  loop x times chosing one row at a time
    while True:
        while True:
            random_index = random.randint(0, len(pool)-1)
            verbs = pool['verbs'][random_index]
            if random_index not in past_index:
                past_index.append(random_index)
                break
        all_verbs.append(verbs)
        count_all_verbs=len(set([item for sublist in all_verbs for item in sublist if item not in ['', None, ']', '[']]))
        if count_all_verbs >= x:
            break
    return list(set([item for sublist in all_verbs for item in sublist if item not in ['', None, ']', '[']]))[:x]


def create_json_stats(train_df, val_df, test_df, name_scenario):
    list_train_nouns_classes = list(set([item for sublist in train_df['nouns'] for item in sublist if item not in ['', None]]))
    list_train_verbs_classes = list(set([item for sublist in train_df['verbs'] for item in sublist if item not in ['', None]]))
    list_val_nouns_classes = list(set([item for sublist in val_df['nouns'] for item in sublist if item not in ['', None]]))
    list_val_verbs_classes = list(set([item for sublist in val_df['verbs'] for item in sublist if item not in ['', None]]))
    list_test_nouns_classes = list(set([item for sublist in test_df['nouns'] for item in sublist if item not in ['', None]]))
    list_test_verbs_classes = list(set([item for sublist in test_df['verbs'] for item in sublist if item not in ['', None]]))

    print('number of train classes noun: ', len(list_train_nouns_classes))
    print('list of train classes noun: ', list_train_nouns_classes)
    print('number of train classes verb: ', len(list_train_verbs_classes))
    print('list of train classes verb: ', list_train_verbs_classes)
    print('number of val classes noun: ', len(list_val_nouns_classes))
    print('list of val classes noun: ', list_val_nouns_classes)
    print('number of val classes verb: ', len(list_val_verbs_classes))
    print('list of val classes verb: ', list_val_verbs_classes)
    print('number of test classes noun: ', len(list_test_nouns_classes))
    print('list of test classes noun: ', list_test_nouns_classes)
    print('number of test classes verb: ', len(list_test_verbs_classes))
    print('list of test classes verb: ', list_test_verbs_classes)

    # save json with stats for name_scenario (containing size of train and test set, number of classes  and the classes in train and test set)
    stats = {'size_train': len(train_df), 'size_val': len(val_df) ,'size_test': len(test_df), 'number_classes_train_noun': len(list_train_nouns_classes), 'number_classes_train_verb': len(list_train_verbs_classes), 'number_classes_val_noun': len(list_val_nouns_classes),'number_classes_val_verb': len(list_val_verbs_classes),'number_classes_test_noun': len(list_test_nouns_classes), 'number_classes_test_verb': len(list_test_verbs_classes), 'list_classes_train_noun': list_train_nouns_classes, 'list_classes_train_verb': list_train_verbs_classes, 'list_classes_val_noun': list_val_nouns_classes, 'list_classes_val_verb': list_val_verbs_classes,'list_classes_test_noun': list_test_nouns_classes, 'list_classes_test_verb': list_test_verbs_classes}
    # with open(name_scenario+'_stats.json', 'w') as fp:
    #     json.dump(stats, fp)
    stats_df = pd.DataFrame([stats])

    # Save the DataFrame to a CSV file
    stats_df.to_csv(name_scenario+'_stats.csv', index=False)


def split_train_val(df):
    # If the group has less than 5 rows, put all rows in the train set
    if len(df) < 5:
        return df, pd.DataFrame(), pd.DataFrame()
    # Otherwise, randomly select 20% of the rows for the validation set
    else:
        val_df = df.sample(frac=0.2, random_state=42)
        df =  df.drop(val_df.index)
        test_df = df.sample(frac=0.2, random_state=42)
        train_df = df.drop(test_df.index)
        return train_df, val_df, test_df



def process_labels(pool, unknown_verb_noun=None, unknown_nouns=None, unknown_verbs=None):
    if unknown_verb_noun:
        # Filter the DataFrame to exclude rows with templates in the unknown_templates array
        known_labels = pool[~pool['verb+noun'].isin(unknown_verb_noun)]
        unknown_labels = pool[pool['verb+noun'].isin(unknown_verb_noun)]
    else:
        # Flatten unknown_nouns and unknown_verbs list
        if unknown_nouns:
            flat_unknown_nouns = unknown_nouns
        else:
            flat_unknown_nouns = []
        if unknown_verbs:
            flat_unknown_verbs = unknown_verbs
        else:
            flat_unknown_verbs = []

        # Filter pool dataframe based on unknowns
        known_labels = pool[~pool['nouns'].apply(lambda x: any(item for item in x if item in flat_unknown_nouns))]
        known_labels = known_labels[~known_labels['verbs'].apply(lambda x: any(item for item in x if item in flat_unknown_verbs))]
        unknown_labels = pool[pool['nouns'].apply(lambda x: any(item for item in x if item in flat_unknown_nouns))]
        if len(unknown_labels) == 0:  #check for knuv case
            unknown_labels = pool
        unknown_labels = unknown_labels[unknown_labels['verbs'].apply(lambda x: any(item for item in x if item in flat_unknown_verbs))]

    return known_labels, unknown_labels


# Scenarios
def generate_scenario(known_labels, unknown_labels, un = False, uv = False):
    tag=''
    # Find all unique nouns in known_labels and convert to set
    if un:
        tag += 'un'
        known_verbs = set(known_labels['verbs'].explode().unique())
        # Create a boolean mask of unknown_labels rows where any verb is not in known_verbs
        mask = ~(unknown_labels['verbs'].explode().isin(known_verbs)).groupby(level=0).any()

        # Filter out the rows where the mask is True
        unknown_labels = unknown_labels.loc[~mask]
    else:
        tag += 'kn'

    if uv:
        tag += 'uv'
        known_nouns = set(known_labels['nouns'].explode().unique())
        # Create a boolean mask of unknown_labels rows where any nouns is not in known_nouns
        mask = ~(unknown_labels['nouns'].explode().isin(known_nouns)).groupby(level=0).any()
        
        # Filter out the rows where the mask is True
        unknown_labels = unknown_labels.loc[~mask]
    else:
        tag += 'kv'

    if tag == 'knkv':
        tag = 'full'

    
     # Group the DataFrame by 'verb+noun' and apply the split function to each group
    groups = known_labels.groupby('verb+noun').apply(split_train_val)

    # Concatenate the results to create the final train and validation DataFrames
    train_df = pd.concat([g[0] for g in groups])
    val_df = pd.concat([g[1] for g in groups])
    test_df = pd.concat([g[2] for g in groups])

    print('['+tag+'] size of train_df: ', len(train_df))
    print('['+tag+'] size of val_df: ', len(val_df))
    print('['+tag+'] k - size of test_df: ', len(test_df))
    
    # Append the unknown_labels dataframe to the test set
    test_df = pd.concat([test_df, unknown_labels], ignore_index=True)
    print('['+tag+'] k+u - size of test_df: ', len(test_df))

    create_json_stats(train_df, val_df, test_df, tag)
    
    # Save the train and test dataframes to JSON files
    train_df.to_csv(tag+'_train.csv', index=False)
    val_df.to_csv(tag+'_val.csv', index=False)
    test_df.to_csv(tag+'_test.csv', index=False)

    generate_annotation_and_class_files([train_df, val_df, test_df], tag)



#Generate txt files for annotations (mmaction)
def generate_annotation_and_class_files(dfs, scenario):
    # Initialize a set to collect unique verb_noun
    unique_verbs_nouns = []

    for type_df in dfs:
        # Extract unique verb_noun from the data and add to the set
        for _, item in type_df.iterrows():
            unique_verbs_nouns.append(item['verb+noun'])

    unique_verbs_nouns = set(unique_verbs_nouns)

     # Create a mapping of nouns to class numbers
    verb_noun_to_class = {verb_noun: idx for idx, verb_noun in enumerate(unique_verbs_nouns)}

    # Define the path for annotation
    dataset_path = '/shared/datasets/something/20bn-something-something-v2/'
    class_file = 'class_'+scenario+'.txt'

    for idx, type_df in enumerate(dfs):
        if idx == 0:
            type = 'train'
        elif idx == 1:
            type = 'val'
        else:
            type = 'test'

        annotation_file = 'ann_'+scenario+'_'+type+'.txt'

        with open(annotation_file, 'w') as ann_file:
            for _, item in type_df.iterrows():
                video_id = item['id']
                verb_nouns = item['verb+noun']
                class_number = verb_noun_to_class[verb_nouns]
                video_path = os.path.join(dataset_path, video_id)

                # Write annotation file
                ann_file.write(f'{video_path} {class_number}\n')


    # Create a shared class file
    with open(class_file, 'w') as cls_file:
        class_names = unique_verbs_nouns
        cls_file.write(str(class_names))

if __name__ == "__main__":
    pool = read_jsons()
    add_nouns_verb_column(pool)
    pool.loc[~pool['nouns'].astype(bool), 'nouns'] = pool['placeholders'] #replace None noun with original placeholder
    pool['verb+noun'] = pool['verbs'].str[0] + ' ' + pool['nouns'].str[0]
    counts = pool['verb+noun'].value_counts() # count how many examples each class
    pool_filtered = pool[pool['verb+noun'].isin(counts[counts >= 100].index)] # remove all class less 100 examples
    capped_pool = pool_filtered.groupby('verb+noun').apply(lambda x: x.sample(min(len(x), 100))).reset_index(drop=True) # cap each class to 100 examples
    pool = capped_pool

    # Choose unknowns
    uni_verb_noun = unique_verb_noun(pool)
    print('size uni_verb_noun: ', len(uni_verb_noun))
    unknown_verb_noun = choose_random_verb_nouns(uni_verb_noun, 10)
    print(unknown_verb_noun)
    unknown_nouns = choose_random_nouns(pool, 10)
    print(unknown_nouns)
    unknown_verbs = choose_random_verbs(pool, 10)
    print(unknown_verbs)
    save_unknown_choices(unknown_verb_noun, unknown_nouns, unknown_verbs)

    # Create json Scenarios
    # full
    known_labels, unknown_labels = process_labels(pool, unknown_verb_noun=unknown_verb_noun)
    generate_scenario(known_labels, unknown_labels)
    # unkv
    known_labels, unknown_labels = process_labels(pool, unknown_nouns=unknown_nouns)
    generate_scenario(known_labels, unknown_labels, un=True)
    # knuv
    known_labels, unknown_labels = process_labels(pool, unknown_verbs=unknown_verbs)
    generate_scenario(known_labels, unknown_labels, uv=True)
    # unuv
    known_labels, unknown_labels = process_labels(pool, unknown_nouns=unknown_nouns, unknown_verbs=unknown_verbs)
    generate_scenario(known_labels, unknown_labels, un=True, uv=True)

