
import json
import sys
import pandas as pd
import random
import nltk
from sklearn.model_selection import train_test_split


nltk.download('averaged_perceptron_tagger')   # Downloading the required NLTK model
nltk.download('punkt')




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

# A function that adds a new column to the dataframe pool called nouns
# for each existing row, the function will extract the nouns in the column placeholders
# and add to the new column
def extract_nouns(row):
    text_array = row['placeholders'] #Replace 'placeholders' with the actual name of the column containing your text data
    nouns = []
    for text in text_array:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns.extend([word.lower() for word, pos in tagged if (pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS')])
    return nouns


# A function that adds a new column to the dataframe pool called verbs
# for each existing row, the function will extract the verbs in the column template
# and add to the new column
def extract_verbs(row):
    text= row['template'] #Replace 'template' with the actual name of the column containing your text data
    verbs = []
    
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    verbs = [word.lower() for word, pos in tagged if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ')]
    return verbs

def add_nouns_verb_column(pool):
    pool['nouns'] = pool.apply(extract_nouns, axis=1)
    pool['verbs'] = pool.apply(extract_verbs, axis=1)


# Function that chooses x random rows from the dataframe pool
def choose_random_nouns(pool, x):
    all_nouns = []
    past_index = []
    #  loop x times chosing one row at a time
    for i in range(x):
        while True:
            random_index = random.randint(0, len(pool)-1)
            if random_index not in past_index:
                past_index.append(random_index)
                break
        nouns = pool['nouns'][random_index]
        all_nouns.append(nouns)
    return all_nouns

def choose_random_verbs(pool, x):
    all_verbs = []
    past_index = []
    #  loop x times chosing one row at a time
    for i in range(x):
        while True:
            random_index = random.randint(0, len(pool)-1)
            if random_index not in past_index:
                past_index.append(random_index)
                break
        verbs = pool['verbs'][random_index]
        all_verbs.append(verbs)
    return all_verbs

# Function that creates the train and test jsons for
# Unknown nouns and known verbs
def unknown_noun_known_verb(pool, unknown_nouns):
    known_labels = pool[~pool['nouns'].apply(lambda x: bool(set(x) & set(unknown_nouns)))]
    unknown_labels = pool[pool['nouns'].apply(lambda x: bool(set(x) & set(unknown_nouns)))]

    # # Find all unique verbs in known_labels
    # known_verbs = set(known_labels['verbs'].sum())
    
    # # Iterate over each row of unknown_labels
    # for i, row in unknown_labels.iterrows():
    #     # Check if any verb in current row is not in known_verbs
    #     if any(v not in known_verbs for v in row['verbs']):
    #         # If found, delete the current row from unknown_labels
    #         unknown_labels.drop(i, inplace=True)
    
    # Find all unique verbs in known_labels and convert to set
    known_verbs = set(known_labels['verbs'].explode().unique())
    
    # Create a boolean mask of unknown_labels rows where any verb is not in known_verbs
    mask = ~(unknown_labels['verbs'].explode().isin(known_verbs)).groupby(level=0).any()
    
    # Filter out the rows where the mask is True
    unknown_labels = unknown_labels.loc[~mask]
    
    # Split the known_labels dataframe into training and testing sets
    train_df = known_labels.sample(frac=0.7, random_state=42)
    test_df = known_labels.drop(train_df.index)
    
    # Append the unknown_labels dataframe to the test set
    test_df = test_df.append(unknown_labels, ignore_index=True)
    
    # Save the train and test dataframes to JSON files
    train_df.to_json('unkv_train.json', orient='records', lines=True)
    test_df.to_json('unkv_test.json', orient='records', lines=True)
    


    


if __name__ == "__main__":
    pool = read_jsons()
    add_nouns_verb_column(pool)
    print(pool)
    unknown_nouns = choose_random_nouns(pool,10)
    print(unknown_nouns)
    unknown_verbs = choose_random_verbs(pool,10)
    print(unknown_verbs)
    unknown_noun_known_verb(pool, unknown_nouns)


    # add_verbs_column(pool)
    # print(pool)

    # unknwon_nouns = choose_nouns(pool)
    # unknown_verbs = choose_verbs(pool)
    # unknown_noun_known_verb(pool, unknwon_nouns)
    # # unknown_verb_known_noun(pool, unknown_verbs)
    