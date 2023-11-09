
import pandas as pd
import sys

path_file = sys.argv[1]

# Read the txt file into a list of strings
with open(path_file, 'r') as f:
    lines = f.readlines()

# Split each line into path and label
data = [line.strip().split() for line in lines]

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['path', 'label'])

# Get unique labels and their counts
unique_labels = df['label'].value_counts()

# Get labels with count higher than 10
high_count_labels = unique_labels[unique_labels > 10].index

# Split the dataframe into train and val
val = df[df['label'].isin(high_count_labels)].sample(frac=0.2)
# train has all the rows not in val
train = df[~df.isin(val)].dropna()

print(train.shape)
print(val.shape)


train_output_path = path_file.split('.')[0]+'v2.txt'
val_output_path = path_file.split('.')[0].replace('train', 'val')+'v2.txt'

# Save the train and val dataframes to csv
train.to_csv(train_output_path, index=False, sep=' ')
val.to_csv(val_output_path, index=False, sep=' ')



