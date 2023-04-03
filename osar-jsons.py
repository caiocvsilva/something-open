
import json
import sys
import pandas as pd
import random




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

    # list of articles in english language



    print(num_placeholders)
    print(num_templates)

    # generate num_placeholders/10 random numbers in the range 0 to num_placeholders
    while(core_check):
        core_check=False
        place_dict=[]
        placeholders_index = [random.randint(0, num_placeholders-1) for i in range(int(num_placeholders*percent_placeholder))]
        for i in range(int(num_placeholders*percent_placeholder)):
            for j, val in enumerate(placeholders[placeholders_index[i]]):
                core_check = False
                if " " in val:
                    exploded = val.split(" ")
                    for e, vale in enumerate(exploded):
                        if vale != 'a' and vale != 'an':
                            if (vale not in place_dict):
                                place_dict.append(vale)
                            else:
                                core_check = True
                                break
                if core_check:
                    break
                else:
                    if (val not in place_dict):
                        place_dict.append(val)
                    else:
                        core_check = True
            if core_check:
                break

    for i in range(int(num_placeholders*percent_placeholder)):
        chosen_placeholder.append(placeholders[placeholders_index[i]])

    print("Placeholders chosen: ", chosen_placeholder)

    core_check = True
    while(core_check):
        core_check=False
        place_dict=[]
        template_index = [random.randint(0, num_templates-1) for i in range(int(num_templates*percent_template))]
        for i in range(int(num_templates*percent_template)):
            for j, val in enumerate(templates[template_index[i]]):
                core_check = False
                if "[something] " in val:
                    exploded = val.split("[something]")
                    for e, vale in enumerate(exploded):
                        if vale != 'a' and vale != 'an' and vale != 'the' and vale != ',' and vale !='and':
                            if (vale not in place_dict):
                                place_dict.append(vale)
                            else:
                                core_check = True
                                break
                if core_check:
                    break
                else:
                    if (val not in place_dict):
                        place_dict.append(val)
                    else:
                        core_check = True
            if core_check:
                break

    for i in range(int(num_templates*percent_template)):
        chosen_template.append(templates[template_index[i]])

    print("template chosen: ", chosen_template)

    
    
    # template_index = [random.randint(0, num_templates) for i in range(num_templates/10)]


    # placeholders_to_choose = set(random.sample((placeholders), int(num_placeholders * 0.1)))
    # templates_to_choose = set(random.sample(sorted(templates), int(num_templates * 0.1)))

    # print("Placeholders to choose: ", placeholders_to_choose)

if __name__ == "__main__":
    pool = read_jsons()
    choose_placeholders(pool)