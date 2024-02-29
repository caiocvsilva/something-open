import sys

scenarios = ["knuv", "unkv", "unuv", "full"]

server = ["lab", "gaivi"]

for i, val in enumerate(server):
    print(val)
    for j, val2 in enumerate(scenarios):
        print(val2)
        if val == "lab":
            train_txt = "/shared/something-open/video/video_ann_"+scenarios[j]+"_train_10_classes.txt"
            test_txt = "/shared/something-open/video/video_ann_"+scenarios[j]+"_test_10_classes.txt"
        elif val == "gaivi":
            train_txt = "/shared/something-open/gaivi/gaivi_video_ann_"+scenarios[j]+"_train_10_classes.txt"
            test_txt = "/shared/something-open/gaivi/gaivi_video_ann_"+scenarios[j]+"_test_10_classes.txt"

        # Read the first text file
        with open(train_txt, 'r') as file:
            train_data = {line.split()[1] for line in file}

        # Read the second text file
        with open(test_txt, 'r') as file:
            test_data = {line.split()[1] for line in file}

        # remove duplicates in train_data
        train_data = set(train_data)
        # prins number of classes in train_data
        print(len(train_data))
        print(train_data)

        test_data = set(test_data)
        # prins number of classes in test_data
        print(len(test_data))
        print(test_data)

        # Extract the IDs from the second text file that are not in the first text file
        unknown_ids = test_data.difference(train_data)
        print(len(unknown_ids))
        print(unknown_ids)

        # Write the new IDs to a file
        with open('unknown_classes_'+scenarios[j]+'_10_classes.txt', 'w') as file:
            ids_list = [id for id in unknown_ids]
            file.write(str(ids_list))
