import sys
import pickle
import re

log_file_path = sys.argv[1]
top1_all_train = 0.0
top5_all_train = 0.0
iteration = 0
epoch = 0
arr_top1 = []
arr_top5 = []

with open(log_file_path, "r") as file:
    for line in file:
        top1_matches = re.findall(r'top1_acc_train: (\d+\.\d+)', line)
        top5_matches = re.findall(r'top5_acc_train: (\d+\.\d+)', line)
        
        for match in top1_matches:
            top1_acc_train = float(match)
            top1_all_train += top1_acc_train
            iteration += 1

        for match in top5_matches:
            top5_acc_train = float(match)
            top5_all_train += top5_acc_train

        if "Epoch(train)" in line:
            if iteration > 0:
                top1_all_train /= iteration
                top5_all_train /= iteration
                print(f"Epoch {epoch}: top1_acc_train = {top1_all_train}, top5_acc_train = {top5_all_train}")
                arr_top1.append(top1_acc_train)
                arr_top5.append(top5_acc_train)

                top1_all_train = 0.0
                top5_all_train = 0.0
                iteration = 0
                epoch += 1

with open("arr_top1.pkl", "wb") as file:
    pickle.dump(arr_top1, file)

with open("arr_top5.pkl", "wb") as file:
    pickle.dump(arr_top5, file)