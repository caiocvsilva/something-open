import sys
import pickle

log_file_path = sys.argv[1]
top1_all_train = 0.0
top5_all_train = 0.0
iteration = 0
epoch = 0
arr_top1 = []
arr_top5 = []

with open(log_file_path, "r") as file:
    start_reading = False
    for line in file:
        if "mmengine - INFO - Checkpoints will be saved to" in line:
            start_reading = True
            continue
        if start_reading:
            if "Epoch(train)" in line:
                if iteration > 0:
                    top1_all_train /= iteration
                    top5_all_train /= iteration
                    print(f"Iteration {epoch}: top1_acc_train = {top1_all_train}, top5_acc_train = {top5_all_train}")
                    arr_top1.append(top1_acc_train)
                    arr_top5.append(top5_acc_train)

                    top1_all_train = 0.0
                    top5_all_train = 0.0
                    iteration = 0
                    epoch += 1
                continue
            if "top1_acc_train" in line:
                top1_acc_train = float(line.split(":")[1].strip())
                top1_all_train += top1_acc_train
                iteration += 1
            elif "top5_acc_train" in line:
                if "top1_acc_train" in line:
                    line1, line2 = line.split("top1_acc_train")
                    top1_acc_train = float(line2.split(":")[1].strip())
                    top1_all_train += top1_acc_train
                    iteration += 1
                else:
                    line1 = line
                top5_acc_train = float(line1.split(":")[1].strip())
                top5_all_train += top5_acc_train

with open("arr_top1.pkl", "wb") as file:
    pickle.dump(arr_top1, file)

with open("arr_top5.pkl", "wb") as file:
    pickle.dump(arr_top5, file)
                
                
