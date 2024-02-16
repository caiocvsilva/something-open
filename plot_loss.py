import os
import matplotlib.pyplot as plt
import pandas as pd
import sys

def parse_log_file(log_file):
    data = {'iteration': [], 'train_loss': [], 'val_loss': []}
    iteration = 0
    val_loss = 2.3053  # Initialize val_loss variable
    with open(log_file, 'r') as file:
        for line in file:
            if 'Epoch(train)' in line and 'loss:' in line:
                parts = line.split()
                epoch = int(parts[parts.index('Epoch(train)') + 1].split('[')[1].split(']')[0])
                loss = float(parts[parts.index('loss:') + 1])
                iteration += 1
                data['iteration'].append(iteration)
                data['train_loss'].append(loss)
                if val_loss is not None:
                    data['val_loss'].append(val_loss)
            if 'Epoch(val)' in line and 'acc/loss_cls:' in line:
                parts = line.split()
                epoch = int(parts[parts.index('Epoch(val)') + 1].split('[')[1].split(']')[0])
                loss = float(parts[parts.index('acc/loss_cls:') + 1])
                val_loss = loss
    return pd.DataFrame(data)

# def plot_loss(df):
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['iteration'], df['loss'], label='Loss')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.title('Loss over iterations')
#     plt.legend()
#     plt.savefig('loss_over_epochs_knuv.png')  # saves the plot to a file 

def plot_loss(df):
    plt.figure(figsize=(10, 6))
    # Calculate the rolling average over 10 iterations
    df['loss_smooth'] = df['train_loss'].rolling(window=10).mean()
    # Plot the smooth train loss and the val loss
    plt.plot(df['iteration'], df['loss_smooth'], label='Train loss')
    plt.plot(df['iteration'], df['val_loss'], label='Validation loss')
    # plt.plot(df['iteration'], df['loss_smooth'], label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.legend()
    plt.savefig('loss_over_epochs_unkv_val.png')  # saves the plot to a file

# log_file = '20231116_173410.log'  # replace with your log file path
log_file = sys.argv[1]
df = parse_log_file(log_file)
print(df)
plot_loss(df)