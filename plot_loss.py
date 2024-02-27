import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle

def parse_log_file(log_file):
    data = {'iteration': [], 'train_loss': [], 'val_loss': [], 'val_acc_top1': [], 'val_acc_top5': [], 'train_acc_top1': [], 'train_acc_top5': []}
    iteration = 0
    val_loss = 0 # Initialize val_loss variable
    val_acc_top1 = 0.0
    val_acc_top5 = 0.0
    train_acc_top1 = 0.0
    train_acc_top5 = 0.0
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
                    data['val_acc_top1'].append(val_acc_top1)
                    data['val_acc_top5'].append(val_acc_top5)
                    data['train_acc_top1'].append(train_acc_top1)
                    data['train_acc_top5'].append(train_acc_top5)
            if 'Epoch(val)' in line and 'acc/loss_cls:' in line:
                parts = line.split()
                epoch = int(parts[parts.index('Epoch(val)') + 1].split('[')[1].split(']')[0])
                val_loss = float(parts[parts.index('acc/loss_cls:') + 1])
                val_acc_top1 = float(parts[parts.index('acc/top1:') + 1])
                val_acc_top5 = float(parts[parts.index('acc/top5:') + 1])
                train_acc_top1 = float(parts[parts.index('acc/top1_acc_train::') + 1])
                train_acc_top5 = float(parts[parts.index('acc/top5_acc_train::') + 1])

    # copy the first non zero value of val_loss, val_acc_top1, val_acc_top5, train_acc_top1, train_acc_top5 to the first zero values
    for i in range(0,5):
        data['val_loss'][i] = data['val_loss'][5]
        data['val_acc_top1'][i] = data['val_acc_top1'][5]
        data['val_acc_top5'][i] = data['val_acc_top5'][5]
        data['train_acc_top1'][i] = data['train_acc_top1'][5]
        data['train_acc_top5'][i] = data['train_acc_top5'][5]
    return pd.DataFrame(data)


def plot_loss(df, scenario, model):
    plt.figure(figsize=(10, 6))
    # Calculate the rolling average over 10 iterations
    df['loss_smooth'] = df['train_loss'].rolling(window=10).mean()
    # Plot the smooth train loss and the val loss
    plt.plot(df['iteration'], df['loss_smooth'], label='Train loss')
    plt.plot(df['iteration'], df['val_loss'], label='Validation loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs '+scenario+' '+model)
    plt.legend()
    plt.savefig('loss_over_epochs_'+scenario+'_'+model+'.png')  # saves the plot to a file

    window_size = 20  # Define the size of the window
    if 'train_acc_top1' in df.columns and 'train_acc_top5' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['iteration'], df['train_acc_top1'].rolling(window_size).mean(), label='Train accuracy top1', color='blue', zorder=3)
        plt.plot(df['iteration'], df['val_acc_top1'].rolling(window_size).mean(), label='Validation accuracy top1', color='red', zorder=2)
        plt.plot(df['iteration'], df['train_acc_top5'].rolling(window_size).mean(), label='Train accuracy top5', color='darkblue', alpha=0.8, zorder=1)
        plt.plot(df['iteration'], df['val_acc_top5'].rolling(window_size).mean(), label='Validation accuracy top5', color='darkred', alpha=0.8, zorder=0)
        plt.xlabel('Epocu')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs '+scenario+' '+model)
        plt.legend()
        plt.savefig('accuracy_over_epochs_'+scenario+'_'+model+'.png')

log_file = sys.argv[1]
scenario = sys.argv[2]
model = sys.argv[3]
df = parse_log_file(log_file)
print(df.head(10))
plot_loss(df, scenario, model)