#! /usr/bin/python3

helpstr = """

    plot history
    
"""

import time
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Open first file
def openDf (path) :
    # Title
    tmp = open(path, 'r')
    tmpstr = tmp.readline()
    if tmpstr[:6] == "title:" : 
        title = tmpstr[6:]
        skip = 1
    else :
        title = ""
        skip = 0
    tmp.close()
    # Data
    df = pd.read_csv(path, skiprows=skip)
    df.drop(0, inplace=True)
    df.rename(columns=lambda x : x.lower(), inplace=True)
    print (title, "-->", df.columns)
    return title, df
    

def plot_hist(df_history, savefig=None, auxloss=True):
    
    '''    
    if auxloss:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    else:
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    '''
    
    if auxloss:
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        # Aux Loss subplot
        ax1.set_title('Auxiliary Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.yaxis.grid(True)
        sns.lineplot(x='epoch', y='aux_train_loss', data=df_history, label='training', ax=ax1)
        sns.lineplot(x='epoch', y='aux_val_loss', data=df_history, label='validation', ax=ax1)
        # Accuracy subplot
        ax2.set_title('Auxiliary task accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.yaxis.grid(True)
        sns.lineplot(x='epoch', y='aux_train_acc', data=df_history, label='training', ax=ax2)
        sns.lineplot(x='epoch', y='aux_val_acc', data=df_history, label='validation', ax=ax2)
        plt.show(block=False)
        
        if savefig != None:
            plt.savefig('./results/' + savefig + '.jpg')
        
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    # Loss subplot
    ax1.set_title('Evolution of cross-entropy loss over training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.yaxis.grid(True)
    sns.lineplot(x='epoch', y='train_loss', data=df_history, label='training', ax=ax1)
    sns.lineplot(x='epoch', y='val_loss', data=df_history, label='validation', ax=ax1)
    
    # Accuracy subplot
    ax2.set_title('Evolution of classification accuracy over training')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.yaxis.grid(True)
    sns.lineplot(x='epoch', y='train_acc', data=df_history, label='training', ax=ax2)
    sns.lineplot(x='epoch', y='val_acc', data=df_history, label='validation', ax=ax2)
    plt.show()

    if savefig != None:
        plt.savefig('./results/' + savefig + '.jpg')


print (sys.argv[1])
title, df = openDf (sys.argv[1])
plot_hist(df)
