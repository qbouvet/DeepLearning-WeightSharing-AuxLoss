import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

"""
This file contains a number of utility methods useful for evaluating and inspecting models and architectures.
"""


def accuracy(model, X, Y, mini_batch_size=100):
    """
    Given a trained classification model, an input set and an output set,
    this method predicts the class for the inputs and compares the predictions to the true targets.
    It counts the errors and computes the accuracy of the model on the given sets.
    
    Parameters
    ----------
    model: pytorch Model
        trained classification model
    
    X: tensor
        input data (features)
        
    Y: tensor
        target data (class)
        
    mini_batch_size: int
        the predictions are done in batches and not at once, specifies batch size
    ----------
    
    Returns
    ----------
    accuracy: float
        accuracy value, between 0 and 1
    ----------
    """
    
    nb_errors = 0
    
    # Proceed in batches
    for b in range(0, X.size(0), mini_batch_size):
        
        # Predict batch
        output = model(X.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        
        # Count errors in batch
        for k in range(mini_batch_size):
            if Y[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    
    accuracy = 1 - nb_errors/X.shape[0]
    return accuracy


def train_model(model, X, Y, tX, tY,
                mini_batch_size=100, eta=1e-3, epochs=25,
                criterion=nn.CrossEntropyLoss(), opt=torch.optim.Adam):
    """
    This method trains a pytorch model on a given training set over the course
    of a specified number of epochs, evaluating its performance on a validation
    set at each epoch. It returns the training and validation history as a dict.
    """
    
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    optimizer = opt(model.parameters(), lr=eta)
    
    for e in range(epochs):
        sum_loss = 0
        
        with torch.no_grad():
            # Compute validation loss and accuracy
            val_acc = accuracy(model, tX, tY)
            history['val_acc'].append(val_acc)

            # Compute training accuracy w/o messing with training
            train_acc = accuracy(model, X, Y)
            history['train_acc'].append(train_acc)
            
            # Compute validation loss
            val_output = model(tX)
            val_loss = criterion(val_output, tY)
            history['val_loss'].append(val_loss.item())
            
        for b in range(0, X.size(0), mini_batch_size):
            # Classify batch, compute loss and perform backpropagation with parameter updates
            output = model(X.narrow(0, b, mini_batch_size))
            loss = criterion(output, Y.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss = sum_loss + loss.item()

                
        history['train_loss'].append(sum_loss)
        clear_output(wait=True)
        print('Epoch ' + str(e+1) + '/' + str(epochs))
    
    return history








def plot_hist(history, savefig=None):
    """
    This method takes a training history in the form of a python dictionary and plots
    the evolution of the model loss and accuracy over the epochs using matplotlib and seaborn.
    
    Parameters
    ----------
    history: dict
        training history, a dictionary containing both training and validation losses and accuracies
        
    savefig: str
        if the default 'None' value is overwritten, saves a JPG image of the plot with the given name
        in the results folder
    ----------
    """
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    # Loss subplot
    ax1.set_title('Evolution of cross-entropy loss over training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.yaxis.grid(True)
    sns.lineplot(range(len(history['train_loss'])) ,history['train_loss'], label='training', ax=ax1)
    sns.lineplot(range(len(history['val_loss'])) ,history['val_loss'], label='validation', ax=ax1)
    
    # Accuracy subplot
    ax2.set_title('Evolution of classification accuracy over training')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.yaxis.grid(True)
    sns.lineplot(range(len(history['train_acc'])) ,history['train_acc'], label='training', ax=ax2)
    sns.lineplot(range(len(history['val_acc'])) ,history['val_acc'], label='validation', ax=ax2)

    if savefig != None:
        plt.savefig('./results/' + savefig + '.jpg')
    plt.show()


def cv_k_fold(model, 
              X, Y, 
              tX, tY, 
              cv=5, 
              mini_batch_size=100, eta=1e-3, epochs=25,
              criterion=nn.CrossEntropyLoss(), opt=torch.optim.Adam):
    
    split_X = X.chunk(cv)
    split_Y = Y.chunk(cv)
    
    df_hist = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch'])
    
    for i in range(cv):
        m = model()
        rest_X = torch.cat([f for j, f in enumerate(split_X) if j!=i])
        rest_Y = torch.cat([f for j, f in enumerate(split_Y) if j!=i])
        fold_hist = train_model(m, split_X[i], split_Y[i], rest_X, rest_Y, 
                                mini_batch_size=mini_batch_size, eta=eta, epochs=epochs,
                                criterion=criterion, opt=opt)
        
        df_fold = pd.DataFrame(fold_hist)
        df_fold['epoch'] = range(df_fold.shape[0])
        df_fold['fold'] = [i for e in range(df_fold.shape[0])]
        df_hist = pd.concat([df_hist, df_fold])
        
    return df_hist


def end_of_hist_result(df_hist):
    df_avg = df_hist.groupby('epoch').mean()
    
    # Last epoch result
    last_dict = df_avg.loc[df_avg.index.max()][['train_loss', 'train_acc', 'val_loss', 'val_acc']].to_dict()
    last_dict['epoch'] = df_avg.index.max()
    
    # Best validation accuracy result
    df_avg = df_hist.groupby('epoch').mean()
    best_dict = df_avg.loc[df_avg.val_acc.idxmax()][['train_loss', 'train_acc', 'val_loss', 'val_acc']].to_dict()
    best_dict['epoch'] = df_avg.val_acc.idxmax()
    
    return pd.concat([pd.DataFrame(last_dict, index=['last']), pd.DataFrame(best_dict, index=['best'])])


def grid_search(eval_funct, params):
    param_tuples = list(product(*params.values()))
    
    df_grid_search = pd.DataFrame(columns=['mean_train_loss', 'mean_train_acc', 'mean_val_loss', 'mean_val_acc'])
    for p in param_tuples:
        df_params = eval_funct(p)
        res = end_of_hist_result(df_params).loc['last']
        df_grid_search.loc[str(p)] = [res['train_loss'], res['train_acc'], res['val_loss'], res['val_acc']]
        
    df_grid_search['rank_val_acc'] = df_grid_search.rank(method='first')['mean_val_acc'].astype(int)
    return df_grid_search
        
