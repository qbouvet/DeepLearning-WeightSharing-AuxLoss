#! /usr/bin/python3

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn


''' Given a trained classification model, an input set and an output set,
    this method predicts the class for the inputs and compares the 
    predictions to the true targets. It counts the errors and computes 
    the accuracy of the model on the given sets.
    
    Parameters : 
        model: pytorch Model    trained classification model
        X: tensor               input data (features)
        Y: tensor               target data (class)
        mini_batch_size: int    specifies predictions batch size
    
    Returns : 
        accuracy: float         accuracy value, between 0 and 1
'''
def accuracy_mt (model, X, Y, mini_batch_size=100) :
    nb_errors = 0
    
    # Proceed in batches
    for b in range(0, X.size(0), mini_batch_size):
        
        # Predict batch
        output = model(X.narrow(0, b, mini_batch_size))[0]
        _, predicted_classes = output.data.max(1)
        
        # Count errors in batch
        for k in range(mini_batch_size):
            if Y[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    
    accuracy = 1 - nb_errors/X.shape[0]
    return accuracy


''' Loss function for multitask-model
    Parameters : 
        [...]
        model       trained model so that : model : X -> ((pred_aux_1, pred_aux_2), pred_main)
        Y_aux       tuple (targ_aux_1, targ_aux_2)
'''
def loss_mt (model, X, Y_main, Y_aux,
             main_weight=0.5,
             loss_main=nn.CrossEntropyLoss(), loss_aux=nn.CrossEntropyLoss() ) :
    prediction_main, prediction_aux = model(X)
    # main loss
    mainloss = loss_main(prediction_main, Y_main)
    # aux loss
    auxloss1 = loss_aux(prediction_aux[0], Y_aux[0])
    auxloss2 = loss_aux(prediction_aux[1], Y_aux[1])
    auxloss = (auxloss1+auxloss2)/2.0
    # mix and return
    return main_weight*mainloss + (1-main_weight)*auxloss;
    
    

   
''' This method trains a pytorch model on a given training set over the 
    course of a specified number of epochs, evaluating its performance 
    on a validation set at each epoch. It returns the training and 
    validation history as a dict.
    Parameters
        Model       trained model that produces (mainPred, (auxpred1, auxpred2))
        [...]
        trAuxY      auxiliary results, shape (2, 1000)
'''
def train_model_mt (
        model, 
        trX, trY, trAuxY,
        valX, valY, valAuxY,
        eta=1e-3, 
        mini_batch_size=100, epochs=25,
        criterion=nn.CrossEntropyLoss(), 
        opt=torch.optim.Adam ) :
    
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    optimizer = opt(model.parameters(), lr=eta)
    
    for e in range(epochs):
        sum_loss = 0
        
        with torch.no_grad():
            # Compute validation loss and accuracy
            val_acc = accuracy_mt(model, valX, valY)
            history['val_acc'].append(val_acc)

            # Compute training accuracy w/o messing with training
            train_acc = accuracy_mt(model, trX, trY)
            history['train_acc'].append(train_acc)
            
            # Compute validation loss
            val_output = model(valX)
            val_loss = criterion(val_output[0], valY)
            history['val_loss'].append(val_loss.item())
            
        for b in range(0, trX.size(0), mini_batch_size):
            # Classify batch, compute loss and perform backpropagation with parameter updates
            #   REPLACED : 
            #xmain = model(X.narrow(0, b, mini_batch_size))
            #loss = criterion(output, Y.narrow(0, b, mini_batch_size))
            loss = loss_mt (
                model, 
                trX.narrow(0, b, mini_batch_size), 
                trY.narrow(0, b, mini_batch_size), 
                trAuxY[:,b:b+mini_batch_size],
                main_weight=0.01
            )
            model.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss = sum_loss + loss.item()

                
        history['train_loss'].append(sum_loss)
        #clear_output(wait=True)
        print('Epoch ' + str(e+1) + '/' + str(epochs))
    
    return history
    



