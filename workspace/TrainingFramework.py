#! /usr/bin/python3
import time

import torch
from torch import nn
from torch.autograd import Variable
from IPython.display import display, clear_output

import dlc_practical_prologue as prologue

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' Dataset that supports multi-task learning / auxiliary tasks
    Aims to decouple data from the operations performed on it by the 
    model.
    Parameters : 
        model           Here, model is such that model(X) returns 
                        (mainPrediction, (auxPrediction1, auxPrediction2))
        validationProp  Should be so that all sets have a size multiple
                        of batchsize        
'''
class TrainingFramework : 
    
    def __init__(self, 
    batchsize=100,
    mainLoss=nn.CrossEntropyLoss(), 
    auxLoss=nn.CrossEntropyLoss()) : 
        
        # Load data a first time (avoid a non-initialized framework)
        self.load_new_data()
        
        # Loss functions
        self.batchsize= batchsize
        self.mainLoss = mainLoss
        self.auxLoss = auxLoss
        print ("trn :", self.trnX.shape, "\ntrnY :", self.trnY.shape, "\ntrnAuxY :", self.trnAuxY.shape)
           
    def load_new_data(self):
        
        trnX, trnY, trnAuxY, tstX, tstY, tstAuxY = prologue.generate_pair_sets(1000)
        
        # Train data
        self.trnX = Variable(trnX)
        self.trnY = Variable(trnY)
        self.trnAuxY = Variable(trnAuxY).transpose(dim0=0, dim1=1)
        
        # Validation data TODO: implement
        self.valX = Variable(tstX)
        self.valY = Variable(tstY)
        self.valAuxY = Variable(tstAuxY).transpose(dim0=0, dim1=1)
        
    def __batch (self, index) : 
        #print ("__batch :", index)
        return self.trnX.narrow(0, index, self.batchsize), \
            self.trnY.narrow(0, index, self.batchsize), \
            self.trnAuxY[:,index:index+self.batchsize]
    
    def __accuracy (self, preds, Y) : 
        if self.auxLoss is not None : 
            preds = preds[0]
        #print ("  __accuracy : \n    preds :", preds.shape, "\n    Y :", Y.shape)
        nb_errors = 0
        _, predictedClasses = preds.data.max(1)
        for n in range(predictedClasses.size(0)):
            if Y[n] != predictedClasses[n] :
                nb_errors = nb_errors + 1
        # normalize and return
        accuracy = 1 - nb_errors/predictedClasses.shape[0]
        return accuracy
    
    def __loss (self, preds, Y) : 
        if self.auxLoss is None : 
            return self.mainLoss(preds, Y)
        else : 
            return self.mainLoss(preds[0], Y)
    
    ''' Here, auxY is a [2, ...] tensor and preds a [..., 10] tensor
    '''
    def __auxAccuracy (self, preds, auxY) : 
        nb_errors = 0
        aux1 = preds[0]
        aux2 = preds[1]
        aux1 = aux1.argmax(dim=1)
        aux2 = aux2.argmax(dim=1)
        #print ("  __auxAccu : aux1 :", aux1.shape, "aux2 :", aux2.shape, "\n              expected : ", auxY[0].shape)
        for n in range(aux1.shape[0]) :
            if aux1[n] != auxY[0,n] :
                nb_errors = nb_errors + 1
            if aux2[n] != auxY[1,n] :
                nb_errors = nb_errors + 1
        # normalize and return
        accuracy = 1 - (nb_errors/(2*aux1.shape[0]))
        return accuracy

    ''' Here, auxY is a [2, ...] tensor and preds a [..., 10] tensor
    '''
    def __auxLoss (self, preds, auxY) : 
        if self.auxLoss is None : 
            print ("can't call __auxloss() is self.auxloss is None")
            exit
        aux1, aux2 = preds[0], preds[1]
        #print ("  __auxloss :\n    classes :", aux1.shape, aux2.shape)
        #print ("    Expected :", auxY[0].shape, auxY[1].shape)
        l1 = self.auxLoss(aux1, auxY[0])
        l2 = self.auxLoss(aux2, auxY[1])
        return (l1 + l2)/2.0
        
    
    def trainModel(self, 
    model, 
    epochs=50, eta=1e-3,
    auxLossWeight=0.5,
    opt=torch.optim.Adam) : 
        
        # Measure training time
        start_time = time.perf_counter()
        
        self.auxWeight = auxLossWeight
        optimizer = opt(model.parameters(), lr=eta)
        history = {
            'train_loss':[], 'train_acc':[], 
            'aux_train_loss':[], 'aux_train_acc':[], 
            'val_loss':[], 'val_acc':[],
            'aux_val_loss':[], 'aux_val_acc':[]
        }
        
        for e in range(epochs):
            sum_loss = 0
            
            with torch.no_grad():
                model.train(False)
                # validation loss & accuracy
                valPreds = model(self.valX) # NB contains both main and aux predictions
                history['val_acc'].append(
                    self.__accuracy(valPreds, self.valY) )
                history['val_loss'].append(
                    self.__loss(valPreds, self.valY).item() )
                # Training accuracy w/o messing with training
                trnPreds = model (self.trnX)
                history['train_acc'].append(
                    self.__accuracy(trnPreds, self.trnY) )
                history['train_loss'].append(
                    self.__loss(trnPreds, self.trnY).item() )
                if self.auxLoss is not None : 
                    # Validation auxiliary loss / accuracy
                    history['aux_val_acc'].append(
                        self.__auxAccuracy(valPreds[1], self.valAuxY) )
                    history['aux_val_loss'].append(
                        self.__auxLoss(valPreds[1], self.valAuxY).item() )
                    # Training auxiliary accuracy
                    history['aux_train_acc'].append(
                        self.__auxAccuracy(trnPreds[1], self.trnAuxY) )
                    history['aux_train_loss'].append(   # is this correct here ? 
                        self.__auxLoss(trnPreds[1], self.trnAuxY).item() )
            
            model.train(True)
            for index in range(0, self.trnX.size(0), self.batchsize):
                # predict
                X, Y, auxY = self.__batch(index)
                preds = model(X)
                # compute loss
                mainLoss = self.__loss (preds, Y) 
                if self.auxLoss is not None :
                    auxPreds = preds[1]
                    auxLoss = self.__auxLoss (auxPreds, auxY) 
                    mixedLoss = self.auxWeight * auxLoss + (1-self.auxWeight) * mainLoss
                else :
                    mixedLoss = mainLoss
                # gradient step
                model.zero_grad()
                mixedLoss.backward()
                optimizer.step()
                sum_loss += mixedLoss.item()
                
            if self.auxLoss == None:
                df_result = pd.DataFrame(history, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
            else:
                df_result = pd.DataFrame(history)
            df_result['epoch'] = range(df_result.shape[0])
            
            clear_output(wait=True)
            display("Epoch " + str(e+1) + '/' + str(epochs))
            end_time = time.perf_counter()
                
        return df_result, end_time - start_time

    
    def validateModel(self, 
    model_funct, 
    nb_iter=10, eta=1e-3, epochs=50, 
    auxLossWeight=0.5, opt=torch.optim.Adam):


        df_hist = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc',
            'aux_train_loss', 'aux_train_acc', 'aux_val_loss', 'aux_val_acc', 'epoch'])

        times = []
        for i in range(nb_iter):
            
            # Load new data and init model
            self.load_new_data()
            m = model_funct()
            
            hist, train_time = self.trainModel(m, 
                                    epochs=epochs, eta=eta,
                                    auxLossWeight=auxLossWeight, opt=opt)

            times.append(train_time)
            df_fold = pd.DataFrame(hist)
            df_fold['epoch'] = range(df_fold.shape[0])
            df_fold['iter'] = [i for e in range(df_fold.shape[0])]
            df_hist = pd.concat([df_hist, df_fold])
        
        return df_hist, times

    
''' TODO: document method
''' 
def end_of_hist_result(df_hist):
    df_avg = df_hist.groupby('epoch').mean()
    cols = ['train_loss', 'train_acc', 'val_loss', 'val_acc',
            'aux_train_loss', 'aux_train_acc', 'aux_val_loss', 'aux_val_acc']
    
    # Last epoch result
    last_dict = df_avg.loc[df_avg.index.max()][cols].to_dict()
    last_dict['epoch'] = df_avg.index.max()
    
    # Best validation accuracy result
    df_avg = df_hist.groupby('epoch').mean()
    best_dict = df_avg.loc[df_avg.val_acc.idxmax()][cols].to_dict()
    best_dict['epoch'] = df_avg.val_acc.idxmax()
    
    return pd.concat([pd.DataFrame(last_dict, index=['last']), pd.DataFrame(best_dict, index=['best'])])
    

''' This method takes a training history in the form of a python 
    dictionary and plots the evolution of the model loss and accuracy 
    over the epochs using matplotlib and seaborn.
    Parameters : 
        history     dict containing data series 
        savefig     str, if the default 'None' value is overwritten, 
                    saves a JPG image of the plot with the given name 
                    in the results folder
'''
def plot_hist(df_history, savefig=None, auxloss=True):
    
    if auxloss:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    else:
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
    
    if auxloss:
        # Aux Loss subplot
        ax3.set_title('Auxiliary Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.yaxis.grid(True)
        sns.lineplot(x='epoch', y='aux_train_loss', data=df_history, label='training', ax=ax3)
        sns.lineplot(x='epoch', y='aux_val_loss', data=df_history, label='validation', ax=ax3)

        # Accuracy subplot
        ax4.set_title('Auxiliary task accuracy')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy')
        ax4.yaxis.grid(True)
        sns.lineplot(x='epoch', y='aux_train_acc', data=df_history, label='training', ax=ax4)
        sns.lineplot(x='epoch', y='aux_val_acc', data=df_history, label='validation', ax=ax4)
    
    display(end_of_hist_result(df_history))

    if savefig != None:
        plt.savefig('./results/' + savefig + '.jpg')
    plt.show()

