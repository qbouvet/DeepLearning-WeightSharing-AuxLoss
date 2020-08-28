#! /usr/bin/python3
import time
import torch

import dlc_practical_prologue as prologue

from torch import nn
from torch.autograd import Variable
from utils import Logger, validation_results

"""
This class implements all the training and validation procedures necessary to train and
validate pytorch models that adhere to the high level architecture of the models in architectures.py.
It has great flexibility, allowing to change a number of settings and parameters on the same object,
greatly reducing code replication and the verbosity of the final test.py file.
"""
class TrainingFramework: 
    
    """
    The constructor instantiates the Logger instance responsible of logging the actions of the object, and then
    initializes a number of important fields, from the losses to be optimized, to the optimizer itself and the size
    of the training batches. The method additionally loads data a first time to ensure the object is always in a valid state.
    
    Parameters:
    ----------
    mini_batch_size: int, default 100
        size of training batches of examples
        
    optimizer: torch optimizer from torch.optim, default Adam
        optimizer to use during training
        
    log_level: string, default 'info'
        level for logger (either 'debug', 'info' or 'warn')
        
    main_loss: loss function from torch.nn, default CrossEntropyLoss
        loss function for the main classification task
        
    aux_loss: loss function from torch.nn, default CrossEntropyLoss
        loss function for the auxiliary task (single image classification)
        
    aux_task: boolean, default False
        whether the framework should use the auxiliary loss function during training or not
        
    aux_task_weight: float, default 0.5
        weight that has to be given to the auxiliary task (between 0 and 1) if it is enabled
    ----------
    """
    def __init__(self, mini_batch_size=100, optimizer=torch.optim.Adam, log_level='info', 
                 main_loss=nn.CrossEntropyLoss(), aux_loss=nn.CrossEntropyLoss(), aux_task=False, aux_task_weight=0.5): 
        
        # Init logger
        self.log = Logger(level=log_level)
        
        # Load data a first time (avoid a non-initialized framework)
        self.load_new_data()
        
        # Init attributes with defaults
        self.set_main_loss(main_loss)
        self.set_aux_loss(aux_loss)
        self.enable_aux_task(aux_task, aux_task_weight)
        self.set_mini_batch_size(mini_batch_size)
        self.set_optimizer(optimizer)
            
    
    def set_main_loss(self, loss_function):
        self.main_loss = loss_function
        self.log.info('Main loss function : %s' % loss_function)
        
    
    def set_aux_loss(self, loss_function):
        self.aux_loss = loss_function
        self.log.info('Auxiliary loss function : %s' % loss_function)
        
    
    def enable_aux_task(self, enable, aux_task_weight=0.5):
        self.aux_task = enable
        self.aux_task_weight = aux_task_weight
        self.log.info('Auxiliary task : %s' % enable)
    
    
    def set_mini_batch_size(self, mini_batch_size):
        self.mini_batch_size = mini_batch_size
        self.log.info('Mini batch size : %s examples' % mini_batch_size)
        
    
    def set_optimizer(self, opt):
        self.optimizer = opt
        self.log.info('Optimizer : %s' % opt)
        
    
    def set_log_level(self, log_level):
        self.log_level = level
           
    
    """
    This method uses the prologue function to load a fresh training and test (validation) set
    from MNIST data. The loaded data is stored as object attributes, which will be used for training
    models until new data is loaded and replaces the old one.
    """
    def load_new_data(self):
        
        trnX, trnY, trnAuxY, valX, valY, valAuxY = prologue.generate_pair_sets(1000)
        
        # Train data
        self.trnX = Variable(trnX)
        self.trnY = Variable(trnY)
        self.trnAuxY = Variable(trnAuxY).transpose(dim0=0, dim1=1)
        
        # Validation data
        self.valX = Variable(valX)
        self.valY = Variable(valY)
        self.valAuxY = Variable(valAuxY).transpose(dim0=0, dim1=1)
        
        #Log
        self.log.info('Load new data')
        self.log.debug('Data:\n\ttrnX: %s\ttrnY: %s\ttrnAuxY: %s\n\tvalX: %s\tvalY: %s\tvalAuxY: %s' % 
                      (self.trnX.shape, self.trnY.shape, self.trnAuxY.shape, self.valX.shape, self.valY.shape, self.valAuxY.shape))
    
    
    
    """
    This method returns a batch of training data, staring from a particular index
    """
    def __trn_batch(self, index): 
        return self.trnX.narrow(0, index, self.mini_batch_size), \
            self.trnY.narrow(0, index, self.mini_batch_size), \
            self.trnAuxY[:,index:index+self.mini_batch_size]
        
    
    """
    This method computes the accuracy of a set of predictions on the primary
    classification task compared to the true targets.
    """
    def __main_accuracy(self, main_pred, Y): 
        self.log.debug("__main_accuracy: \n\tmain_pred: %s\tY: %s" % (main_pred.shape, Y.shape))

        nb_errors = 0
        _, predictedClasses = main_pred.data.max(1)
        
        # Compute errors
        for n in range(predictedClasses.size(0)):
            if Y[n] != predictedClasses[n] :
                nb_errors = nb_errors + 1
                
        accuracy = 1 - nb_errors/predictedClasses.shape[0]
        return accuracy
    
    
    """
    This method uses the main loss function to compute the loss of a set of predictions
    on the primary classification task on the true targets.
    """
    def __main_loss(self, main_pred, Y): 
        self.log.debug("__main_loss: \n\tmain_pred: %s\tY: %s" % (main_pred.shape, Y.shape))
        return self.main_loss(main_pred, Y)

    
    """
    This method computes the accuracy of a set of predictions on the auxiliary
    classification task compared to the true class targets of images.
    """
    def __aux_accuracy(self, aux_pred, auxY): 
        if not self.aux_task:
            self.log.warn('Cannot call __aux_accuracy() when auxiliary task is not enabled')
            exit
              
        # Get class predictions
        nb_errors = 0
        aux1 = aux_pred[0].argmax(dim=1)
        aux2 = aux_pred[1].argmax(dim=1)
        self.log.debug("__aux_accuracy: \n\taux1: %s\taux1: %s\tauxY: %s" % (aux1.shape, aux2.shape, auxY.shape))
        
        # Compute errors
        for n in range(aux1.shape[0]) :
            if aux1[n] != auxY[0,n] :
                nb_errors = nb_errors + 1
            if aux2[n] != auxY[1,n] :
                nb_errors = nb_errors + 1
                
        accuracy = 1 - (nb_errors/(2*aux1.shape[0]))
        return accuracy

    
    """
    This method uses the auxiliary loss function to compute the loss of a set of predictions
    on the single image classification task on the true class targets of images.
    """
    def __aux_loss(self, aux_pred, auxY): 
        if not self.aux_task:
            self.log.warn('Cannot call __aux_loss() when auxiliary task is not enabled')
            exit
            
        self.log.debug("__aux_loss: \n\taux_pred_1: %s\tauxY_1: %s" % (aux_pred[0].shape, auxY[0].shape))
            
        l1 = self.aux_loss(aux_pred[0], auxY[0])
        l2 = self.aux_loss(aux_pred[1], auxY[1])
        return (l1 + l2)/2.0
    
    
    """
    This method takes a model during training and predicts auxiliary and primary targets
    for both training and validation data, recording the loss and accuracy of the model on everything
    in a dictionnary structure that holds the values at each epoch. Autograd is disabled during the
    execution of this method.
    
    Parameters
    ----------
    history: dict
        contains epoch training and validation accuracy and losses for both primary and auxiliary tasks
        
    model: pytorch module
        model to be used for prediction
    ----------
    """
    def __update_history(self, history, model):
        
        with torch.no_grad():
            # Disable training mode
            model.train(False)
            
            # Training loss & accuracy
            trnPreds = model(self.trnX)
            history['train_acc'].append(self.__main_accuracy(trnPreds[0], self.trnY))
            history['train_loss'].append(self.__main_loss(trnPreds[0], self.trnY).item())
            
            # Validation loss & accuracy
            valPreds = model(self.valX)
            history['val_acc'].append(self.__main_accuracy(valPreds[0], self.valY))
            history['val_loss'].append(self.__main_loss(valPreds[0], self.valY).item())
            
                
            if self.aux_task: 
                # Training auxiliary loss & accuracy
                history['aux_train_acc'].append(self.__aux_accuracy(trnPreds[1], self.trnAuxY))
                history['aux_train_loss'].append(self.__aux_loss(trnPreds[1], self.trnAuxY).item())
                
                # Validation auxiliary loss & accuracy
                history['aux_val_acc'].append(self.__aux_accuracy(valPreds[1], self.valAuxY))
                history['aux_val_loss'].append(self.__aux_loss(valPreds[1], self.valAuxY).item())
                    
    
    """
    This method takes a pytorch model and trains it for a given number of epochs on the current framework data.
    It uses all the current framework settings, including the auxiliary task if enabled. Training is done in 
    batches and the performance of the model on the training and validation data is recorded for each epoch.
    Additionally, the training time is measured.
    
    Parameters
    ----------
    model: pytorch module
        model to train
    
    epochs: int, default 50
        number of training epochs
        
    eta: float, default 1e-3
        learning rate
    ----------
    """
    def train_model(self, model, epochs=50, eta=1e-3): 
        
        # Measure training time
        start_time = time.perf_counter()
        
        # Init optimizer
        optimizer = self.optimizer(model.parameters(), lr=eta)
        
        # Init training history
        history = {
            'train_loss':[], 'train_acc':[], 
            'aux_train_loss':[], 'aux_train_acc':[], 
            'val_loss':[], 'val_acc':[],
            'aux_val_loss':[], 'aux_val_acc':[]
        }
        
        # Logging
        self.log.warn("Training model %s" % model.__class__.__name__)
        self.log.debug("Model structure: %s" % model)
        
        # Train model
        for e in range(epochs):
            # Logging
            end_char = '\n' if (e == epochs-1) else '\r'
            print("\tProgress[" + ("#" * (e+1)) + ("-" * (epochs-e-1)) + "]", end=end_char)
            self.__update_history(history, model)
            
            sum_loss = 0
            model.train(True)
            
            # Iterate over training data in batches
            for index in range(0, self.trnX.size(0), self.mini_batch_size):
                # Predict
                X, Y, auxY = self.__trn_batch(index)
                preds = model(X)
                
                # Compute loss(es)
                mainLoss = self.__main_loss (preds[0], Y) 
                if self.aux_task:
                    auxLoss = self.__aux_loss(preds[1], auxY) 
                    mixedLoss = self.aux_task_weight * auxLoss + (1-self.aux_task_weight) * mainLoss
                else :
                    mixedLoss = mainLoss
                    
                # Gradient step
                model.zero_grad()
                mixedLoss.backward()
                optimizer.step()
                sum_loss += mixedLoss.item()
                
    
        end_time = time.perf_counter()
        history['time'] = [end_time-start_time]
        self.log.warn("Results: time = %s\tval_acc = %s" % (history['time'][-1], history['val_acc'][-1]))
        return history

    
    """
    This method takes the initialization function of a model and trains the model a given number of times,
    each time with newly loaded data and with the parameters reinitialized. The training history of each
    training is recorded, and the method ultimately returns a list of training histories.
    
    Parameters
    ----------
    model_funct: function
        initialization function of the model, should not require parameters (externalize parametrization of models
        through the usage of lambda functions)
        
    nb_iter: int, default 10
        number of time the model has to be trained
    
    epochs: int, default 50
        number of training epochs for each training run
        
    eta: float, default 1e-3
        learning rate
    ----------
    """
    def validate_model(self, model_funct, nb_iter=10, eta=1e-3, epochs=50):
        
        print()
        self.log.warn("Validate model : %s\n\tNumber of parameters: %s\n\tWeight sharing: %s\n\tAuxiliary loss: %s" %
                      (model_funct().__class__.__name__, model_funct().count_parameters(), model_funct().weight_sharing, self.aux_task))

        histories = []
        
        # Train nb_iter models and record histories
        for i in range(nb_iter):
            # Load new data and init model
            self.load_new_data()
            m = model_funct()
            
            self.log.warn("Validation run %s/%s" % (i+1, nb_iter))
            
            # Train model and record history
            hist = self.train_model(m, epochs=epochs, eta=eta)
            histories.append(hist)
            
        # Compute and display average model performance
        mean_val_acc, std_val_acc, mean_time = validation_results(histories)
        self.log.warn("Validation results:\n\tMean validation accuracy: %s +- %s\n\tMean training time: %s" %
                     (mean_val_acc, std_val_acc, mean_time))
        
        print()
        return histories


