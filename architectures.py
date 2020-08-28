import torch
from torch import nn


"""
This file contains the three neural net architectures evaluated in the context of the first miniproject.
Each architecture follows the same high-level structural approach, with two subnets forming two 'pipes',
each accepting a single image as input, and a subsequent subnet acting as a final classifier for the task.
The models allow the two pipes to share parameters when weight sharing is active, and always return the
outputs of the pipes on top of the final network output, in order to enable the usage of an auxiliary task
during training.

The file contains the following models:
- ShallowFullyConnectedArchitecture
- DeepFullyConnectedArchitecture
- ConvolutionalArchitecture

authors:
    Tobia Albergoni, Quentin Bouvet, Matteo Yann Feo
"""


"""
This module implements an architecture where each 'pipe' is composed by a shallow net with only one hidden
layer of 500 units, and the final classifier is a shallow net aswell, with 50 hidden units.
"""
class ShallowFullyConnectedArchitecture(nn.Module):
    
    def __init__(self, weight_sharing=False):
        super(ShallowFullyConnectedArchitecture, self).__init__()
        self.input_size = 196
        self.output_pipe_size = 10
        self.output_classifier_size = 2        
        self.weight_sharing = weight_sharing
        
        # Create first pipe
        self.pipe1 = self.__create_pipe()

        # Depending on WS, create second pipe or use the same
        if self.weight_sharing:
            self.pipe2 = self.pipe1
        else:
            self.pipe2 = self.__create_pipe()
            
        # Create final classifier
        self.classifier = self.__create_classifier()
        

    def forward(self, x):        
        x1 = x[:,0:1,:,:]
        x2 = x[:,1:2,:,:]
        
        x1_pipe = self.pipe1(x1.view(-1, self.input_size))
        x2_pipe = self.pipe2(x2.view(-1, self.input_size))
        
        x_main = torch.cat([x1_pipe, x2_pipe], dim=1)
        x_main = self.classifier(x_main)
        
        return x_main, (x1_pipe, x2_pipe)
    
    
    def count_parameters(self): 
        nb_params = 0
        for parameter in self.parameters():
            nb_params = nb_params + parameter.numel()
        return nb_params    
    
    
    def __create_pipe(self): 
        return nn.Sequential(
                nn.Linear(self.input_size, 500),
                nn.ReLU(True),
                nn.Linear(500, self.output_pipe_size)
            )
    
    
    def __create_classifier(self):
        return nn.Sequential(
                nn.Linear(self.output_pipe_size * 2, 50),
                nn.ReLU(True),
                nn.Linear(50, self.output_classifier_size)
            )

    
"""
This module implements an architecture where each 'pipe' is composed by a deep fully connected net
with four hidden layers. The final classifier is also deeper than the other models, with two hidden
layers instead of one.
"""
class DeepFullyConnectedArchitecture(nn.Module):
    
    def __init__(self, weight_sharing=False):  
        super(DeepFullyConnectedArchitecture, self).__init__()
        self.input_size = 196
        self.output_pipe_size = 10
        self.output_classifier_size = 2    
        self.weight_sharing = weight_sharing
        
        # Create first pipe
        self.pipe1 = self.__create_pipe()

        # Depending on WS, create second pipe or use the same
        if self.weight_sharing:
            self.pipe2 = self.pipe1
        else:
            self.pipe2 = self.__create_pipe()
            
        # Create final classifier
        self.classifier = self.__create_classifier()
        
    
    def forward(self, x):        
        x1 = x[:,0:1,:,:]
        x2 = x[:,1:2,:,:]
        
        x1_pipe = self.pipe1(x1.view(-1, self.input_size))
        x2_pipe = self.pipe2(x2.view(-1, self.input_size))
        
        x_main = torch.cat([x1_pipe, x2_pipe], dim=1)
        x_main = self.classifier(x_main)
        
        return x_main, (x1_pipe, x2_pipe)
    
    
    def count_parameters(self): 
        nb_params = 0
        for parameter in self.parameters():
            nb_params = nb_params + parameter.numel()
        return nb_params
    

    def __create_pipe(self):
        return nn.Sequential(
                nn.Linear(self.input_size, 300),
                nn.ReLU(True),
                nn.Linear(300, 100),
                nn.ReLU(True),
                nn.Linear(100, 70),
                nn.ReLU(True),
                nn.Linear(70, 30),
                nn.ReLU(True),
                nn.Linear(30, 20),
                nn.ReLU(True),
                nn.Linear(20, self.output_pipe_size),
            )
    
    
    def __create_classifier(self):
        return nn.Sequential(
                nn.Linear(self.output_pipe_size * 2, 30),
                nn.ReLU(True),
                nn.Linear(30, 20),
                nn.ReLU(True),
                nn.Linear(20, 10),
                nn.ReLU(True),
                nn.Linear(10, self.output_classifier_size)
            )


"""
This module implements an architecture where each 'pipe' is composed by a convolutional net
with a structure inspired by classic ConvNet approaches to image classification. The final
classifier is a shallow net with a single hidden layer of 32 units.
"""
class ConvolutionalArchitecture(nn.Module):
    
    def __init__(self, weight_sharing=False):
        super(ConvolutionalArchitecture, self).__init__()
        self.input_size = 196
        self.conv_out_dim = 64
        self.output_pipe_size = 10
        self.output_classifier_size = 2   
        self.weight_sharing = weight_sharing
        
        # Create first pipe
        self.pipe1 = self.__create_pipe()
        self.pipe_classifier_1 = self.__create_pipe_classifier()

        # Depending on WS, create second pipe or use the same
        if self.weight_sharing:
            self.pipe2 = self.pipe1
            self.pipe_classifier_2 = self.pipe_classifier_1
        else:
            self.pipe2 = self.__create_pipe()
            self.pipe_classifier_2 = self.__create_pipe_classifier()
        
        # Create final classifier
        self.classifier = self.__create_classifier()
        

    def forward(self, x):        
        x1 = x[:,0:1,:,:]
        x2 = x[:,1:2,:,:]
        
        x1_pipe = self.pipe_classifier_1(self.pipe1(x1).view(-1, self.conv_out_dim))
        x2_pipe = self.pipe_classifier_2(self.pipe2(x2).view(-1, self.conv_out_dim))
        
        x_main = torch.cat([x1_pipe, x2_pipe], dim=1)
        x_main = self.classifier(x_main)
        
        return x_main, (x1_pipe, x2_pipe)
    
    
    def count_parameters(self): 
        nb_params = 0
        for parameter in self.parameters():
            nb_params = nb_params + parameter.numel()
        return nb_params
        
        
    def __create_pipe(self):
        return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(16, 32, kernel_size=3),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=4, stride=4)   
            )
    
    
    def __create_pipe_classifier(self): 
        return nn.Sequential(
                nn.Linear(self.conv_out_dim, 100),
                nn.ReLU(True),
                nn.Linear(100, self.output_pipe_size)
            )
    
    
    def __create_classifier(self):
        return nn.Sequential(
                nn.Linear(self.output_pipe_size * 2, 32),
                nn.ReLU(True),
                nn.Linear(32, self.output_classifier_size)
            )