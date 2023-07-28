import torch.nn as nn
import torch
from torch import Tensor

class SiameseBackboneNetwork():
    # def __init__
    # AQUI VAI TODA A CAMADA CNN1
    # TODO
    # self.siamese_backbone_forward()
    
    pass

class SiameseNetwork(nn.Module):
    def __init__(self, img_height: int, img_width: int, num_output: int = 256):
        super(SiameseNetwork, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.num_output =  num_output
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, self.num_output, kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3)
        )
        
        # Compute the input size for the fully connected layers based on the output shape of the last convolutional layer
        self.fc_input_size = self._get_fc_input_size()

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,1)
        )
        
    def _get_fc_input_size(self):
        # Calculate the output shape of the last convolutional layer
        dummy_input = torch.zeros(1, 1, self.img_height, self.img_width)
        with torch.no_grad():
            dummy_output = self.cnn1(dummy_input)
        _, channels, height, width = dummy_output.shape

        # Calculate the input size for the fully connected layers
        return channels * height * width
  
    def forward_once(self, x):
        # Forward pass 
        output: Tensor = self.cnn1(x)
        
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1: Tensor, input2: Tensor):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
