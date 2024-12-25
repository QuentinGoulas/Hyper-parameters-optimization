# model.py
import torch
import torch.nn as nn

# Hyperparameters initialization
in_channels = 1
conv1_out_channels = 6
conv1_kernel_size = 5

conv2_out_channels = 16
conv2_kernel_size = 5

conv3_out_channels = 120
conv3_kernel_size = 5

fc1_out_features = 84
fc2_out_features = 10

pool_kernel_size = 2
pool_stride = 2

init_hyperparam = {
    'C1_chan' : conv1_out_channels,
    'C3_chan' : conv2_out_channels,
    'C5_chan' : conv3_out_channels,
    'C1_kernel' : conv1_kernel_size,
    'C3_kernel' : conv2_kernel_size,
    'C5_kernel' : conv3_kernel_size,
    'F6' : fc1_out_features
}

class LeNet5(nn.Module):
    def __init__(self, hyperparam={}):
        super(LeNet5, self).__init__()

        hyperparam = update_hyperparam(init_hyperparam,hyperparam)
        self.hyperparam = hyperparam
        
        # First convolutional layer (C1)
        self.conv1 = nn.Conv2d(in_channels, hyperparam['C1_chan'], kernel_size=hyperparam['C1_kernel'])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        # Second convolutional layer (C3)
        self.conv2 = nn.Conv2d(hyperparam['C1_chan'], hyperparam['C3_chan'], kernel_size=hyperparam['C3_kernel'])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        # Third convolutional layer (C5)
        self.conv3 = nn.Conv2d(hyperparam['C3_chan'], hyperparam['C5_chan'], kernel_size=hyperparam['C5_kernel'])
        self.relu3 = nn.ReLU()
        
        # Fully connected layers (F6 and OUTPUT)
        self.fc1 = nn.Linear(hyperparam['C5_chan'], hyperparam['F6'])
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(hyperparam['F6'], fc2_out_features)  # 10 classes output
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        
        # Flatten the feature maps
        x = x.view(-1, self.hyperparam['C5_chan'])
        
        # Fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        
        return x

def update_hyperparam(hyperparam, mods):
    '''
    A function to update the hyperparameter values of the module
    '''
    for hp in mods.keys():
        hyperparam[hp] = mods[hp]

    return hyperparam